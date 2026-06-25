from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.config import cfg_device, cfg_first, cfg_int, load_config, resolve_device
from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.prediction_schema import (
    prediction_heatmap_payload,
    resolve_prediction_heatmaps,
)
from seer_ad_v2.models.feature_prior import feature_prior_scores, load_feature_prior_components
from seer_ad_v2.utils.io import load_checkpoint


MAP_SOURCES = ["heatmaps", "score_heatmaps", "final_heatmaps", "feature", "feature_raw", "feature_raw_distance", "feature_raw_cosine"]
MODES = ["raw", "center", "relu_center", "zscore", "relu_zscore", "robust_zscore", "relu_robust_zscore", "ratio", "log_ratio"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize train-normal calibrated pixel heatmaps from a saved feature-first run.")
    p.add_argument("--source-run-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--category", default=None)
    p.add_argument("--feature-prior-checkpoint", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--train-max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--source-map", choices=MAP_SOURCES, default="feature_raw")
    p.add_argument("--normal-source-map", choices=MAP_SOURCES, default=None)
    p.add_argument("--modes", default="relu_zscore,relu_robust_zscore,ratio,log_ratio")
    p.add_argument("--device", default=None)
    p.add_argument("--allow-random-feature-weights", action="store_true")
    p.add_argument("--ablation-prefix", default="feature_pixelcal")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _run_args(run_dir: Path) -> dict[str, Any]:
    payload = _load_json(run_dir / "run_args.json")
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    return args if isinstance(args, dict) else {}


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _image_npz_heatmaps(run_dir: Path, count: int, key: str) -> np.ndarray:
    maps: list[np.ndarray] = []
    for idx in range(count):
        npz_path = run_dir / "images" / f"{idx:05d}" / "residual_heatmap.npz"
        data = np.load(npz_path)
        if key not in data:
            raise KeyError(f"{npz_path} does not contain heatmap key '{key}'. Available: {list(data.keys())}")
        maps.append(np.asarray(data[key], dtype=np.float32))
    return np.stack(maps).astype(np.float32)


def _source_maps(run_dir: Path, pred: np.lib.npyio.NpzFile, source: str) -> np.ndarray:
    if source in pred.files:
        return np.asarray(pred[source], dtype=np.float32)
    if source in {"heatmaps", "score_heatmaps", "final_heatmaps"}:
        raise KeyError(f"{run_dir / 'predictions.npz'} does not contain '{source}'")
    return _image_npz_heatmaps(run_dir, len(pred["labels"]), source)


def _feature_output_map(output: Any, source: str) -> np.ndarray:
    if source in {"heatmaps", "feature"}:
        return output.heatmaps
    if source == "feature_raw":
        return output.raw_heatmaps
    if source == "feature_raw_distance":
        return output.raw_distance_heatmaps
    if source == "feature_raw_cosine":
        return output.raw_cosine_heatmaps
    if source == "score_heatmaps":
        return output.raw_cosine_heatmaps
    if source == "final_heatmaps":
        return output.raw_heatmaps
    raise ValueError(f"Cannot compute train-normal calibration for source map: {source}")


def _normal_stack(args: argparse.Namespace, cfg: dict[str, Any], category: str, feature_ckpt: Path, image_size: int, device: str) -> np.ndarray:
    checkpoint = load_checkpoint(feature_ckpt)
    feature_prior, extractor, layers = load_feature_prior_components(
        checkpoint,
        device,
        allow_random_weights=args.allow_random_feature_weights,
    )
    train_ds = build_dataset(
        cfg_first(cfg, ("dataset.name",), "mvtec"),
        cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD"),
        category,
        "train",
        image_size,
        max_samples=args.train_max_samples,
    )
    batch_size = int(args.batch_size or cfg_int(cfg, ("feature_prior.batch_size", "training.batch_size"), 8))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=cfg_int(cfg, ("dataset.num_workers",), 0))
    chunks: list[np.ndarray] = []
    normal_source = args.normal_source_map or args.source_map
    for batch in tqdm(loader, desc=f"normal-cal:{category}", leave=False):
        out = feature_prior_scores(feature_prior, extractor, layers, batch["image"].to(device), device, image_size)
        chunks.append(_feature_output_map(out, normal_source).astype(np.float32))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _stats(normal: np.ndarray) -> dict[str, np.ndarray]:
    q25, median, q75 = np.percentile(normal, [25, 50, 75], axis=0)
    return {
        "mean": normal.mean(axis=0).astype(np.float32),
        "std": normal.std(axis=0).astype(np.float32),
        "median": median.astype(np.float32),
        "iqr": (q75 - q25).astype(np.float32),
    }


def _calibrate(maps: np.ndarray, stats: dict[str, np.ndarray], mode: str) -> np.ndarray:
    eps = 1e-6
    if mode == "raw":
        out = maps
    elif mode == "center":
        out = maps - stats["mean"]
    elif mode == "relu_center":
        out = np.maximum(maps - stats["mean"], 0.0)
    elif mode == "zscore":
        out = (maps - stats["mean"]) / (stats["std"] + eps)
    elif mode == "relu_zscore":
        out = np.maximum((maps - stats["mean"]) / (stats["std"] + eps), 0.0)
    elif mode == "robust_zscore":
        out = (maps - stats["median"]) / (stats["iqr"] + eps)
    elif mode == "relu_robust_zscore":
        out = np.maximum((maps - stats["median"]) / (stats["iqr"] + eps), 0.0)
    elif mode == "ratio":
        out = maps / (stats["mean"] + eps)
    elif mode == "log_ratio":
        out = np.log1p(np.maximum(maps, 0.0)) - np.log1p(np.maximum(stats["mean"], 0.0))
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _write_scores(source: Path, out: Path, ablation: str) -> None:
    src = source / "scores.csv"
    if not src.exists():
        return
    with src.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "ablation" not in fields:
        fields.append("ablation")
    for row in rows:
        row["ablation"] = ablation
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_metric_csv(metrics: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


def _json_safe(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        out[key] = value
    return out


def _write_run_args(source_args: dict[str, Any], out: Path, ablation: str, mode: str, source_map: str) -> None:
    args = dict(source_args)
    args["run_name"] = out.name
    args["ablation"] = ablation
    args["pixel_heatmap_source"] = f"train_normal_{source_map}_{mode}"
    payload = {"command": "materialize_train_normal_pixel_calibration", "args": args}
    (out / "run_args.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = Path(args.source_run_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    source_args = _run_args(source)
    config_path = Path(args.config or source_args.get("config") or "configs/mvtec.yaml")
    cfg = load_config(config_path)
    category = str(args.category or source_args.get("category") or cfg_first(cfg, ("dataset.category",), ""))
    if not category:
        raise ValueError("Provide --category or use a source run with category in run_args.json")
    image_size = int(args.image_size or source_args.get("image_size") or cfg_first(cfg, ("dataset.image_size",), 256))
    feature_ckpt = _resolve_path(args.feature_prior_checkpoint) or _resolve_path(source_args.get("feature_prior_checkpoint"))
    if feature_ckpt is None:
        raise ValueError("Provide --feature-prior-checkpoint or use a source run that records one")
    device = resolve_device(str(args.device or source_args.get("device") or cfg_device(cfg)))

    pred = np.load(source / "predictions.npz")
    source_maps = _source_maps(source, pred, args.source_map)
    normal = _normal_stack(args, cfg, category, feature_ckpt, image_size, device)
    stats = _stats(normal)
    np.savez_compressed(out_root / "normal_pixel_stats.npz", **stats)
    _source_detection, source_verification, source_score = (
        resolve_prediction_heatmaps(pred)
    )

    rows: list[dict[str, Any]] = []
    for mode in _split_csv(args.modes):
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choices: {MODES}")
        heatmaps = _calibrate(source_maps, stats, mode)
        ablation = f"{args.ablation_prefix}_{args.source_map}_{mode}"
        out = out_root / ablation
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "predictions.npz",
            labels=pred["labels"],
            image_scores=pred["image_scores"],
            masks=pred["masks"],
            **prediction_heatmap_payload(
                heatmaps,
                source_verification,
                source_score,
            ),
            paths=pred["paths"] if "paths" in pred.files else np.asarray([str(i) for i in range(len(pred["labels"]))]),
            ablation=np.asarray(ablation),
        )
        for name in ["config.yaml", "roi_budget.json", "roi_budget.jsonl", "pareto.csv", "crv_score_drop.npy", "efficiency.csv"]:
            _copy_if_exists(source / name, out / name)
        _write_scores(source, out, ablation)
        _write_run_args(source_args, out, ablation, mode, args.source_map)
        metrics = detection_metrics(pred["labels"], pred["image_scores"], pred["masks"], heatmaps)
        metrics["image_score_mode"] = source_args.get("image_score_mode", "")
        metrics["image_score_source"] = source_args.get("image_score_source", "")
        metrics["pixel_heatmap_source"] = f"train_normal_{args.source_map}_{mode}"
        metrics["calibration_mode"] = mode
        metrics["calibration_source_map"] = args.source_map
        metrics["calibration_normal_source_map"] = args.normal_source_map or args.source_map
        safe = _json_safe(metrics)
        (out / "metrics.json").write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")
        (out / "eval_metrics.json").write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")
        _write_metric_csv(safe, out / "metrics.csv")
        rows.append(
            {
                "mode": mode,
                "run": str(out),
                "image_auroc": safe.get("image_auroc"),
                "pixel_auroc": safe.get("pixel_auroc"),
                "pixel_ap": safe.get("pixel_ap"),
                "aupro": safe.get("aupro"),
                "dice": safe.get("dice"),
                "quality": float(safe.get("pixel_ap") or 0.0) + float(safe.get("aupro") or 0.0) + float(safe.get("dice") or 0.0),
            }
        )

    fields = ["mode", "run", "image_auroc", "pixel_auroc", "pixel_ap", "aupro", "dice", "quality"]
    with (out_root / "pixel_calibration_search.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: float(row["quality"]))
    (out_root / "recommended_pixel_calibration.json").write_text(json.dumps(best, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"source": str(source), "out_root": str(out_root), "recommended": best}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
