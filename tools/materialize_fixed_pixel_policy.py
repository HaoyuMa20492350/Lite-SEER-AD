from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
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
CALIBRATION_MODES = ["raw", "center", "relu_center", "zscore", "relu_zscore", "robust_zscore", "relu_robust_zscore", "ratio", "log_ratio"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize a fixed train-normal pixel policy without test-set mode selection.")
    p.add_argument("--source-run-dir", required=True)
    p.add_argument("--out-run-dir", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--category", default=None)
    p.add_argument("--feature-prior-checkpoint", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--train-max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--source-map", choices=MAP_SOURCES, default="heatmaps")
    p.add_argument("--normal-source-map", choices=MAP_SOURCES, default="feature_raw")
    p.add_argument("--calibration-mode", choices=CALIBRATION_MODES, default="relu_robust_zscore")
    p.add_argument("--postprocess-mode", default="highpass:9")
    p.add_argument("--ablation-name", default="feature_pixel_policy")
    p.add_argument("--device", default=None)
    p.add_argument("--allow-random-feature-weights", action="store_true")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    raise ValueError(f"Cannot compute train-normal map source: {source}")


def _normal_stack(
    cfg: dict[str, Any],
    category: str,
    feature_ckpt: Path,
    image_size: int,
    device: str,
    source: str,
    batch_size: int | None,
    train_max_samples: int | None,
    allow_random_feature_weights: bool,
) -> np.ndarray:
    checkpoint = load_checkpoint(feature_ckpt)
    feature_prior, extractor, layers = load_feature_prior_components(
        checkpoint,
        device,
        allow_random_weights=allow_random_feature_weights,
    )
    train_ds = build_dataset(
        cfg_first(cfg, ("dataset.name",), "mvtec"),
        cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD"),
        category,
        "train",
        image_size,
        max_samples=train_max_samples,
    )
    bs = int(batch_size or cfg_int(cfg, ("feature_prior.batch_size", "training.batch_size"), 8))
    loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=cfg_int(cfg, ("dataset.num_workers",), 0))
    chunks: list[np.ndarray] = []
    for batch in tqdm(loader, desc=f"fixed-pixel-policy:{category}", leave=False):
        out = feature_prior_scores(feature_prior, extractor, layers, batch["image"].to(device), device, image_size)
        chunks.append(_feature_output_map(out, source).astype(np.float32))
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


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    k = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)


def _postprocess_one(img: np.ndarray, mode: str) -> np.ndarray:
    img = img.astype(np.float32)
    if mode in {"", "none", "raw"}:
        out = img
    elif mode.startswith("gaussian:"):
        out = _gaussian(img, float(mode.split(":", 1)[1]))
    elif mode.startswith("highpass:"):
        sigma = float(mode.split(":", 1)[1])
        out = np.maximum(img - _gaussian(img, sigma), 0.0)
    elif mode.startswith("tophat:"):
        out = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, _kernel(int(float(mode.split(":", 1)[1]))))
    elif mode.startswith("closing:"):
        out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, _kernel(int(float(mode.split(":", 1)[1]))))
    else:
        raise ValueError(f"Unknown postprocess mode: {mode}")
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _postprocess(maps: np.ndarray, mode: str) -> np.ndarray:
    return np.stack([_postprocess_one(h, mode) for h in maps]).astype(np.float32)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


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


def _write_run_args(source_args: dict[str, Any], out: Path, args: argparse.Namespace) -> None:
    payload_args = dict(source_args)
    payload_args["run_name"] = out.name
    payload_args["ablation"] = args.ablation_name
    payload_args["pixel_heatmap_source"] = (
        f"fixed_train_normal_{args.source_map}_{args.normal_source_map}_{args.calibration_mode}_{args.postprocess_mode}"
    )
    payload_args["pixel_policy_calibration_mode"] = args.calibration_mode
    payload_args["pixel_policy_postprocess_mode"] = args.postprocess_mode
    payload_args["pixel_policy_source_map"] = args.source_map
    payload_args["pixel_policy_normal_source_map"] = args.normal_source_map
    payload = {"command": "materialize_fixed_pixel_policy", "args": payload_args}
    (out / "run_args.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = Path(args.source_run_dir)
    out = Path(args.out_run_dir)
    out.mkdir(parents=True, exist_ok=True)
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
    normal = _normal_stack(
        cfg,
        category,
        feature_ckpt,
        image_size,
        device,
        args.normal_source_map,
        args.batch_size,
        args.train_max_samples,
        args.allow_random_feature_weights,
    )
    stats = _stats(normal)
    calibrated = _calibrate(source_maps, stats, args.calibration_mode)
    heatmaps = _postprocess(calibrated, args.postprocess_mode)
    _source_detection, source_verification, source_score = (
        resolve_prediction_heatmaps(pred)
    )

    np.savez_compressed(out / "normal_pixel_stats.npz", **stats)
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
        ablation=np.asarray(args.ablation_name),
    )
    for name in ["config.yaml", "roi_budget.json", "roi_budget.jsonl", "pareto.csv", "crv_score_drop.npy", "efficiency.csv"]:
        _copy_if_exists(source / name, out / name)
    _write_scores(source, out, args.ablation_name)
    _write_run_args(source_args, out, args)
    metrics = detection_metrics(pred["labels"], pred["image_scores"], pred["masks"], heatmaps)
    metrics["image_score_mode"] = source_args.get("image_score_mode", "")
    metrics["image_score_source"] = source_args.get("image_score_source", "")
    metrics["pixel_heatmap_source"] = (
        f"fixed_train_normal_{args.source_map}_{args.normal_source_map}_{args.calibration_mode}_{args.postprocess_mode}"
    )
    metrics["pixel_policy_calibration_mode"] = args.calibration_mode
    metrics["pixel_policy_postprocess_mode"] = args.postprocess_mode
    metrics["pixel_policy_source_map"] = args.source_map
    metrics["pixel_policy_normal_source_map"] = args.normal_source_map
    safe = _json_safe(metrics)
    (out / "metrics.json").write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")
    (out / "eval_metrics.json").write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")
    _write_metric_csv(safe, out / "metrics.csv")
    print(json.dumps({"source": str(source), "out": str(out), "metrics": safe}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
