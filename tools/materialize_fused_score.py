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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.config import cfg_device, cfg_first, cfg_int, cfg_seed, image_size as cfg_image_size, load_config, resolve_device
from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.data.hard_negative_mining import ROI
from seer_ad_v2.evaluation.score_aggregation import IMAGE_SCORE_MODES, image_scores_from_heatmaps
from seer_ad_v2.models.counterfactual.repair_verification import apply_crv_to_heatmap
from seer_ad_v2.models.feature_prior import feature_prior_scores, load_feature_prior_components
from seer_ad_v2.utils.io import load_checkpoint
from seer_ad_v2.utils.seed import seed_everything


FEATURE_SCORE_SOURCES = ["feature", "feature_raw", "feature_raw_distance", "feature_raw_cosine"]
ALL_SCORE_SOURCES = ["final", "base", *FEATURE_SCORE_SOURCES]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize a tuned-CRV run with fused image-level anomaly scores.")
    p.add_argument("--source-run-dir", required=True)
    p.add_argument("--out-run-dir", required=True)
    p.add_argument("--recommendation", default=None)
    p.add_argument("--crv-weight", type=float, default=None)
    p.add_argument("--base-key", choices=["auto", "residual", "feature", "final"], default="auto")
    p.add_argument("--score-a-source", choices=ALL_SCORE_SOURCES, default="feature_raw")
    p.add_argument("--score-a-mode", choices=IMAGE_SCORE_MODES, default="p95")
    p.add_argument("--score-b-source", choices=ALL_SCORE_SOURCES, default="feature_raw_cosine")
    p.add_argument("--score-b-mode", choices=IMAGE_SCORE_MODES, default="max")
    p.add_argument("--weight-a", type=float, default=0.5)
    p.add_argument("--fusion", choices=["raw", "rank", "zscore", "train_zscore"], default="train_zscore")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--category", default=None)
    p.add_argument("--feature-prior-checkpoint", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--allow-random-feature-weights", action="store_true")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_roi_budget(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_image(roi_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for row in roi_rows:
        out.setdefault(int(row.get("image_index", -1)), []).append(row)
    return out


def _roi_from_row(row: dict[str, Any]) -> ROI:
    x1, y1, x2, y2 = [int(v) for v in row.get("bbox", [0, 0, 0, 0])]
    return ROI(x1=x1, y1=y1, x2=x2, y2=y2, area=max(0, (x2 - x1) * (y2 - y1)), peak=float(row.get("residual_score", 0.0)))


def _crv_weight(args: argparse.Namespace) -> float:
    if args.crv_weight is not None:
        return float(args.crv_weight)
    if not args.recommendation:
        raise SystemExit("Provide --recommendation or --crv-weight")
    data = _load_json(Path(args.recommendation))
    if "crv_weight" not in data:
        raise KeyError(f"{args.recommendation} does not contain crv_weight")
    return float(data["crv_weight"])


def _image_npz_heatmaps(run_dir: Path, count: int, key: str) -> np.ndarray:
    heatmaps: list[np.ndarray] = []
    for idx in range(count):
        npz_path = run_dir / "images" / f"{idx:05d}" / "residual_heatmap.npz"
        data = np.load(npz_path)
        if key not in data:
            raise KeyError(f"{npz_path} does not contain heatmap key '{key}'. Available: {list(data.keys())}")
        heatmaps.append(np.asarray(data[key], dtype=np.float32))
    return np.stack(heatmaps).astype(np.float32)


def _base_heatmaps(run_dir: Path, count: int, base_key: str) -> np.ndarray:
    heatmaps: list[np.ndarray] = []
    for idx in range(count):
        npz_path = run_dir / "images" / f"{idx:05d}" / "residual_heatmap.npz"
        data = np.load(npz_path)
        key = base_key
        if key == "auto":
            if "feature" in data and np.asarray(data["feature"]).max() > 0:
                key = "feature"
            else:
                key = "residual"
        if key not in data:
            raise KeyError(f"{npz_path} does not contain heatmap key '{key}'. Available: {list(data.keys())}")
        heatmaps.append(np.asarray(data[key], dtype=np.float32))
    return np.stack(heatmaps).astype(np.float32)


def _final_heatmaps(source: Path, count: int, base_key: str, crv_weight: float) -> tuple[np.ndarray, np.ndarray]:
    base = _base_heatmaps(source, count, base_key)
    roi_rows = _load_roi_budget(source / "roi_budget.json")
    by_image = _rows_by_image(roi_rows)
    fused: list[np.ndarray] = []
    for idx, heatmap in enumerate(base):
        rows = by_image.get(idx, [])
        rois = [_roi_from_row(row) for row in rows]
        drops = [float(row.get("sdr", 0.0)) for row in rows]
        fused.append(apply_crv_to_heatmap(heatmap, rois, drops, weight=crv_weight))
    return base, np.stack(fused).astype(np.float32)


def _test_scores(source: Path, base: np.ndarray, final_heatmaps: np.ndarray, source_name: str, mode: str) -> np.ndarray:
    if source_name == "final":
        maps = final_heatmaps
    elif source_name == "base":
        maps = base
    else:
        maps = _image_npz_heatmaps(source, len(final_heatmaps), source_name)
    return image_scores_from_heatmaps(maps, mode=mode).astype(np.float32)


def _feature_output_maps(output: Any, source_name: str) -> np.ndarray:
    if source_name == "feature":
        return output.heatmaps
    if source_name == "feature_raw":
        return output.raw_heatmaps
    if source_name == "feature_raw_distance":
        return output.raw_distance_heatmaps
    if source_name == "feature_raw_cosine":
        return output.raw_cosine_heatmaps
    raise ValueError(f"Training calibration supports only feature sources, got {source_name}")


def _train_feature_scores(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    category: str,
    source_name: str,
    mode: str,
    image_size: int,
    device: str,
) -> np.ndarray:
    if source_name not in FEATURE_SCORE_SOURCES:
        raise ValueError(f"--fusion train_zscore requires feature sources; got {source_name}")
    ckpt_path = args.feature_prior_checkpoint
    if ckpt_path is None:
        run_args = _load_json(Path(args.source_run_dir) / "run_args.json").get("args", {})
        ckpt_path = run_args.get("feature_prior_checkpoint")
    if not ckpt_path:
        raise SystemExit("Provide --feature-prior-checkpoint or use a source run with it in run_args.json")
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    checkpoint, extractor, layers = load_feature_prior_components(checkpoint, device, allow_random_weights=args.allow_random_feature_weights)
    train_ds = build_dataset(
        cfg_first(cfg, ("dataset.name",), "mvtec"),
        cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD"),
        category,
        "train",
        image_size,
        max_samples=args.max_samples,
    )
    batch_size = int(args.batch_size or cfg_int(cfg, ("feature_prior.batch_size", "training.batch_size"), 8))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=cfg_int(cfg, ("dataset.num_workers",), 0))
    scores: list[np.ndarray] = []
    for batch in loader:
        output = feature_prior_scores(checkpoint, extractor, layers, batch["image"].to(device), device, image_size)
        maps = _feature_output_maps(output, source_name)
        scores.append(image_scores_from_heatmaps(maps, mode=mode))
    return np.concatenate(scores).astype(np.float32)


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values))
    return (order / max(1, len(values) - 1)).astype(np.float32)


def _zscore(values: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    ref = values if reference is None else reference
    return ((values - float(np.mean(ref))) / (float(np.std(ref)) + 1e-8)).astype(np.float32)


def _fuse(a: np.ndarray, b: np.ndarray, weight_a: float, fusion: str, train_a: np.ndarray | None, train_b: np.ndarray | None) -> np.ndarray:
    weight_a = float(np.clip(weight_a, 0.0, 1.0))
    if fusion == "raw":
        aa, bb = a, b
    elif fusion == "rank":
        aa, bb = _rank01(a), _rank01(b)
    elif fusion == "zscore":
        aa, bb = _zscore(a), _zscore(b)
    elif fusion == "train_zscore":
        if train_a is None or train_b is None:
            raise ValueError("train_zscore requires training calibration scores")
        aa, bb = _zscore(a, train_a), _zscore(b, train_b)
    else:
        raise ValueError(f"Unknown fusion mode: {fusion}")
    return (weight_a * aa + (1.0 - weight_a) * bb).astype(np.float32)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_scores(source: Path, out: Path, image_scores: np.ndarray, ablation_name: str) -> None:
    src = source / "scores.csv"
    if not src.exists():
        return
    with src.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "image_score" not in fields:
        fields.append("image_score")
    if "ablation" not in fields:
        fields.append("ablation")
    for idx, row in enumerate(rows):
        if idx < len(image_scores):
            row["image_score"] = str(float(image_scores[idx]))
        row["ablation"] = ablation_name
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg_seed(cfg, args.seed)
    seed_everything(seed)
    source = Path(args.source_run_dir)
    out = Path(args.out_run_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_args = _load_json(source / "run_args.json").get("args", {})
    category = str(args.category or run_args.get("category") or cfg_first(cfg, ("dataset.category",), "bottle"))
    image_size = cfg_image_size(cfg, args.image_size or run_args.get("image_size"))
    max_samples = args.max_samples
    if max_samples is None and run_args.get("max_samples") is not None:
        args.max_samples = int(run_args["max_samples"])
    device = resolve_device(cfg_device(cfg, args.device or run_args.get("device")))
    crv_weight = _crv_weight(args)

    pred = np.load(source / "predictions.npz")
    labels = pred["labels"]
    masks = pred["masks"]
    paths = pred["paths"] if "paths" in pred.files else np.asarray([str(i) for i in range(len(labels))])
    base, final_heatmaps = _final_heatmaps(source, len(labels), args.base_key, crv_weight)
    score_a = _test_scores(source, base, final_heatmaps, args.score_a_source, args.score_a_mode)
    score_b = _test_scores(source, base, final_heatmaps, args.score_b_source, args.score_b_mode)

    train_a = train_b = None
    if args.fusion == "train_zscore":
        train_a = _train_feature_scores(args, cfg, category, args.score_a_source, args.score_a_mode, image_size, device)
        train_b = _train_feature_scores(args, cfg, category, args.score_b_source, args.score_b_mode, image_size, device)
    image_scores = _fuse(score_a, score_b, args.weight_a, args.fusion, train_a, train_b)
    score_name = f"{args.fusion}:{args.weight_a:.2f}*{args.score_a_source}:{args.score_a_mode}+{1.0 - args.weight_a:.2f}*{args.score_b_source}:{args.score_b_mode}"

    np.savez_compressed(
        out / "predictions.npz",
        labels=labels,
        image_scores=image_scores,
        masks=masks,
        heatmaps=final_heatmaps,
        score_a=score_a,
        score_b=score_b,
        paths=paths,
        ablation=np.asarray("feature_tuned_crv"),
    )
    for name in ["config.yaml", "roi_budget.json", "roi_budget.jsonl", "pareto.csv", "crv_score_drop.npy"]:
        _copy_if_exists(source / name, out / name)
    _write_scores(source, out, image_scores, "feature_tuned_crv")
    payload = _load_json(source / "run_args.json")
    source_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    if isinstance(source_args, dict):
        source_args = dict(source_args)
        source_args["run_name"] = out.name
        source_args["ablation"] = "feature_tuned_crv"
        source_args["crv_weight"] = crv_weight
        source_args["image_score_source"] = "fused"
        source_args["image_score_mode"] = score_name
    (out / "run_args.json").write_text(json.dumps({"command": "materialize_fused_score", "args": source_args}, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "crv_weight": crv_weight,
                "image_score_source": "fused",
                "image_score_mode": score_name,
                "score_a_source": args.score_a_source,
                "score_a_mode": args.score_a_mode,
                "score_b_source": args.score_b_source,
                "score_b_mode": args.score_b_mode,
                "weight_a": float(args.weight_a),
                "fusion": args.fusion,
                "train_score_a_mean": None if train_a is None else float(np.mean(train_a)),
                "train_score_a_std": None if train_a is None else float(np.std(train_a)),
                "train_score_b_mean": None if train_b is None else float(np.mean(train_b)),
                "train_score_b_std": None if train_b is None else float(np.std(train_b)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"source": str(source), "out": str(out), "image_score_mode": score_name}, indent=2))


if __name__ == "__main__":
    main()
