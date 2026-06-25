from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.config import cfg_device, cfg_first, cfg_int, cfg_seed, dataset_category, image_size as cfg_image_size, load_config, make_run_dir, resolve_device
from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.pixel_policy import (
    MULTISCALE_STATS_NAME,
    feature_output_map,
    multiscale_candidate_maps,
)
from seer_ad_v2.evaluation.prediction_schema import prediction_heatmap_payload
from seer_ad_v2.evaluation.score_aggregation import IMAGE_SCORE_MODES, image_scores_from_heatmaps
from seer_ad_v2.models.feature_prior import feature_prior_scores, load_feature_prior_components
from seer_ad_v2.utils.io import load_checkpoint
from seer_ad_v2.utils.run import save_run_metadata
from seer_ad_v2.utils.seed import seed_everything


HEATMAP_SOURCES = ["feature", "feature_raw", "feature_raw_distance", "feature_raw_cosine", "multiscale_fusion"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize a feature-prior-only candidate run for held-out policy selection.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--category", default=None)
    p.add_argument("--feature-prior-checkpoint", required=True)
    p.add_argument("--secondary-feature-prior-checkpoint", default=None)
    p.add_argument("--multiscale-primary-source", default="feature_raw")
    p.add_argument("--multiscale-secondary-source", default="feature_raw")
    p.add_argument("--multiscale-calibration-mode", choices=["relu_zscore", "relu_robust_zscore"], default="relu_zscore")
    p.add_argument("--multiscale-primary-postprocess", default="raw")
    p.add_argument("--multiscale-secondary-postprocess", default="highpass:3")
    p.add_argument("--multiscale-primary-weight", type=float, default=0.55)
    p.add_argument("--multiscale-fusion-mode", choices=["weighted", "max"], default="weighted")
    p.add_argument("--multiscale-postprocess", default="gaussian:1")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--skip-metrics", action="store_true")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--run-name", required=True)
    p.add_argument("--image-score-mode", choices=IMAGE_SCORE_MODES, default=None)
    p.add_argument("--image-score-source", choices=HEATMAP_SOURCES, default=None)
    p.add_argument("--pixel-heatmap-source", choices=HEATMAP_SOURCES, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--allow-random-feature-weights", action="store_true")
    return p.parse_args()


def _select(out: Any, source: str) -> np.ndarray:
    if source == "feature":
        return out.heatmaps
    if source == "feature_raw":
        return out.raw_heatmaps
    if source == "feature_raw_distance":
        return out.raw_distance_heatmaps
    if source == "feature_raw_cosine":
        return out.raw_cosine_heatmaps
    raise ValueError(f"Unknown heatmap source: {source}")


def _normal_stats(maps: list[np.ndarray]) -> dict[str, np.ndarray]:
    values = np.concatenate(maps, axis=0).astype(np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    median = np.median(values, axis=0)
    iqr = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
    return {
        "mean": mean.astype(np.float32),
        "std": np.maximum(std, 1e-6).astype(np.float32),
        "median": median.astype(np.float32),
        "iqr": np.maximum(iqr, 1e-6).astype(np.float32),
    }


def _materialize_multiscale_stats(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    category: str,
    image_size: int,
    batch_size: int,
    device: str,
    primary: tuple[dict[str, Any], torch.nn.Module, list[str]],
    secondary: tuple[dict[str, Any], torch.nn.Module, list[str]],
    run_dir: Path,
) -> None:
    train_dataset = build_dataset(
        cfg_first(cfg, ("dataset.name",), "mvtec"),
        cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD"),
        category,
        "train",
        image_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg_int(cfg, ("dataset.num_workers",), 0))
    primary_maps: list[np.ndarray] = []
    secondary_maps: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device)
        first = feature_prior_scores(*primary, images, device, image_size)
        second = feature_prior_scores(*secondary, images, device, image_size)
        primary_maps.append(feature_output_map(first, args.multiscale_primary_source))
        secondary_maps.append(feature_output_map(second, args.multiscale_secondary_source))
    primary_stats = _normal_stats(primary_maps)
    secondary_stats = _normal_stats(secondary_maps)
    np.savez_compressed(
        run_dir / MULTISCALE_STATS_NAME,
        **{f"primary_{name}": value for name, value in primary_stats.items()},
        **{f"secondary_{name}": value for name, value in secondary_stats.items()},
    )


def _json_safe(metrics: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        safe[key] = value
    return safe


def _write_scores(path: Path, paths: list[str], labels: np.ndarray, image_scores: np.ndarray, ablation: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "path", "label", "image_score", "latency_ms", "nfe", "ablation"])
        writer.writeheader()
        for idx, (sample_path, label, score) in enumerate(zip(paths, labels, image_scores)):
            writer.writerow(
                {
                    "index": idx,
                    "path": sample_path,
                    "label": int(label),
                    "image_score": float(score),
                    "latency_ms": 0.0,
                    "nfe": 0,
                    "ablation": ablation,
                }
            )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg_seed(cfg, args.seed))
    category = dataset_category(cfg, args.category)
    image_size = cfg_image_size(cfg, args.image_size)
    batch_size = int(args.batch_size or cfg_int(cfg, ("feature_prior.batch_size", "training.batch_size"), 8))
    args.image_score_mode = args.image_score_mode or str(cfg_first(cfg, ("evaluation.image_score_mode",), "top5"))
    args.image_score_source = args.image_score_source or str(cfg_first(cfg, ("evaluation.image_score_source",), "feature_raw_cosine"))
    args.pixel_heatmap_source = args.pixel_heatmap_source or str(cfg_first(cfg, ("evaluation.pixel_heatmap_source",), "feature_raw"))
    if args.secondary_feature_prior_checkpoint:
        args.image_score_source = "multiscale_fusion"
        args.pixel_heatmap_source = "multiscale_fusion"
    if args.image_score_source not in HEATMAP_SOURCES:
        raise ValueError(f"Unsupported image score source for feature-prior-only run: {args.image_score_source}")
    if args.pixel_heatmap_source not in HEATMAP_SOURCES:
        raise ValueError(f"Unsupported pixel heatmap source for feature-prior-only run: {args.pixel_heatmap_source}")

    device = resolve_device(cfg_device(cfg, args.device))
    run_dir = make_run_dir(cfg, args.run_name)
    save_run_metadata(run_dir, cfg, args, device, "materialize_feature_prior_candidate")

    checkpoint = load_checkpoint(args.feature_prior_checkpoint)
    feature_prior, extractor, layers = load_feature_prior_components(checkpoint, device, allow_random_weights=args.allow_random_feature_weights)
    secondary_components = None
    if args.secondary_feature_prior_checkpoint:
        secondary_checkpoint = load_checkpoint(args.secondary_feature_prior_checkpoint)
        secondary_components = load_feature_prior_components(
            secondary_checkpoint,
            device,
            allow_random_weights=args.allow_random_feature_weights,
        )
        _materialize_multiscale_stats(
            cfg,
            args,
            category,
            image_size,
            batch_size,
            device,
            (feature_prior, extractor, layers),
            secondary_components,
            run_dir,
        )
    dataset = build_dataset(
        cfg_first(cfg, ("dataset.name",), "mvtec"),
        cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD"),
        category,
        args.split,
        image_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cfg_int(cfg, ("dataset.num_workers",), 0))

    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    paths: list[str] = []
    pixel_heatmaps: list[np.ndarray] = []
    final_heatmaps: list[np.ndarray] = []
    score_heatmaps: list[np.ndarray] = []
    elapsed = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        out = feature_prior_scores(feature_prior, extractor, layers, images, device, image_size)
        secondary_out = None
        if secondary_components is not None:
            secondary_out = feature_prior_scores(*secondary_components, images, device, image_size)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        labels.append(batch["label"].detach().cpu().numpy().astype(np.uint8))
        masks.append(batch["mask"][:, 0].detach().cpu().numpy().astype(np.uint8))
        paths.extend([str(path) for path in batch["path"]])
        if secondary_out is not None:
            fused = multiscale_candidate_maps(out, secondary_out, vars(args), run_dir).astype(np.float32)
            final_heatmaps.append(fused)
            score_heatmaps.append(fused)
            pixel_heatmaps.append(fused)
        else:
            final_heatmaps.append(out.heatmaps.astype(np.float32))
            score_heatmaps.append(_select(out, args.image_score_source).astype(np.float32))
            pixel_heatmaps.append(_select(out, args.pixel_heatmap_source).astype(np.float32))

    labels_np = np.concatenate(labels, axis=0)
    masks_np = np.concatenate(masks, axis=0)
    heatmaps_np = np.concatenate(pixel_heatmaps, axis=0)
    final_np = np.concatenate(final_heatmaps, axis=0)
    score_np = np.concatenate(score_heatmaps, axis=0)
    image_scores = image_scores_from_heatmaps(score_np, mode=args.image_score_mode)

    np.savez_compressed(
        run_dir / "predictions.npz",
        labels=labels_np,
        image_scores=image_scores,
        masks=masks_np,
        **prediction_heatmap_payload(heatmaps_np, final_np, score_np),
        paths=np.asarray(paths),
        ablation=np.asarray("feature_prior_candidate"),
    )
    metrics = (
        {}
        if args.skip_metrics
        else detection_metrics(labels_np, image_scores, masks_np, heatmaps_np)
    )
    metrics.update(
        {
            "ablation": "feature_prior_candidate",
            "image_score_mode": args.image_score_mode,
            "image_score_source": args.image_score_source,
            "pixel_heatmap_source": args.pixel_heatmap_source,
            "crv_weight": 0.0,
            "reconstruction_steps": 0.0,
            "latency_ms": float(elapsed * 1000.0 / max(1, len(labels_np))),
            "multiscale_enabled": int(secondary_components is not None),
            "dataset_split": args.split,
            "test_images": int(len(labels_np)),
        }
    )
    (run_dir / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "eval_metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2, allow_nan=False), encoding="utf-8")
    _write_scores(run_dir / "scores.csv", paths, labels_np, image_scores, "feature_prior_candidate")
    print(json.dumps({"run_dir": str(run_dir), "metrics": _json_safe(metrics)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
