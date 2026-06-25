from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.heatmap_fusion import (
    fuse_heatmaps,
    normal_scale,
    resize_heatmaps,
)
from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.prediction_schema import prediction_heatmap_payload
from seer_ad_v2.evaluation.score_aggregation import image_scores_from_heatmaps
from seer_ad_v2.evaluation.synthetic_validation import (
    evaluate_synthetic_validation,
)


SYNTHETIC_ARRAYS = (
    "clean_heatmaps",
    "synthetic_heatmaps",
    "flipped_synthetic_heatmaps",
    "photometric_synthetic_heatmaps",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Materialize a label-free heatmap-fusion candidate calibrated only "
            "with retained normal synthetic-validation views."
        )
    )
    p.add_argument("--source-a", required=True)
    p.add_argument("--source-b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weight-a", type=float, required=True)
    p.add_argument("--calibration-seed", type=int, default=7)
    p.add_argument("--synthetic-seeds", default="7,13,23")
    p.add_argument("--center-quantile", type=float, default=0.5)
    p.add_argument("--upper-quantile", type=float, default=0.995)
    p.add_argument("--image-score-mode", default="top5")
    p.add_argument("--save-synthetic-arrays", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def split_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def assert_equal(
    source_a: np.lib.npyio.NpzFile,
    source_b: np.lib.npyio.NpzFile,
    key: str,
) -> None:
    if key in source_a.files and key in source_b.files:
        if not np.array_equal(source_a[key], source_b[key]):
            raise ValueError(f"Fusion sources differ for {key}")


def source_latency(run_dir: Path, seed: int) -> float:
    payload = read_json(
        run_dir / f"synthetic_validation_seed{seed}_metrics.json"
    )
    try:
        return float(payload.get("latency_ms", 0.0))
    except (TypeError, ValueError):
        return 0.0


def save_metrics(
    path: Path,
    metrics: dict[str, Any],
) -> None:
    safe = {
        key: value.item()
        if isinstance(value, np.generic)
        else None
        if isinstance(value, float) and not np.isfinite(value)
        else value
        for key, value in metrics.items()
    }
    path.write_text(
        json.dumps(safe, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_a_dir = Path(args.source_a)
    source_b_dir = Path(args.source_b)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.npz"
    if predictions_path.exists() and not args.overwrite:
        print(json.dumps({"out": str(out_dir), "status": "exists"}, indent=2))
        return

    calibration_a = np.load(
        source_a_dir
        / f"synthetic_validation_seed{args.calibration_seed}.npz"
    )
    calibration_b = np.load(
        source_b_dir
        / f"synthetic_validation_seed{args.calibration_seed}.npz"
    )
    target_shape = tuple(calibration_b["clean_heatmaps"].shape[1:])
    scale_a = normal_scale(
        resize_heatmaps(calibration_a["clean_heatmaps"], target_shape),
        center_quantile=args.center_quantile,
        upper_quantile=args.upper_quantile,
    )
    scale_b = normal_scale(
        calibration_b["clean_heatmaps"],
        center_quantile=args.center_quantile,
        upper_quantile=args.upper_quantile,
    )

    pred_a = np.load(source_a_dir / "predictions.npz")
    pred_b = np.load(source_b_dir / "predictions.npz")
    for key in ("labels", "paths"):
        assert_equal(pred_a, pred_b, key)
    fused = fuse_heatmaps(
        pred_a["heatmaps"],
        pred_b["heatmaps"],
        weight_a=args.weight_a,
        scale_a=scale_a,
        scale_b=scale_b,
        target_shape=tuple(pred_b["heatmaps"].shape[1:]),
    )
    image_scores = image_scores_from_heatmaps(
        fused, mode=args.image_score_mode
    )
    np.savez_compressed(
        predictions_path,
        labels=pred_b["labels"],
        image_scores=image_scores.astype(np.float32),
        masks=pred_b["masks"],
        **prediction_heatmap_payload(fused, fused, fused),
        paths=pred_b["paths"]
        if "paths" in pred_b.files
        else np.arange(len(fused)).astype(str),
        ablation=np.asarray("normal_calibrated_heatmap_fusion"),
    )
    full_metrics = detection_metrics(
        pred_b["labels"],
        image_scores,
        pred_b["masks"],
        fused,
    )
    full_metrics.update(
        {
            "pixel_heatmap_source": "normal_calibrated_fusion",
            "image_score_source": "normal_calibrated_fusion",
            "image_score_mode": args.image_score_mode,
        }
    )
    save_metrics(out_dir / "metrics.json", full_metrics)

    seeds = split_ints(args.synthetic_seeds)
    for seed in seeds:
        artifact_a = np.load(
            source_a_dir / f"synthetic_validation_seed{seed}.npz"
        )
        artifact_b = np.load(
            source_b_dir / f"synthetic_validation_seed{seed}.npz"
        )
        for key in ("paths", "variant_ids", "mask_modes"):
            assert_equal(artifact_a, artifact_b, key)
        synthetic_shape = tuple(artifact_b["clean_heatmaps"].shape[1:])
        arrays = {
            key: fuse_heatmaps(
                artifact_a[key],
                artifact_b[key],
                weight_a=args.weight_a,
                scale_a=scale_a,
                scale_b=scale_b,
                target_shape=synthetic_shape,
            )
            for key in SYNTHETIC_ARRAYS
        }
        clean_scores = image_scores_from_heatmaps(
            arrays["clean_heatmaps"], mode=args.image_score_mode
        )
        synthetic_scores = image_scores_from_heatmaps(
            arrays["synthetic_heatmaps"], mode=args.image_score_mode
        )
        synthetic_metrics = evaluate_synthetic_validation(
            arrays["clean_heatmaps"],
            arrays["synthetic_heatmaps"],
            artifact_b["synthetic_masks"],
            clean_scores,
            synthetic_scores,
            [
                arrays["flipped_synthetic_heatmaps"],
                arrays["photometric_synthetic_heatmaps"],
            ],
        )
        synthetic_metrics.update(
            {
                "latency_ms": source_latency(source_a_dir, seed)
                + source_latency(source_b_dir, seed),
                "candidate_run": str(out_dir),
                "seed": seed,
                "selection_data": "normal_images_plus_synthetic_masks",
                "uses_real_anomaly_labels_for_selection": False,
                "uses_real_anomaly_masks_for_selection": False,
                "fusion_source_a": str(source_a_dir),
                "fusion_source_b": str(source_b_dir),
                "fusion_weight_a": args.weight_a,
                "calibration_seed": args.calibration_seed,
                "normal_scale_a": list(scale_a),
                "normal_scale_b": list(scale_b),
                "pixel_heatmap_source": "normal_calibrated_fusion",
                "image_score_source": "normal_calibrated_fusion",
                "image_score_mode": args.image_score_mode,
            }
        )
        if args.save_synthetic_arrays:
            np.savez_compressed(
                out_dir / f"synthetic_validation_seed{seed}.npz",
                **arrays,
                clean_score_heatmaps=arrays["clean_heatmaps"],
                synthetic_score_heatmaps=arrays["synthetic_heatmaps"],
                synthetic_masks=artifact_b["synthetic_masks"],
                clean_image_scores=clean_scores.astype(np.float32),
                synthetic_image_scores=synthetic_scores.astype(np.float32),
                paths=artifact_b["paths"],
                variant_ids=artifact_b["variant_ids"],
                mask_modes=artifact_b["mask_modes"],
                seed=np.asarray(seed, dtype=np.int64),
            )
        save_metrics(
            out_dir
            / f"synthetic_validation_seed{seed}_metrics.json",
            synthetic_metrics,
        )

    source_b_args = read_json(source_b_dir / "run_args.json")
    base_args = source_b_args.get("args", {})
    if not isinstance(base_args, dict):
        base_args = {}
    run_args = {
        **base_args,
        "run_name": out_dir.name,
        "pixel_heatmap_source": "normal_calibrated_fusion",
        "image_score_source": "normal_calibrated_fusion",
        "image_score_mode": args.image_score_mode,
        "fusion_source_a": str(source_a_dir),
        "fusion_source_b": str(source_b_dir),
        "fusion_weight_a": args.weight_a,
        "calibration_seed": args.calibration_seed,
        "normal_scale_a": list(scale_a),
        "normal_scale_b": list(scale_b),
        "uses_real_anomaly_labels_for_selection": False,
        "uses_real_anomaly_masks_for_selection": False,
        "synthetic_arrays_materialized": bool(
            args.save_synthetic_arrays
        ),
    }
    (out_dir / "run_args.json").write_text(
        json.dumps(
            {
                "command": "materialize_heatmap_fusion_candidate",
                "args": run_args,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(out_dir),
                "synthetic_seeds": seeds,
                "weight_a": args.weight_a,
                "synthetic_arrays_materialized": bool(
                    args.save_synthetic_arrays
                ),
                "normal_scale_a": scale_a,
                "normal_scale_b": scale_b,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
