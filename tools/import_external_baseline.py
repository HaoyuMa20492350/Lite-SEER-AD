from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.registry import BASELINES
from baselines.official_sources import (
    load_official_source_manifest,
    validate_official_provenance,
)
from seer_ad_v2.config import cfg_device, dataset_category, image_size as cfg_image_size, load_config, make_run_dir, resolve_device
from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.metrics_plan import efficiency_summary
from seer_ad_v2.evaluation.pareto import write_pareto
from seer_ad_v2.evaluation.pixel_threshold_policy import (
    load_pixel_threshold_policy,
)
from seer_ad_v2.evaluation.prediction_schema import prediction_heatmap_payload
from seer_ad_v2.utils.image import heatmap_to_uint8, save_image
from seer_ad_v2.utils.io import save_json
from seer_ad_v2.utils.run import save_run_metadata


REQUIRED_KEYS = ("labels", "image_scores", "masks", "heatmaps")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import a third-party baseline prediction artifact into the Lite-SEER-AD output contract.")
    p.add_argument("--method", choices=sorted(BASELINES), required=True)
    p.add_argument("--predictions", required=True, help="External predictions.npz with labels, image_scores, masks, and heatmaps.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--category", default=None)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--latency-ms-mean", type=float, default=0.0)
    p.add_argument("--nfe-mean", type=float, default=0.0)
    p.add_argument("--display-method", default=None)
    p.add_argument(
        "--implementation-variant",
        default="external_predictions_unverified",
    )
    p.add_argument("--official-implementation", action="store_true")
    p.add_argument("--source-url", default="")
    p.add_argument("--source-commit", default="")
    p.add_argument("--dataset-id", default=None)
    p.add_argument("--provenance", default=None)
    p.add_argument("--pixel-threshold-policy", default=None)
    p.add_argument(
        "--source-manifest",
        default="baselines/official_sources.json",
    )
    return p.parse_args()


def _metric_csv(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


def _validate_arrays(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [key for key in REQUIRED_KEYS if key not in data.files]
    if missing:
        raise SystemExit(f"Missing required prediction arrays: {', '.join(missing)}")
    labels = np.asarray(data["labels"], dtype=np.uint8).reshape(-1)
    image_scores = np.asarray(data["image_scores"], dtype=np.float32).reshape(-1)
    masks = np.asarray(data["masks"], dtype=np.uint8)
    heatmaps = np.asarray(data["heatmaps"], dtype=np.float32)
    if len(labels) != len(image_scores) or len(labels) != len(masks) or len(labels) != len(heatmaps):
        raise SystemExit(
            "Prediction arrays must agree on image count: "
            f"labels={len(labels)}, image_scores={len(image_scores)}, masks={len(masks)}, heatmaps={len(heatmaps)}"
        )
    if masks.shape != heatmaps.shape:
        raise SystemExit(f"masks and heatmaps must have identical shape, got masks={masks.shape}, heatmaps={heatmaps.shape}")
    return labels, image_scores, masks, heatmaps


def _paths_from_dataset(cfg: dict[str, Any], category: str, image_size: int, count: int) -> np.ndarray:
    dataset = build_dataset(cfg.get("dataset", {}).get("name", "mvtec"), cfg.get("dataset", {}).get("root", "SEER-AD-dataset/MVTec-AD"), category, "test", image_size)
    paths = [dataset[i]["path"] for i in range(min(count, len(dataset)))]
    if len(paths) < count:
        paths.extend([""] * (count - len(paths)))
    return np.asarray(paths)


def _dataset_id(cfg: dict[str, Any], explicit: str | None) -> str:
    if explicit:
        return explicit
    name = str((cfg.get("dataset", {}) or {}).get("name", ""))
    return {"mvtec": "mvtec15", "mvtec_ad": "mvtec15"}.get(name, name)


def _load_provenance(
    path: str | None,
) -> tuple[dict[str, Any], Path | None]:
    if not path:
        return {}, None
    provenance_path = Path(path)
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{provenance_path} must contain a JSON object")
    return payload, provenance_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    category = dataset_category(cfg, args.category)
    image_size = cfg_image_size(cfg, args.image_size)
    dataset_id = _dataset_id(cfg, args.dataset_id)
    device = resolve_device(cfg_device(cfg, args.device))
    run_name = args.run_name or f"{args.method}_{category}"
    run_dir = make_run_dir(cfg, run_name)
    save_run_metadata(run_dir, cfg, args, device, "tools/import_external_baseline")
    for sub in ["heatmaps", "masks", "images"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    data = np.load(args.predictions)
    labels, image_scores, masks, heatmaps = _validate_arrays(data)
    paths = np.asarray(data["paths"]) if "paths" in data.files else _paths_from_dataset(cfg, category, image_size, len(labels))
    provenance, provenance_path = _load_provenance(args.provenance)
    threshold_policy_path = (
        Path(args.pixel_threshold_policy)
        if args.pixel_threshold_policy
        else Path(args.predictions).parent / "pixel_threshold_policy.json"
    )
    threshold_policy = (
        load_pixel_threshold_policy(threshold_policy_path)
        if threshold_policy_path.exists()
        else None
    )
    source_manifest = load_official_source_manifest(args.source_manifest)
    source = source_manifest["sources"].get(args.method)
    if args.official_implementation and not provenance:
        raise ValueError(
            "--official-implementation requires a pinned provenance.json"
        )
    if provenance:
        if source is None:
            raise ValueError(
                f"No pinned official source is registered for {args.method}"
            )
        errors = validate_official_provenance(
            provenance,
            source,
            method=args.method,
            dataset=dataset_id,
            category=category,
        )
        if errors:
            raise ValueError(
                "External baseline provenance validation failed:\n- "
                + "\n- ".join(errors)
            )
        args.official_implementation = bool(
            provenance["official_implementation"]
        )
        args.implementation_variant = str(provenance["source_kind"])
        args.source_url = str(provenance["source_repository"])
        args.source_commit = str(provenance["source_commit"])
        args.display_method = str(source["display_name"])

    metrics = detection_metrics(
        labels,
        image_scores,
        masks,
        heatmaps,
        pixel_threshold=(
            float(threshold_policy["threshold"]) if threshold_policy else None
        ),
        threshold_protocol=(
            str(threshold_policy.get("protocol", "fixed_external"))
            if threshold_policy
            else None
        ),
    )
    if threshold_policy:
        metrics.update(
            {
                "pixel_threshold_policy_path": str(threshold_policy_path),
                "max_normal_pixel_fpr": threshold_policy.get(
                    "max_normal_pixel_fpr"
                ),
                "threshold_evidence_normal_fpr": threshold_policy.get(
                    "observed_normal_pixel_fpr"
                ),
                "threshold_uses_real_anomaly_labels": threshold_policy.get(
                    "uses_real_anomaly_labels",
                    False,
                ),
                "threshold_uses_real_anomaly_masks": threshold_policy.get(
                    "uses_real_anomaly_masks",
                    False,
                ),
            }
        )
    eff_rows = [
        {
            "latency_ms": float(args.latency_ms_mean),
            "nfe": float(args.nfe_mean),
            "repaired_area_ratio": 0.0,
            "local_region_ratio": 0.0,
        }
        for _ in labels
    ]
    eff = efficiency_summary(eff_rows)
    metrics.update({f"eff_{key}": value for key, value in eff.items()})
    metrics.update(
        {
            "method": args.method,
            "method_id": args.method,
            "display_method": args.display_method
            or BASELINES[args.method].display_name,
            "implementation_variant": args.implementation_variant,
            "official_implementation": bool(args.official_implementation),
            "source_url": args.source_url,
            "source_commit": args.source_commit,
            "source_kind": args.implementation_variant,
            "source_manifest": str(Path(args.source_manifest)),
            "provenance_path": str(provenance_path or ""),
            "reference_key": BASELINES[args.method].reference_key,
            "dataset_id": dataset_id,
            "category": category,
            "external_baseline": True,
            "external_prediction_path": str(Path(args.predictions)),
            "import_latency_ms": (perf_counter() - start) * 1000.0,
        }
    )

    score_rows = []
    pareto_rows = []
    for idx, (label, score, mask, heatmap) in enumerate(zip(labels, image_scores, masks, heatmaps)):
        stem = f"{idx:05d}"
        path = str(paths[idx]) if idx < len(paths) else ""
        save_image(run_dir / "heatmaps" / f"{stem}.png", heatmap_to_uint8(heatmap))
        save_image(run_dir / "masks" / f"{stem}.png", (mask > 0).astype(np.uint8) * 255)
        image_dir = run_dir / "images" / stem
        image_dir.mkdir(parents=True, exist_ok=True)
        save_image(image_dir / "mask.png", (mask > 0).astype(np.uint8) * 255)
        np.savez_compressed(image_dir / "residual_heatmap.npz", residual=heatmap.astype(np.float32), final=heatmap.astype(np.float32))
        row = {
            "index": idx,
            "path": path,
            "label": int(label),
            "image_score": float(score),
            "latency_ms": float(args.latency_ms_mean),
            "nfe": float(args.nfe_mean),
            "ablation": "baseline",
            "method": args.method,
        }
        score_rows.append(row)
        pareto_rows.append({"index": idx, "latency_ms": float(args.latency_ms_mean), "nfe": float(args.nfe_mean), "image_score": float(score), "ablation": "baseline", "method": args.method})

    save_json(metrics, run_dir / "metrics.json")
    if provenance_path is not None:
        shutil.copy2(provenance_path, run_dir / "provenance.json")
    if threshold_policy:
        shutil.copy2(
            threshold_policy_path,
            run_dir / "pixel_threshold_policy.json",
        )
    _metric_csv(metrics, run_dir / "metrics.csv")
    _metric_csv(eff, run_dir / "efficiency.csv")
    np.savez_compressed(
        run_dir / "predictions.npz",
        labels=labels,
        image_scores=image_scores,
        masks=masks,
        **prediction_heatmap_payload(heatmaps, heatmaps, heatmaps),
        paths=paths,
        method=np.asarray(args.method),
        display_method=np.asarray(
            args.display_method or BASELINES[args.method].display_name
        ),
        implementation_variant=np.asarray(args.implementation_variant),
        official_implementation=np.asarray(
            bool(args.official_implementation)
        ),
        source_commit=np.asarray(args.source_commit),
    )
    with (run_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "path", "label", "image_score", "latency_ms", "nfe", "ablation", "method"])
        writer.writeheader()
        writer.writerows(score_rows)
    save_json([], run_dir / "roi_budget.json")
    np.save(run_dir / "crv_score_drop.npy", np.asarray([], dtype=np.float32))
    write_pareto(run_dir / "pareto.csv", pareto_rows)
    print(f"Imported {args.method} baseline outputs to {run_dir}")


if __name__ == "__main__":
    main()
