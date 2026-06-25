from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.metrics_plan import efficiency_summary, plan_metric_summary
from seer_ad_v2.evaluation.pixel_threshold_policy import (
    load_pixel_threshold_policy,
)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved Lite-SEER-AD predictions.")
    p.add_argument("--predictions", default=None)
    p.add_argument("--pred_dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--pixel-threshold-policy", default=None)
    p.add_argument("--require-fixed-threshold", action="store_true")
    return p.parse_args()


def _write_metric_csv(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, v in metrics.items():
            writer.writerow({"metric": k, "value": v})


def _read_roi_budget(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_pareto(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_metric_value(pred_dir: Path, key: str) -> float | None:
    metrics_path = pred_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get(key) is not None:
                return float(metrics[key])
        except Exception:
            pass
    return None


def _read_metric_payload(pred_dir: Path) -> dict[str, Any]:
    metrics_path = pred_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_crv_weight(pred_dir: Path) -> float | None:
    value = _read_metric_value(pred_dir, "crv_weight")
    if value is not None:
        return value
    args_path = pred_dir / "run_args.json"
    if args_path.exists():
        try:
            payload = json.loads(args_path.read_text(encoding="utf-8"))
            args = payload.get("args", {}) if isinstance(payload, dict) else {}
            if args.get("crv_weight") is not None:
                return float(args["crv_weight"])
        except Exception:
            pass
    return None


def _roi_area_by_image(roi_rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    by_image: dict[int, dict[str, float]] = {}
    for row in roi_rows:
        image_index = int(row.get("image_index", -1))
        if image_index < 0:
            continue
        bucket = by_image.setdefault(image_index, {"local_region_ratio": 0.0, "repaired_area_ratio": 0.0})
        area_ratio = float(row.get("area_ratio", 0.0))
        bucket["local_region_ratio"] += area_ratio
        if int(float(row.get("nfe", 0.0))) > 0 or str(row.get("scheduler_action", "")) != "skip":
            bucket["repaired_area_ratio"] += area_ratio
    for bucket in by_image.values():
        bucket["local_region_ratio"] = min(1.0, bucket["local_region_ratio"])
        bucket["repaired_area_ratio"] = min(1.0, bucket["repaired_area_ratio"])
    return by_image


def _read_efficiency_from_scores(path: Path, roi_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    roi_areas = _roi_area_by_image(roi_rows or [])
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_index = int(float(row.get("index", len(rows))))
            areas = roi_areas.get(image_index, {})
            rows.append(
                {
                    "latency_ms": float(row.get("latency_ms", 0.0)),
                    "nfe": float(row.get("nfe", 0.0)),
                    "repaired_area_ratio": float(areas.get("repaired_area_ratio", 0.0)),
                    "local_region_ratio": float(areas.get("local_region_ratio", 0.0)),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if not args.predictions and not args.pred_dir:
        raise SystemExit("Provide --predictions or --pred_dir")
    pred_dir = Path(args.pred_dir) if args.pred_dir else Path(args.predictions).parent
    pred_path = Path(args.predictions) if args.predictions else pred_dir / "predictions.npz"
    data = np.load(pred_path)
    existing_metrics = _read_metric_payload(pred_dir)
    policy_path = (
        Path(args.pixel_threshold_policy)
        if args.pixel_threshold_policy
        else pred_dir / "pixel_threshold_policy.json"
    )
    threshold_policy = (
        load_pixel_threshold_policy(policy_path) if policy_path.exists() else None
    )
    if args.require_fixed_threshold and threshold_policy is None:
        raise FileNotFoundError(
            f"Fixed pixel threshold policy is required but missing: {policy_path}"
        )
    metrics = detection_metrics(
        data["labels"],
        data["image_scores"],
        data["masks"],
        data["heatmaps"],
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
        metrics["pixel_threshold_policy_path"] = str(policy_path)
        metrics["max_normal_pixel_fpr"] = threshold_policy.get(
            "max_normal_pixel_fpr"
        )
        metrics["threshold_evidence_normal_fpr"] = threshold_policy.get(
            "observed_normal_pixel_fpr"
        )
        metrics["threshold_uses_real_anomaly_labels"] = threshold_policy.get(
            "uses_real_anomaly_labels", False
        )
        metrics["threshold_uses_real_anomaly_masks"] = threshold_policy.get(
            "uses_real_anomaly_masks", False
        )
    roi_rows = _read_roi_budget(pred_dir / "roi_budget.json")
    if not roi_rows:
        roi_rows = _read_roi_budget(pred_dir / "roi_budget.jsonl")
    drops_path = pred_dir / "crv_score_drop.npy"
    drops = np.load(drops_path) if drops_path.exists() else np.asarray([], dtype=np.float32)
    metrics.update(
        plan_metric_summary(
            data["masks"],
            data["heatmaps"],
            roi_rows,
            drops,
            _read_pareto(pred_dir / "pareto.csv"),
            data["labels"],
        )
    )
    crv_weight = _read_crv_weight(pred_dir)
    if crv_weight is not None:
        metrics["crv_weight"] = crv_weight
    for key in ["reconstruction_steps", "prototype_heatmap_weight"]:
        value = _read_metric_value(pred_dir, key)
        if value is not None:
            metrics[key] = value
    for key in [
        "method",
        "method_id",
        "display_method",
        "implementation_variant",
        "official_implementation",
        "source_path",
        "source_url",
        "source_commit",
        "reference_key",
        "category",
        "external_baseline",
        "external_prediction_path",
        "image_score_mode",
        "image_score_source",
        "pixel_heatmap_source",
    ]:
        if key in existing_metrics:
            metrics[key] = existing_metrics[key]
    eff = efficiency_summary(
        _read_efficiency_from_scores(pred_dir / "scores.csv", roi_rows)
    )
    latency_benchmark = _read_json_payload(
        pred_dir / "latency_benchmark.json"
    )
    if latency_benchmark:
        eff.update(latency_benchmark)
        metrics.update(latency_benchmark)
    safe_metrics = json_safe(metrics)
    safe_eff = json_safe(eff)
    out_path = Path(args.out) if args.out else pred_dir / "eval_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(safe_metrics, indent=2, allow_nan=False), encoding="utf-8")
    (pred_dir / "metrics.json").write_text(json.dumps(safe_metrics, indent=2, allow_nan=False), encoding="utf-8")
    _write_metric_csv(safe_metrics, pred_dir / "metrics.csv")
    _write_metric_csv(safe_eff, pred_dir / "efficiency.csv")
    print(json.dumps(safe_metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
