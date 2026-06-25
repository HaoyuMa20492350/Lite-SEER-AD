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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.prediction_schema import (
    prediction_heatmap_payload,
    resolve_prediction_heatmaps,
)


DEFAULT_MODES = [
    "raw",
    "gaussian:1",
    "gaussian:2",
    "gaussian:3",
    "highpass:5",
    "highpass:9",
    "highpass:15",
    "tophat:5",
    "tophat:9",
    "tophat:15",
    "closing:3",
    "closing:5",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search simple pixel heatmap post-processing modes from a saved predictions.npz.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--heatmap-key", default="heatmaps")
    p.add_argument("--modes", default=",".join(DEFAULT_MODES))
    p.add_argument("--out", default=None)
    p.add_argument("--materialize", action="store_true")
    p.add_argument("--ablation-prefix", default="feature_pixelpost")
    return p.parse_args()


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    k = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)


def _apply_one(img: np.ndarray, mode: str) -> np.ndarray:
    img = img.astype(np.float32)
    if mode == "raw":
        out = img
    elif mode.startswith("gaussian:"):
        out = _gaussian(img, float(mode.split(":", 1)[1]))
    elif mode.startswith("highpass:"):
        sigma = float(mode.split(":", 1)[1])
        out = np.maximum(img - _gaussian(img, sigma), 0.0)
    elif mode.startswith("tophat:"):
        size = int(float(mode.split(":", 1)[1]))
        out = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, _kernel(size))
    elif mode.startswith("closing:"):
        size = int(float(mode.split(":", 1)[1]))
        out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, _kernel(size))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def apply_mode(heatmaps: np.ndarray, mode: str) -> np.ndarray:
    return np.stack([_apply_one(h, mode) for h in heatmaps]).astype(np.float32)


def _quality(metrics: dict[str, float]) -> float:
    return float(metrics["pixel_ap"]) + float(metrics["aupro"]) + float(metrics["dice"])


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_metric_csv(metrics: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _json_safe(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        out[key] = value
    return out


def _materialize(run_dir: Path, out_dir: Path, pred: np.lib.npyio.NpzFile, heatmaps: np.ndarray, mode: str, prefix: str, metrics: dict[str, Any]) -> None:
    ablation = f"{prefix}_{mode.replace(':', '')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _source_detection, source_verification, source_score = (
        resolve_prediction_heatmaps(pred)
    )
    np.savez_compressed(
        out_dir / "predictions.npz",
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
        _copy_if_exists(run_dir / name, out_dir / name)
    args_payload = _load_json(run_dir / "run_args.json")
    args = args_payload.get("args", {}) if isinstance(args_payload, dict) else {}
    if isinstance(args, dict):
        args = dict(args)
        args["run_name"] = out_dir.name
        args["ablation"] = ablation
        args["pixel_heatmap_source"] = f"postprocess_{mode}"
    (out_dir / "run_args.json").write_text(json.dumps({"command": "search_pixel_postprocess", "args": args}, indent=2), encoding="utf-8")
    safe = _json_safe(metrics)
    (out_dir / "metrics.json").write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "eval_metrics.json").write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")
    _write_metric_csv(safe, out_dir / "metrics.csv")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = np.load(run_dir / "predictions.npz")
    if args.heatmap_key not in pred.files:
        raise KeyError(f"{run_dir / 'predictions.npz'} does not contain heatmap key '{args.heatmap_key}'")
    base = np.asarray(pred[args.heatmap_key], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for mode in _split_csv(args.modes):
        heatmaps = apply_mode(base, mode)
        metrics = detection_metrics(pred["labels"], pred["image_scores"], pred["masks"], heatmaps)
        metrics["pixel_postprocess_mode"] = mode
        metrics["pixel_heatmap_source"] = f"{args.heatmap_key}_postprocess_{mode}"
        row = {
            "mode": mode,
            "image_auroc": metrics["image_auroc"],
            "pixel_auroc": metrics["pixel_auroc"],
            "pixel_ap": metrics["pixel_ap"],
            "aupro": metrics["aupro"],
            "dice": metrics["dice"],
            "quality": _quality(metrics),
        }
        rows.append(row)
        if args.materialize:
            safe_name = mode.replace(":", "")
            _materialize(run_dir, out_dir / f"{args.ablation_prefix}_{safe_name}", pred, heatmaps, mode, args.ablation_prefix, metrics)
    fields = ["mode", "image_auroc", "pixel_auroc", "pixel_ap", "aupro", "dice", "quality"]
    with (out_dir / "pixel_postprocess_search.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: float(row["quality"]))
    (out_dir / "recommended_pixel_postprocess.json").write_text(json.dumps(best, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "recommended": best}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
