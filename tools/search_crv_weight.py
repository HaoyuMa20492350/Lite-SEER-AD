from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.data.hard_negative_mining import ROI
from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.score_aggregation import IMAGE_SCORE_MODES, image_scores_from_heatmaps
from seer_ad_v2.models.counterfactual.repair_verification import apply_crv_to_heatmap

IMAGE_SCORE_SOURCES = ["final", "base", "feature", "feature_raw", "feature_raw_distance", "feature_raw_cosine"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline CRV fusion weight search from an existing full run.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--weights", default="0.25,0.5,1.0")
    p.add_argument("--base-key", choices=["auto", "residual", "feature", "final"], default="auto")
    p.add_argument("--image-score-mode", choices=IMAGE_SCORE_MODES, default="max_mean")
    p.add_argument("--image-score-source", choices=IMAGE_SCORE_SOURCES, default="final")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _weights(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _load_roi_budget(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _roi_from_row(row: dict[str, Any]) -> ROI:
    x1, y1, x2, y2 = [int(v) for v in row.get("bbox", [0, 0, 0, 0])]
    return ROI(x1=x1, y1=y1, x2=x2, y2=y2, area=max(0, (x2 - x1) * (y2 - y1)), peak=float(row.get("residual_score", 0.0)))


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


def _image_npz_heatmaps(run_dir: Path, count: int, key: str) -> np.ndarray:
    heatmaps: list[np.ndarray] = []
    for idx in range(count):
        npz_path = run_dir / "images" / f"{idx:05d}" / "residual_heatmap.npz"
        data = np.load(npz_path)
        if key not in data:
            raise KeyError(f"{npz_path} does not contain heatmap key '{key}'. Available: {list(data.keys())}")
        heatmaps.append(np.asarray(data[key], dtype=np.float32))
    return np.stack(heatmaps).astype(np.float32)


def _score_heatmaps(run_dir: Path, count: int, source: str, base: np.ndarray) -> np.ndarray | None:
    if source == "final":
        return None
    if source == "base":
        return base
    if source == "feature":
        return _image_npz_heatmaps(run_dir, count, "feature")
    if source == "feature_raw":
        return _image_npz_heatmaps(run_dir, count, "feature_raw")
    if source == "feature_raw_distance":
        return _image_npz_heatmaps(run_dir, count, "feature_raw_distance")
    if source == "feature_raw_cosine":
        return _image_npz_heatmaps(run_dir, count, "feature_raw_cosine")
    raise ValueError(f"Unknown image score source: {source}")


def _rows_by_image(roi_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for row in roi_rows:
        out.setdefault(int(row.get("image_index", -1)), []).append(row)
    return out


def _quality(metrics: dict[str, float]) -> float:
    vals = [metrics.get("pixel_ap", float("nan")), metrics.get("dice", float("nan")), metrics.get("aupro", float("nan"))]
    vals = [float(v) for v in vals if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = np.load(run_dir / "predictions.npz")
    labels = pred["labels"]
    masks = pred["masks"]
    roi_rows = _load_roi_budget(run_dir / "roi_budget.json")
    by_image = _rows_by_image(roi_rows)
    base = _base_heatmaps(run_dir, len(labels), args.base_key)
    fixed_score_heatmaps = _score_heatmaps(run_dir, len(labels), args.image_score_source, base)

    results: list[dict[str, float]] = []
    for weight in _weights(args.weights):
        fused: list[np.ndarray] = []
        for idx, heatmap in enumerate(base):
            rows = by_image.get(idx, [])
            rois = [_roi_from_row(row) for row in rows]
            drops = [float(row.get("sdr", 0.0)) for row in rows]
            fused.append(apply_crv_to_heatmap(heatmap, rois, drops, weight=weight))
        heatmaps = np.stack(fused).astype(np.float32)
        score_heatmaps = heatmaps if fixed_score_heatmaps is None else fixed_score_heatmaps
        image_scores = image_scores_from_heatmaps(score_heatmaps, mode=args.image_score_mode)
        metrics = detection_metrics(labels, image_scores, masks, heatmaps)
        results.append(
            {
                "crv_weight": weight,
                "image_auroc": metrics["image_auroc"],
                "pixel_auroc": metrics["pixel_auroc"],
                "pixel_ap": metrics["pixel_ap"],
                "aupro": metrics["aupro"],
                "dice": metrics["dice"],
                "quality": _quality(metrics),
                "base_key": args.base_key,
                "image_score_mode": args.image_score_mode,
                "image_score_source": args.image_score_source,
            }
        )
    best = max(results, key=lambda row: row["quality"] if np.isfinite(row["quality"]) else -1.0)
    csv_path = out_dir / "crv_weight_search.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crv_weight",
                "image_auroc",
                "pixel_auroc",
                "pixel_ap",
                "aupro",
                "dice",
                "quality",
                "base_key",
                "image_score_mode",
                "image_score_source",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    (out_dir / "recommended_crv_weight.json").write_text(json.dumps(best, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "recommended": best}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
