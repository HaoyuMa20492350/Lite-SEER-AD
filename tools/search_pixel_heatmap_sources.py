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

from seer_ad_v2.evaluation.metrics_detection import detection_metrics


DEFAULT_KEYS = ["predictions", "residual", "feature", "feature_raw", "feature_raw_distance", "feature_raw_cosine", "final", "score"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search pixel-level heatmap sources from a saved Lite-SEER-AD run.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    p.add_argument("--out", default=None)
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _image_npz_heatmaps(run_dir: Path, count: int, key: str) -> np.ndarray:
    maps: list[np.ndarray] = []
    for idx in range(count):
        npz_path = run_dir / "images" / f"{idx:05d}" / "residual_heatmap.npz"
        data = np.load(npz_path)
        if key not in data:
            raise KeyError(f"{npz_path} does not contain heatmap key '{key}'. Available: {list(data.keys())}")
        maps.append(np.asarray(data[key], dtype=np.float32))
    return np.stack(maps).astype(np.float32)


def _quality(row: dict[str, Any]) -> float:
    return float(row["pixel_ap"]) + float(row["dice"]) + float(row["aupro"])


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = np.load(run_dir / "predictions.npz")
    labels = pred["labels"]
    scores = pred["image_scores"]
    masks = pred["masks"]
    rows: list[dict[str, Any]] = []
    for key in split_csv(args.keys):
        try:
            if key == "predictions":
                heatmaps = pred["heatmaps"]
            else:
                heatmaps = _image_npz_heatmaps(run_dir, len(labels), key)
        except KeyError:
            continue
        metrics = detection_metrics(labels, scores, masks, heatmaps)
        rows.append(
            {
                "heatmap_source": key,
                "image_auroc": metrics["image_auroc"],
                "pixel_auroc": metrics["pixel_auroc"],
                "pixel_ap": metrics["pixel_ap"],
                "aupro": metrics["aupro"],
                "dice": metrics["dice"],
                "quality": metrics["pixel_ap"] + metrics["dice"] + metrics["aupro"],
            }
        )
    best = max(rows, key=_quality)
    fields = ["heatmap_source", "image_auroc", "pixel_auroc", "pixel_ap", "aupro", "dice", "quality"]
    with (out_dir / "pixel_heatmap_source_search.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "recommended_pixel_heatmap_source.json").write_text(json.dumps(best, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "recommended": best}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
