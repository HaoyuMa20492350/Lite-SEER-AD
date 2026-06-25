from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.score_aggregation import IMAGE_SCORE_MODES, image_scores_from_heatmaps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search image-level score aggregation modes from saved heatmaps.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--modes", default=",".join(IMAGE_SCORE_MODES))
    p.add_argument("--heatmap-key", default="heatmaps", help="Array key in predictions.npz, e.g. heatmaps or score_heatmaps.")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = np.load(run_dir / "predictions.npz")
    labels = pred["labels"]
    masks = pred["masks"]
    score_heatmaps = pred[args.heatmap_key] if args.heatmap_key in pred.files else pred["heatmaps"]
    eval_heatmaps = pred["heatmaps"]
    rows = []
    for mode in _split(args.modes):
        scores = image_scores_from_heatmaps(score_heatmaps, mode=mode)
        metrics = detection_metrics(labels, scores, masks, eval_heatmaps)
        rows.append(
            {
                "image_score_mode": mode,
                "heatmap_key": args.heatmap_key if args.heatmap_key in pred.files else "heatmaps",
                "image_auroc": metrics["image_auroc"],
                "pixel_auroc": metrics["pixel_auroc"],
                "pixel_ap": metrics["pixel_ap"],
                "aupro": metrics["aupro"],
                "dice": metrics["dice"],
            }
        )
    best = max(rows, key=lambda row: float(row["image_auroc"]))
    fields = ["image_score_mode", "heatmap_key", "image_auroc", "pixel_auroc", "pixel_ap", "aupro", "dice"]
    with (out_dir / "image_score_mode_search.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "recommended_image_score_mode.json").write_text(json.dumps(best, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "recommended": best}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
