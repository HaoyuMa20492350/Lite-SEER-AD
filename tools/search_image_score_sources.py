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
from seer_ad_v2.evaluation.metrics_detection import _safe_ap, _safe_auc, aupro_score, best_f1_iou_dice
from seer_ad_v2.evaluation.score_aggregation import IMAGE_SCORE_MODES, image_scores_from_heatmaps
from seer_ad_v2.models.counterfactual.repair_verification import apply_crv_to_heatmap


IMAGE_SCORE_SOURCES = [
    "final",
    "base",
    "feature",
    "feature_raw",
    "feature_raw_distance",
    "feature_raw_cosine",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search image score source/mode combinations from saved Lite-SEER-AD maps.")
    p.add_argument("--source-run-dir", required=True, help="Run containing predictions and per-image residual_heatmap.npz files.")
    p.add_argument("--recommendation", default=None, help="Path to recommended_crv_weight.json.")
    p.add_argument("--crv-weight", type=float, default=None, help="Override recommendation file.")
    p.add_argument("--base-key", choices=["auto", "residual", "feature", "final"], default="auto")
    p.add_argument("--sources", default="feature_raw,feature_raw_cosine,final,base,feature")
    p.add_argument("--modes", default=",".join(IMAGE_SCORE_MODES))
    p.add_argument("--out", default=None)
    return p.parse_args()


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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


def _crv_weight(args: argparse.Namespace) -> float:
    if args.crv_weight is not None:
        return float(args.crv_weight)
    if not args.recommendation:
        raise SystemExit("Provide --recommendation or --crv-weight")
    data = _load_json(Path(args.recommendation))
    if "crv_weight" not in data:
        raise KeyError(f"{args.recommendation} does not contain crv_weight")
    return float(data["crv_weight"])


def _final_heatmaps(source: Path, count: int, base_key: str, crv_weight: float) -> np.ndarray:
    base = _base_heatmaps(source, count, base_key)
    roi_rows = _load_roi_budget(source / "roi_budget.json")
    by_image = _rows_by_image(roi_rows)
    fused: list[np.ndarray] = []
    for idx, heatmap in enumerate(base):
        rows = by_image.get(idx, [])
        rois = [_roi_from_row(row) for row in rows]
        drops = [float(row.get("sdr", 0.0)) for row in rows]
        fused.append(apply_crv_to_heatmap(heatmap, rois, drops, weight=crv_weight))
    return np.stack(fused).astype(np.float32)


def _score_heatmaps(source: Path, final_heatmaps: np.ndarray, base_heatmaps: np.ndarray, source_name: str) -> np.ndarray:
    if source_name == "final":
        return final_heatmaps
    if source_name == "base":
        return base_heatmaps
    if source_name in {"feature", "feature_raw", "feature_raw_distance", "feature_raw_cosine"}:
        return _image_npz_heatmaps(source, len(final_heatmaps), source_name)
    raise ValueError(f"Unknown image score source: {source_name}")


def main() -> None:
    args = parse_args()
    source = Path(args.source_run_dir)
    out_dir = Path(args.out) if args.out else source
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = np.load(source / "predictions.npz")
    labels = pred["labels"]
    masks = pred["masks"]
    crv_weight = _crv_weight(args)
    base = _base_heatmaps(source, len(labels), args.base_key)
    final_heatmaps = _final_heatmaps(source, len(labels), args.base_key, crv_weight)
    pixel_true = masks.reshape(-1).astype(np.uint8)
    pixel_score = final_heatmaps.reshape(-1)
    pixel_metrics = {
        "pixel_auroc": _safe_auc(pixel_true, pixel_score),
        "pixel_ap": _safe_ap(pixel_true, pixel_score),
        "aupro": aupro_score(masks, final_heatmaps),
    }
    pixel_metrics.update(best_f1_iou_dice(pixel_true, pixel_score))

    rows: list[dict[str, Any]] = []
    for source_name in _split(args.sources):
        if source_name not in IMAGE_SCORE_SOURCES:
            raise ValueError(f"Unknown image score source: {source_name}")
        score_heatmaps = _score_heatmaps(source, final_heatmaps, base, source_name)
        for mode in _split(args.modes):
            scores = image_scores_from_heatmaps(score_heatmaps, mode=mode)
            rows.append(
                {
                    "image_score_source": source_name,
                    "image_score_mode": mode,
                    "crv_weight": crv_weight,
                    "image_auroc": _safe_auc(labels, scores),
                    "pixel_auroc": pixel_metrics["pixel_auroc"],
                    "pixel_ap": pixel_metrics["pixel_ap"],
                    "aupro": pixel_metrics["aupro"],
                    "dice": pixel_metrics["dice"],
                }
            )

    best = max(rows, key=lambda row: float(row["image_auroc"]))
    fields = [
        "image_score_source",
        "image_score_mode",
        "crv_weight",
        "image_auroc",
        "pixel_auroc",
        "pixel_ap",
        "aupro",
        "dice",
    ]
    with (out_dir / "image_score_source_search.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "recommended_image_score_source.json").write_text(json.dumps(best, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(source), "recommended": best}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
