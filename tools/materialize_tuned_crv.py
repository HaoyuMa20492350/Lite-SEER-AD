from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.data.hard_negative_mining import ROI
from seer_ad_v2.evaluation.prediction_schema import prediction_heatmap_payload
from seer_ad_v2.evaluation.score_aggregation import IMAGE_SCORE_MODES, image_scores_from_heatmaps
from seer_ad_v2.models.counterfactual.repair_verification import apply_crv_to_heatmap

IMAGE_SCORE_SOURCES = ["final", "base", "feature", "feature_raw", "feature_raw_distance", "feature_raw_cosine"]
PIXEL_HEATMAP_SOURCES = ["final", "base", "residual", "feature", "feature_raw", "feature_raw_distance", "feature_raw_cosine", "score"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize a tuned-CRV prediction run from an existing feature CRV run.")
    p.add_argument("--source-run-dir", required=True)
    p.add_argument("--out-run-dir", required=True)
    p.add_argument("--recommendation", default=None, help="Path to recommended_crv_weight.json.")
    p.add_argument("--crv-weight", type=float, default=None, help="Override recommendation file.")
    p.add_argument("--base-key", choices=["auto", "residual", "feature", "final"], default="auto")
    p.add_argument("--image-score-mode", choices=IMAGE_SCORE_MODES, default="max_mean")
    p.add_argument("--image-score-source", choices=IMAGE_SCORE_SOURCES, default="final")
    p.add_argument("--pixel-heatmap-source", choices=PIXEL_HEATMAP_SOURCES, default="final")
    p.add_argument("--ablation-name", default="feature_tuned_crv")
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


def _heatmaps_from_source(run_dir: Path, count: int, source: str, *, final: np.ndarray, base: np.ndarray, score: np.ndarray | None = None) -> np.ndarray:
    if source == "final":
        return final
    if source == "base":
        return base
    if source == "score":
        if score is None:
            raise ValueError("score heatmaps are unavailable")
        return score
    if source in {"residual", "feature", "feature_raw", "feature_raw_distance", "feature_raw_cosine"}:
        return _image_npz_heatmaps(run_dir, count, source)
    raise ValueError(f"Unknown heatmap source: {source}")


def _crv_weight(args: argparse.Namespace) -> float:
    if args.crv_weight is not None:
        return float(args.crv_weight)
    if not args.recommendation:
        raise SystemExit("Provide --recommendation or --crv-weight")
    data = _load_json(Path(args.recommendation))
    if "crv_weight" not in data:
        raise KeyError(f"{args.recommendation} does not contain crv_weight")
    return float(data["crv_weight"])


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
    if not rows:
        return
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


def _write_run_args(
    source: Path,
    out: Path,
    ablation_name: str,
    crv_weight: float,
    image_score_mode: str,
    image_score_source: str,
    pixel_heatmap_source: str,
) -> None:
    payload = _load_json(source / "run_args.json")
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    if isinstance(args, dict):
        args = dict(args)
        args["run_name"] = out.name
        args["ablation"] = ablation_name
        args["crv_weight"] = crv_weight
        args["image_score_mode"] = image_score_mode
        args["image_score_source"] = image_score_source
        args["pixel_heatmap_source"] = pixel_heatmap_source
    payload = {"command": "materialize_tuned_crv", "args": args}
    (out / "run_args.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = Path(args.source_run_dir)
    out = Path(args.out_run_dir)
    out.mkdir(parents=True, exist_ok=True)
    crv_weight = _crv_weight(args)

    pred = np.load(source / "predictions.npz")
    labels = pred["labels"]
    masks = pred["masks"]
    paths = pred["paths"] if "paths" in pred.files else np.asarray([str(i) for i in range(len(labels))])
    roi_rows = _load_roi_budget(source / "roi_budget.json")
    by_image = _rows_by_image(roi_rows)
    base = _base_heatmaps(source, len(labels), args.base_key)

    fused: list[np.ndarray] = []
    for idx, heatmap in enumerate(base):
        rows = by_image.get(idx, [])
        rois = [_roi_from_row(row) for row in rows]
        drops = [float(row.get("sdr", 0.0)) for row in rows]
        fused.append(apply_crv_to_heatmap(heatmap, rois, drops, weight=crv_weight))
    heatmaps = np.stack(fused).astype(np.float32)
    score_heatmaps = _heatmaps_from_source(source, len(labels), args.image_score_source, final=heatmaps, base=base)
    image_scores = image_scores_from_heatmaps(score_heatmaps, mode=args.image_score_mode)
    pixel_heatmaps = _heatmaps_from_source(source, len(labels), args.pixel_heatmap_source, final=heatmaps, base=base, score=score_heatmaps)

    np.savez_compressed(
        out / "predictions.npz",
        labels=labels,
        image_scores=image_scores,
        masks=masks,
        **prediction_heatmap_payload(
            pixel_heatmaps,
            heatmaps,
            score_heatmaps,
        ),
        paths=paths,
        ablation=np.asarray(args.ablation_name),
    )
    for name in ["config.yaml", "roi_budget.json", "roi_budget.jsonl", "pareto.csv", "crv_score_drop.npy"]:
        _copy_if_exists(source / name, out / name)
    _write_scores(source, out, image_scores, args.ablation_name)
    _write_run_args(source, out, args.ablation_name, crv_weight, args.image_score_mode, args.image_score_source, args.pixel_heatmap_source)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "crv_weight": crv_weight,
                "image_score_mode": args.image_score_mode,
                "image_score_source": args.image_score_source,
                "pixel_heatmap_source": args.pixel_heatmap_source,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source": str(source),
                "out": str(out),
                "crv_weight": crv_weight,
                "base_key": args.base_key,
                "image_score_mode": args.image_score_mode,
                "image_score_source": args.image_score_source,
                "pixel_heatmap_source": args.pixel_heatmap_source,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
