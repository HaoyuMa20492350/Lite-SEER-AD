from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.prediction_schema import (
    prediction_heatmap_payload,
    resolve_prediction_heatmaps,
)


METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select a pixel policy on a validation split and evaluate it on a held-out split.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--categories", required=True, help="Comma-separated categories.")
    p.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate as name=run/template/with/{category}. Repeat for each candidate.",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--val-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--select-metric", choices=["quality", "pixel_ap", "dice", "aupro", "pixel_auroc"], default="quality")
    p.add_argument("--materialize", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_candidates(values: list[str]) -> list[tuple[str, str]]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Candidate must be name=template: {value}")
        name, template = value.split("=", 1)
        out.append((name.strip(), template.strip()))
    return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _quality(metrics: dict[str, float], select_metric: str) -> float:
    if select_metric == "quality":
        vals = [metrics.get("pixel_ap", np.nan), metrics.get("dice", np.nan), metrics.get("aupro", np.nan)]
        vals = [float(v) for v in vals if np.isfinite(float(v))]
        return float(sum(vals)) if vals else float("-inf")
    value = float(metrics.get(select_metric, float("nan")))
    return value if np.isfinite(value) else float("-inf")


def _metrics(pred: np.lib.npyio.NpzFile, indices: np.ndarray) -> dict[str, float]:
    return detection_metrics(
        pred["labels"][indices],
        pred["image_scores"][indices],
        pred["masks"][indices],
        pred["heatmaps"][indices],
    )


def _stable_seed(seed: int, category: str, label: int) -> int:
    digest = hashlib.sha256(f"{seed}:{category}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def stratified_split(labels: np.ndarray, category: str, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    val_ratio = min(max(float(val_ratio), 0.1), 0.9)
    labels = labels.astype(np.uint8).reshape(-1)
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(set(int(v) for v in labels.tolist())):
        idx = np.where(labels == label)[0]
        rng = np.random.default_rng(_stable_seed(seed, category, label))
        idx = idx.copy()
        rng.shuffle(idx)
        if len(idx) <= 1:
            val_count = len(idx)
        else:
            val_count = int(round(len(idx) * val_ratio))
            val_count = min(max(1, val_count), len(idx) - 1)
        val_parts.append(idx[:val_count])
        test_parts.append(idx[val_count:])
    val_idx = np.concatenate(val_parts) if val_parts else np.asarray([], dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.asarray([], dtype=np.int64)
    return np.sort(val_idx), np.sort(test_idx)


def _assert_compatible(ref: np.lib.npyio.NpzFile, cur: np.lib.npyio.NpzFile, path: Path) -> None:
    if len(ref["labels"]) != len(cur["labels"]):
        raise ValueError(f"{path} has different sample count")
    if not np.array_equal(ref["labels"], cur["labels"]):
        raise ValueError(f"{path} labels do not match the first candidate")
    if "paths" in ref.files and "paths" in cur.files and not np.array_equal(ref["paths"], cur["paths"]):
        raise ValueError(f"{path} paths do not match the first candidate")


def _row(prefix: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    out = dict(prefix)
    for metric in METRICS:
        out[metric] = metrics.get(metric)
    out["f1"] = metrics.get("f1")
    out["iou"] = metrics.get("iou")
    out["threshold"] = metrics.get("threshold")
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def materialize_selected(out_dir: Path, category: str, selected_name: str, selected_path: Path, pred: np.lib.npyio.NpzFile, test_idx: np.ndarray) -> None:
    run_dir = out_dir / "selected_runs" / f"{category}_{selected_name}_heldout"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(pred, test_idx)
    detection, verification, image_score_heatmaps = resolve_prediction_heatmaps(
        pred
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        labels=pred["labels"][test_idx],
        image_scores=pred["image_scores"][test_idx],
        masks=pred["masks"][test_idx],
        **prediction_heatmap_payload(
            detection[test_idx],
            verification[test_idx],
            image_score_heatmaps[test_idx],
        ),
        paths=pred["paths"][test_idx] if "paths" in pred.files else test_idx.astype(str),
        ablation=np.asarray("oracle_heldout_selected_pixel_policy"),
    )
    source_args = _load_json(selected_path / "run_args.json")
    payload_args = source_args.get("args", {}) if isinstance(source_args, dict) else {}
    if isinstance(payload_args, dict):
        payload_args = dict(payload_args)
        payload_args["run_name"] = run_dir.name
        payload_args["ablation"] = "oracle_heldout_selected_pixel_policy"
        payload_args["selected_policy"] = selected_name
        payload_args["selected_source_run"] = str(selected_path)
        payload_args["selection_protocol"] = "oracle_upper_bound_with_real_anomaly_masks"
        payload_args["uses_real_anomaly_labels_for_selection"] = True
        payload_args["uses_real_anomaly_masks_for_selection"] = True
        payload_args["paper_role"] = "oracle_upper_bound_only"
    (run_dir / "run_args.json").write_text(
        json.dumps({"command": "select_pixel_policy_with_val_split", "args": payload_args}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "selection_evidence.json").write_text(
        json.dumps(
            {
                "selection_protocol": "oracle_upper_bound_with_real_anomaly_masks",
                "uses_real_anomaly_labels_for_selection": True,
                "uses_real_anomaly_masks_for_selection": True,
                "paper_role": "oracle_upper_bound_only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    safe = {k: (v.item() if isinstance(v, np.generic) else None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()}
    (run_dir / "metrics.json").write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = parse_candidates(args.candidate)
    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_test_rows: list[dict[str, Any]] = []

    for category in split_csv(args.categories):
        loaded: list[tuple[str, Path, np.lib.npyio.NpzFile, dict[str, Any]]] = []
        ref_pred: np.lib.npyio.NpzFile | None = None
        for name, template in candidates:
            run_dir = Path(template.format(category=category))
            pred_path = run_dir / "predictions.npz"
            if not pred_path.exists():
                continue
            pred = np.load(pred_path)
            if ref_pred is None:
                ref_pred = pred
            else:
                _assert_compatible(ref_pred, pred, pred_path)
            metrics_meta = _load_json(run_dir / "metrics.json")
            loaded.append((name, run_dir, pred, metrics_meta))
        if not loaded or ref_pred is None:
            continue
        val_idx, test_idx = stratified_split(ref_pred["labels"], category, args.val_ratio, args.seed)
        if len(test_idx) == 0:
            continue
        scored: list[tuple[float, str, Path, np.lib.npyio.NpzFile, dict[str, float], dict[str, float], dict[str, Any]]] = []
        for name, run_dir, pred, meta in loaded:
            val_metrics = _metrics(pred, val_idx)
            test_metrics = _metrics(pred, test_idx)
            score = _quality(val_metrics, args.select_metric)
            scored.append((score, name, run_dir, pred, val_metrics, test_metrics, meta))
            candidate_rows.append(
                _row(
                    {
                        "dataset": args.dataset,
                        "category": category,
                        "candidate": name,
                        "run": str(run_dir),
                        "split": "val",
                        "selection_score": score,
                        "pixel_heatmap_source": meta.get("pixel_heatmap_source", ""),
                    },
                    val_metrics,
                )
            )
            candidate_rows.append(
                _row(
                    {
                        "dataset": args.dataset,
                        "category": category,
                        "candidate": name,
                        "run": str(run_dir),
                        "split": "heldout_test",
                        "selection_score": score,
                        "pixel_heatmap_source": meta.get("pixel_heatmap_source", ""),
                    },
                    test_metrics,
                )
            )
        best = max(scored, key=lambda item: item[0])
        score, selected_name, selected_path, selected_pred, selected_val, selected_test, selected_meta = best
        selection_rows.append(
            {
                "dataset": args.dataset,
                "category": category,
                "selected_candidate": selected_name,
                "selected_run": str(selected_path),
                "selection_score": score,
                "val_count": int(len(val_idx)),
                "test_count": int(len(test_idx)),
                "val_indices": " ".join(str(int(i)) for i in val_idx),
                "test_indices": " ".join(str(int(i)) for i in test_idx),
                "pixel_heatmap_source": selected_meta.get("pixel_heatmap_source", ""),
                "selection_protocol": "oracle_upper_bound_with_real_anomaly_masks",
                "paper_role": "oracle_upper_bound_only",
                "uses_real_anomaly_labels_for_selection": True,
                "uses_real_anomaly_masks_for_selection": True,
            }
        )
        selected_test_rows.append(
            _row(
                {
                    "dataset": args.dataset,
                    "category": category,
                    "candidate": selected_name,
                    "run": str(selected_path),
                    "split": "heldout_test",
                    "selection_score": score,
                    "pixel_heatmap_source": selected_meta.get("pixel_heatmap_source", ""),
                },
                selected_test,
            )
        )
        if args.materialize:
            materialize_selected(out_dir, category, selected_name, selected_path, selected_pred, test_idx)

    fields = [
        "dataset",
        "category",
        "candidate",
        "run",
        "split",
        "selection_score",
        "pixel_heatmap_source",
        *METRICS,
        "f1",
        "iou",
        "threshold",
    ]
    write_csv(out_dir / "candidate_split_metrics.csv", candidate_rows, fields)
    write_csv(
        out_dir / "selected_heldout_metrics.csv",
        selected_test_rows,
        fields,
    )
    write_csv(
        out_dir / "selection.csv",
        selection_rows,
        [
            "dataset",
            "category",
            "selected_candidate",
            "selected_run",
            "selection_score",
            "val_count",
            "test_count",
            "pixel_heatmap_source",
            "selection_protocol",
            "paper_role",
            "uses_real_anomaly_labels_for_selection",
            "uses_real_anomaly_masks_for_selection",
            "val_indices",
            "test_indices",
        ],
    )
    means: dict[str, Any] = {"dataset": args.dataset, "categories": len(selected_test_rows)}
    for metric in METRICS:
        vals = [float(row[metric]) for row in selected_test_rows if row.get(metric) not in {None, ""} and np.isfinite(float(row[metric]))]
        means[metric] = float(np.mean(vals)) if vals else None
    summary = {
        "dataset": args.dataset,
        "select_metric": args.select_metric,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "selection_protocol": "oracle_upper_bound_with_real_anomaly_masks",
        "paper_role": "oracle_upper_bound_only",
        "uses_real_anomaly_labels_for_selection": True,
        "uses_real_anomaly_masks_for_selection": True,
        "means": means,
        "selected_counts": {
            name: sum(1 for row in selection_rows if row["selected_candidate"] == name)
            for name, _ in candidates
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
