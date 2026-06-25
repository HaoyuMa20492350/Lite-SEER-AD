from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.metrics_detection import detection_metrics


METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare held-out selected Lite-SEER-AD runs against baselines on the same sample paths.")
    p.add_argument("--heldout", action="append", required=True, help="dataset=heldout_selector_dir")
    p.add_argument("--baseline-table", action="append", required=True, help="dataset=baseline_table.csv")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", required=True)
    return p.parse_args()


def parse_mapping(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected dataset=path: {value}")
        dataset, path = value.split("=", 1)
        out[dataset.strip()] = Path(path.strip())
    return out


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm_path(value: Any) -> str:
    return str(value).replace("/", "\\").lower()


def to_float(row: dict[str, Any], key: str) -> float:
    try:
        value = row.get(key)
        if value in {"", None}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = finite([to_float(row, key) for row in rows])
    return float(sum(vals) / len(vals)) if vals else None


def metrics_from_npz(pred: np.lib.npyio.NpzFile, indices: np.ndarray | None = None) -> dict[str, float]:
    if indices is None:
        indices = np.arange(len(pred["labels"]))
    return detection_metrics(
        pred["labels"][indices],
        pred["image_scores"][indices],
        pred["masks"][indices],
        pred["heatmaps"][indices],
    )


def row_with_metrics(prefix: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    out = dict(prefix)
    for metric in METRICS:
        out[metric] = metrics.get(metric)
    out["f1"] = metrics.get("f1")
    out["iou"] = metrics.get("iou")
    out["threshold"] = metrics.get("threshold")
    return out


def method_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("source", "")), str(row.get("method", ""))), []).append(row)
    out = []
    for (dataset, source, method), items in sorted(grouped.items()):
        item: dict[str, Any] = {
            "dataset": dataset,
            "source": source,
            "method": method,
            "categories": len({row.get("category", "") for row in items}),
        }
        for metric in METRICS:
            item[metric] = mean(items, metric)
        out.append(item)
    return out


def best_baselines(rows: list[dict[str, Any]], dataset: str, category: str) -> dict[str, dict[str, Any]]:
    candidates = [row for row in rows if row.get("dataset") == dataset and row.get("category") == category]
    out: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        vals = [row for row in candidates if math.isfinite(to_float(row, metric))]
        if vals:
            out[metric] = max(vals, key=lambda row: to_float(row, metric))
    return out


def category_deltas(ours_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for ours in sorted(ours_rows, key=lambda row: (str(row.get("dataset", "")), str(row.get("category", "")))):
        dataset = str(ours.get("dataset", ""))
        category = str(ours.get("category", ""))
        best = best_baselines(baseline_rows, dataset, category)
        out: dict[str, Any] = {"dataset": dataset, "category": category, "method": ours.get("method", "")}
        for metric in METRICS:
            ours_value = to_float(ours, metric)
            baseline = best.get(metric, {})
            baseline_value = to_float(baseline, metric) if baseline else float("nan")
            out[f"ours_{metric}"] = ours_value
            out[f"best_baseline_{metric}"] = baseline_value
            out[f"best_baseline_{metric}_method"] = baseline.get("method", "")
            out[f"delta_{metric}"] = ours_value - baseline_value if math.isfinite(ours_value) and math.isfinite(baseline_value) else None
            out[f"win_{metric}"] = bool(math.isfinite(ours_value) and math.isfinite(baseline_value) and ours_value >= baseline_value)
        rows.append(out)
    return rows


def win_summary(delta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in METRICS:
        rows = [row for row in delta_rows if row.get(f"win_{metric}") in {True, False}]
        out[metric] = {
            "wins": sum(1 for row in rows if bool(row.get(f"win_{metric}"))),
            "total": len(rows),
            "mean_delta": mean(delta_rows, f"delta_{metric}"),
        }
    return out


def selected_run_path(heldout_dir: Path, selection: dict[str, Any]) -> Path:
    category = str(selection.get("category", ""))
    candidate = str(selection.get("selected_candidate", ""))
    materialized = heldout_dir / "selected_runs" / f"{category}_{candidate}_heldout" / "predictions.npz"
    if materialized.exists():
        return materialized
    return Path(str(selection.get("selected_run", ""))) / "predictions.npz"


def aligned_indices(pred: np.lib.npyio.NpzFile, selected_paths: set[str]) -> np.ndarray:
    path_to_idx = {norm_path(path): idx for idx, path in enumerate(pred["paths"])}
    idx = [path_to_idx[path] for path in selected_paths if path in path_to_idx]
    return np.asarray(sorted(idx), dtype=np.int64)


def main() -> None:
    args = parse_args()
    heldout_map = parse_mapping(args.heldout)
    baseline_map = parse_mapping(args.baseline_table)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out)

    ours_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for dataset, heldout_dir in heldout_map.items():
        selections = read_csv(heldout_dir / "selection.csv")
        baselines = read_csv(baseline_map.get(dataset, Path()))
        baseline_by_category: dict[str, list[dict[str, Any]]] = {}
        for baseline in baselines:
            baseline_by_category.setdefault(str(baseline.get("category", "")), []).append(baseline)

        for selection in selections:
            category = str(selection.get("category", ""))
            selected_path = selected_run_path(heldout_dir, selection)
            if not selected_path.exists():
                status_rows.append({"dataset": dataset, "category": category, "status": "missing_selected", "path": str(selected_path)})
                continue
            selected_pred = np.load(selected_path)
            selected_paths = {norm_path(path) for path in selected_pred["paths"]}
            ours_metrics = metrics_from_npz(selected_pred)
            ours_rows.append(
                row_with_metrics(
                    {
                        "dataset": dataset,
                        "source": "ours_selected",
                        "method": "lite_seer_ad_selected_heldout",
                        "category": category,
                        "run": str(selected_path.parent),
                        "selected_candidate": selection.get("selected_candidate", ""),
                        "sample_count": len(selected_paths),
                    },
                    ours_metrics,
                )
            )
            for baseline in baseline_by_category.get(category, []):
                run = str(baseline.get("run", ""))
                method = str(baseline.get("method", "") or baseline.get("ablation", "") or "baseline")
                pred_path = runs_root / run / "predictions.npz"
                if not pred_path.exists():
                    status_rows.append({"dataset": dataset, "category": category, "status": "missing_baseline", "path": str(pred_path), "method": method})
                    continue
                pred = np.load(pred_path)
                idx = aligned_indices(pred, selected_paths)
                if len(idx) != len(selected_paths):
                    status_rows.append(
                        {
                            "dataset": dataset,
                            "category": category,
                            "status": "partial_overlap",
                            "path": str(pred_path),
                            "method": method,
                            "overlap": len(idx),
                            "expected": len(selected_paths),
                        }
                    )
                if len(idx) == 0:
                    continue
                baseline_rows.append(
                    row_with_metrics(
                        {
                            "dataset": dataset,
                            "source": "baseline",
                            "method": method,
                            "category": category,
                            "run": run,
                            "selected_candidate": "",
                            "sample_count": len(idx),
                        },
                        metrics_from_npz(pred, idx),
                    )
                )

    all_rows = [*ours_rows, *baseline_rows]
    main_fields = ["dataset", "source", "method", "category", "run", "selected_candidate", "sample_count", *METRICS, "f1", "iou", "threshold"]
    write_csv(out_dir / "table_heldout_sota_cross_dataset.csv", all_rows, main_fields)
    write_csv(out_dir / "table_mean_by_dataset_method.csv", method_means(all_rows), ["dataset", "source", "method", "categories", *METRICS])
    delta_rows = category_deltas(ours_rows, baseline_rows)
    delta_fields = ["dataset", "category", "method"]
    for metric in METRICS:
        delta_fields.extend([f"ours_{metric}", f"best_baseline_{metric}", f"best_baseline_{metric}_method", f"delta_{metric}", f"win_{metric}"])
    write_csv(out_dir / "table_category_deltas.csv", delta_rows, delta_fields)
    status_fields = sorted({key for row in status_rows for key in row.keys()}) or ["dataset", "status"]
    write_csv(out_dir / "table_alignment_status.csv", status_rows, status_fields)
    summary = {
        "wins": win_summary(delta_rows),
        "means": method_means(all_rows),
        "status_rows": len(status_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), **summary}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
