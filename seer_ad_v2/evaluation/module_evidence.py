from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


DETECTOR_METRICS = (
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "pixel_ap",
    "dice",
)
REPAIR_METRICS = ("fprr", "rdc", "sdr_mean", "pareto_area")


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def comparison_rows(
    dataset: str,
    metric_rows: Iterable[dict[str, Any]],
    efficiency_rows: Iterable[dict[str, Any]],
    *,
    module: str,
    comparison: str,
    target: str,
    baseline: str,
) -> list[dict[str, Any]]:
    metrics = {
        (str(row.get("category", "")), str(row.get("ablation", ""))): row
        for row in metric_rows
    }
    efficiency = {
        (str(row.get("category", "")), str(row.get("ablation", ""))): row
        for row in efficiency_rows
    }
    categories = sorted(
        category
        for category, ablation in metrics
        if ablation == target and (category, baseline) in metrics
    )
    rows = []
    for category in categories:
        target_row = metrics[(category, target)]
        baseline_row = metrics[(category, baseline)]
        row: dict[str, Any] = {
            "dataset": dataset,
            "category": category,
            "module": module,
            "comparison": comparison,
            "target": target,
            "baseline": baseline,
        }
        for metric in (*DETECTOR_METRICS, *REPAIR_METRICS):
            target_value = as_float(target_row.get(metric))
            baseline_value = as_float(baseline_row.get(metric))
            row[f"target_{metric}"] = target_value
            row[f"baseline_{metric}"] = baseline_value
            row[f"delta_{metric}"] = target_value - baseline_value
        target_efficiency = efficiency.get((category, target), {})
        baseline_efficiency = efficiency.get((category, baseline), {})
        for metric in ("latency_ms_mean", "nfe_mean"):
            target_value = as_float(target_efficiency.get(metric))
            baseline_value = as_float(baseline_efficiency.get(metric))
            row[f"target_{metric}"] = target_value
            row[f"baseline_{metric}"] = baseline_value
            row[f"delta_{metric}"] = target_value - baseline_value
        rows.append(row)
    return rows


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else float("nan")


def summarize_comparisons(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    all_rows = list(rows)
    for row in all_rows:
        key = (
            str(row["dataset"]),
            str(row["module"]),
            str(row["comparison"]),
        )
        grouped[key].append(row)
        grouped[("all", key[1], key[2])].append(row)

    summary = []
    delta_metrics = [
        *(f"delta_{metric}" for metric in DETECTOR_METRICS),
        *(f"delta_{metric}" for metric in REPAIR_METRICS),
        "delta_latency_ms_mean",
        "delta_nfe_mean",
    ]
    for (dataset, module, comparison), items in sorted(grouped.items()):
        row: dict[str, Any] = {
            "dataset": dataset,
            "module": module,
            "comparison": comparison,
            "target": items[0]["target"],
            "baseline": items[0]["baseline"],
            "categories": len(items),
        }
        for metric in delta_metrics:
            values = [as_float(item.get(metric)) for item in items]
            row[f"mean_{metric}"] = _finite_mean(values)
            row[f"positive_{metric}"] = sum(
                np.isfinite(value) and value > 0 for value in values
            )
            row[f"negative_{metric}"] = sum(
                np.isfinite(value) and value < 0 for value in values
            )
        summary.append(row)
    return summary
