from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize synthetic-normal policy-gate metrics and selection stability across seeds."
    )
    p.add_argument(
        "--gate",
        action="append",
        required=True,
        help="Gate package as dataset=path/to/gate_root. Repeat for multiple datasets.",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--baseline-candidate", default="pixelraw")
    p.add_argument("--stability-threshold", type=float, default=1.0)
    p.add_argument("--metric-std-threshold", type=float, default=0.003)
    return p.parse_args()


def parse_gates(values: list[str]) -> list[tuple[str, Path]]:
    gates = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Gate must be dataset=path: {value}")
        dataset, path = value.split("=", 1)
        gates.append((dataset.strip(), Path(path.strip())))
    return gates


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(values: list[float]) -> list[float]:
    return [value for value in values if np.isfinite(value)]


def stats(values: list[float]) -> dict[str, float | int | None]:
    values = finite(values)
    return {
        "seeds": len(values),
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values, ddof=0)) if values else None,
        "min": float(np.min(values)) if values else None,
        "max": float(np.max(values)) if values else None,
    }


def seed_number(path: Path) -> int:
    return int(path.name.removeprefix("seed"))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_metric_rows: list[dict[str, Any]] = []
    seed_delta_rows: list[dict[str, Any]] = []
    selection_by_category: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    category_metrics: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for dataset, gate_root in parse_gates(args.gate):
        seed_dirs = sorted(
            (path for path in gate_root.glob("seed*") if path.is_dir()),
            key=seed_number,
        )
        for seed_dir in seed_dirs:
            seed = seed_number(seed_dir)
            summary_path = seed_dir / "summary.json"
            selection_path = seed_dir / "selection.csv"
            metrics_path = seed_dir / "selected_heldout_metrics.csv"
            if not (summary_path.exists() and selection_path.exists() and metrics_path.exists()):
                continue

            summary = read_json(summary_path)
            means = summary.get("means", {})
            selected_rows = read_csv(metrics_path)
            candidate_rows = read_csv(seed_dir / "candidate_split_metrics.csv")
            seed_metric_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "categories": means.get("categories"),
                    **{metric: means.get(metric) for metric in METRICS},
                }
            )
            selected_split = selected_rows[0].get("split", "heldout_test") if selected_rows else "heldout_test"
            baseline_by_category = {
                row["category"]: row
                for row in candidate_rows
                if row.get("candidate") == args.baseline_candidate
                and row.get("split") == selected_split
            }
            delta_row: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "baseline_candidate": args.baseline_candidate,
                "categories": 0,
            }
            matched_categories = 0
            for metric in METRICS:
                metric_deltas = []
                for selected in selected_rows:
                    baseline = baseline_by_category.get(selected["category"])
                    if baseline is None:
                        continue
                    selected_value = as_float(selected.get(metric))
                    baseline_value = as_float(baseline.get(metric))
                    if np.isfinite(selected_value) and np.isfinite(baseline_value):
                        metric_deltas.append(selected_value - baseline_value)
                delta_row[f"delta_{metric}"] = (
                    float(np.mean(metric_deltas)) if metric_deltas else None
                )
                matched_categories = max(matched_categories, len(metric_deltas))
            delta_row["categories"] = matched_categories
            seed_delta_rows.append(delta_row)
            for row in read_csv(selection_path):
                selection_by_category[(dataset, row["category"])].append(
                    (seed, row["selected_candidate"])
                )
            for row in selected_rows:
                category = row["category"]
                for metric in METRICS:
                    category_metrics[(dataset, category, metric)].append(as_float(row.get(metric)))

    metric_summary_rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in seed_metric_rows}):
        dataset_rows = [row for row in seed_metric_rows if row["dataset"] == dataset]
        for metric in METRICS:
            row_stats = stats([as_float(row.get(metric)) for row in dataset_rows])
            metric_summary_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    **row_stats,
                    "std_threshold": args.metric_std_threshold,
                    "passes_std_threshold": (
                        row_stats["std"] is not None
                        and float(row_stats["std"]) <= args.metric_std_threshold
                    ),
                }
            )

    delta_summary_rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in seed_delta_rows}):
        dataset_rows = [row for row in seed_delta_rows if row["dataset"] == dataset]
        for metric in METRICS:
            row_stats = stats(
                [as_float(row.get(f"delta_{metric}")) for row in dataset_rows]
            )
            delta_summary_rows.append(
                {
                    "dataset": dataset,
                    "metric": f"delta_{metric}",
                    **row_stats,
                    "std_threshold": args.metric_std_threshold,
                    "passes_std_threshold": (
                        row_stats["std"] is not None
                        and float(row_stats["std"]) <= args.metric_std_threshold
                    ),
                }
            )

    stability_rows: list[dict[str, Any]] = []
    for (dataset, category), values in sorted(selection_by_category.items()):
        counts = Counter(candidate for _, candidate in values)
        dominant_candidate, dominant_count = counts.most_common(1)[0]
        agreement = dominant_count / len(values)
        stability_rows.append(
            {
                "dataset": dataset,
                "category": category,
                "seeds": len(values),
                "dominant_candidate": dominant_candidate,
                "agreement": agreement,
                "stable": agreement >= args.stability_threshold,
                "selection_by_seed": ";".join(
                    f"{seed}:{candidate}" for seed, candidate in sorted(values)
                ),
            }
        )

    category_metric_rows: list[dict[str, Any]] = []
    for (dataset, category, metric), values in sorted(category_metrics.items()):
        category_metric_rows.append(
            {
                "dataset": dataset,
                "category": category,
                "metric": metric,
                **stats(values),
            }
        )

    write_csv(
        out_dir / "table_seed_metrics.csv",
        seed_metric_rows,
        ["dataset", "seed", "categories", *METRICS],
    )
    write_csv(
        out_dir / "table_metric_mean_std.csv",
        metric_summary_rows,
        [
            "dataset",
            "metric",
            "seeds",
            "mean",
            "std",
            "min",
            "max",
            "std_threshold",
            "passes_std_threshold",
        ],
    )
    write_csv(
        out_dir / "table_seed_delta_metrics.csv",
        seed_delta_rows,
        [
            "dataset",
            "seed",
            "baseline_candidate",
            "categories",
            *[f"delta_{metric}" for metric in METRICS],
        ],
    )
    write_csv(
        out_dir / "table_delta_mean_std.csv",
        delta_summary_rows,
        [
            "dataset",
            "metric",
            "seeds",
            "mean",
            "std",
            "min",
            "max",
            "std_threshold",
            "passes_std_threshold",
        ],
    )
    write_csv(
        out_dir / "table_category_selection_stability.csv",
        stability_rows,
        [
            "dataset",
            "category",
            "seeds",
            "dominant_candidate",
            "agreement",
            "stable",
            "selection_by_seed",
        ],
    )
    write_csv(
        out_dir / "table_category_metric_mean_std.csv",
        category_metric_rows,
        ["dataset", "category", "metric", "seeds", "mean", "std", "min", "max"],
    )

    unstable = [row for row in stability_rows if not row["stable"]]
    failed_std = [
        row
        for row in delta_summary_rows
        if row["metric"] in {"delta_pixel_ap", "delta_dice"}
        and not row["passes_std_threshold"]
    ]
    payload = {
        "datasets": sorted({row["dataset"] for row in seed_metric_rows}),
        "seed_runs": len(seed_metric_rows),
        "category_selection_agreement": (
            float(np.mean([row["agreement"] for row in stability_rows]))
            if stability_rows
            else None
        ),
        "unstable_categories": unstable,
        "metric_mean_std": metric_summary_rows,
        "delta_mean_std": delta_summary_rows,
        "ap_dice_delta_std_acceptance": {
            "threshold": args.metric_std_threshold,
            "passed": not failed_std,
            "failures": failed_std,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
