from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a combined held-out pixel-policy selection summary.")
    p.add_argument("--input", action="append", required=True, help="dataset=heldout_dir")
    p.add_argument("--baseline-candidate", default="pixelraw")
    p.add_argument("--out", required=True)
    return p.parse_args()


def parse_inputs(values: list[str]) -> list[tuple[str, Path]]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Input must be dataset=path: {value}")
        dataset, path = value.split("=", 1)
        out.append((dataset.strip(), Path(path.strip())))
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


def to_float(row: dict[str, Any], key: str) -> float:
    try:
        value = row.get(key)
        if value in {"", None}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [to_float(row, key) for row in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def means(dataset: str, rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    out: dict[str, Any] = {"dataset": dataset, "method": method, "categories": len({row.get("category", "") for row in rows})}
    for metric in METRICS:
        out[metric] = mean(rows, metric)
    return out


def deltas(selected: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_by_cat = {row.get("category", ""): row for row in baseline}
    out = []
    for row in selected:
        category = str(row.get("category", ""))
        base = base_by_cat.get(category)
        if not base:
            continue
        item: dict[str, Any] = {
            "dataset": row.get("dataset", ""),
            "category": category,
            "selected_candidate": row.get("candidate", ""),
        }
        for metric in METRICS:
            selected_value = to_float(row, metric)
            base_value = to_float(base, metric)
            item[f"selected_{metric}"] = selected_value
            item[f"baseline_{metric}"] = base_value
            item[f"delta_{metric}"] = selected_value - base_value if np.isfinite(selected_value) and np.isfinite(base_value) else None
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    all_candidate_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    all_delta_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for dataset, path in parse_inputs(args.input):
        candidate_rows = read_csv(path / "candidate_split_metrics.csv")
        selected_rows = read_csv(path / "selected_heldout_metrics.csv")
        selection_rows = read_csv(path / "selection.csv")
        baseline_rows = [
            row
            for row in candidate_rows
            if row.get("candidate") == args.baseline_candidate and row.get("split") == "heldout_test"
        ]
        for row in candidate_rows:
            row.setdefault("dataset", dataset)
        for row in selected_rows:
            row.setdefault("dataset", dataset)
        all_candidate_rows.extend(candidate_rows)
        all_selected_rows.extend(selected_rows)
        all_delta_rows.extend(deltas(selected_rows, baseline_rows))
        mean_rows.append(means(dataset, baseline_rows, f"{args.baseline_candidate}_heldout"))
        mean_rows.append(means(dataset, selected_rows, "selected_heldout"))
        status_rows.append(
            {
                "dataset": dataset,
                "path": str(path),
                "candidate_rows": len(candidate_rows),
                "selected_rows": len(selected_rows),
                "selection_rows": len(selection_rows),
                "selected_counts": json.dumps(
                    {
                        name: sum(1 for row in selection_rows if row.get("selected_candidate") == name)
                        for name in sorted({row.get("selected_candidate", "") for row in selection_rows})
                    },
                    sort_keys=True,
                ),
            }
        )

    candidate_fields = sorted({key for row in all_candidate_rows for key in row.keys()}) or ["dataset"]
    selected_fields = sorted({key for row in all_selected_rows for key in row.keys()}) or ["dataset"]
    delta_fields = ["dataset", "category", "selected_candidate"]
    for metric in METRICS:
        delta_fields.extend([f"selected_{metric}", f"baseline_{metric}", f"delta_{metric}"])
    write_csv(out_dir / "table_candidate_split_metrics.csv", all_candidate_rows, candidate_fields)
    write_csv(out_dir / "table_selected_heldout_metrics.csv", all_selected_rows, selected_fields)
    write_csv(out_dir / "table_selected_vs_baseline_deltas.csv", all_delta_rows, delta_fields)
    write_csv(out_dir / "table_mean_by_dataset.csv", mean_rows, ["dataset", "method", "categories", *METRICS])
    write_csv(out_dir / "table_status.csv", status_rows, ["dataset", "path", "candidate_rows", "selected_rows", "selection_rows", "selected_counts"])
    summary = {
        "status": status_rows,
        "means": mean_rows,
        "delta_means": {
            metric: mean(all_delta_rows, f"delta_{metric}") for metric in METRICS
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), **summary}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
