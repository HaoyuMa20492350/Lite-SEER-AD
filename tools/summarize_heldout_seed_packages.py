from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize held-out pixel-policy packages across split seeds.")
    p.add_argument(
        "--package",
        action="append",
        required=True,
        help="Package as seed=path/to/heldout_package. Repeat for each seed.",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def parse_packages(values: list[str]) -> list[tuple[str, Path]]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Package must be seed=path: {value}")
        seed, path = value.split("=", 1)
        out.append((seed.strip(), Path(path.strip())))
    return out


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    try:
        if value in {"", None}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [as_float(row.get(key)) for row in rows]
    return [value for value in values if np.isfinite(value)]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for seed, package_path in parse_packages(args.package):
        summary = read_json(package_path / "summary.json")
        if not summary:
            continue
        delta_means = summary.get("delta_means", {})
        row: dict[str, Any] = {"seed": seed, "package": str(package_path)}
        for metric in METRICS:
            row[f"delta_{metric}"] = delta_means.get(metric)
        seed_rows.append(row)
        for item in summary.get("means", []):
            mean_row = {"seed": seed, **item}
            mean_rows.append(mean_row)
        for item in summary.get("status", []):
            status_rows.append({"seed": seed, **item})

    aggregate_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        key = f"delta_{metric}"
        values = finite_values(seed_rows, key)
        aggregate_rows.append(
            {
                "metric": key,
                "seeds": len(values),
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=0)) if values else None,
                "min": float(np.min(values)) if values else None,
                "max": float(np.max(values)) if values else None,
            }
        )

    write_csv(out_dir / "table_seed_delta_summary.csv", seed_rows, ["seed", "package", *[f"delta_{m}" for m in METRICS]])
    write_csv(out_dir / "table_seed_mean_by_dataset.csv", mean_rows, ["seed", "dataset", "method", "categories", *METRICS])
    write_csv(out_dir / "table_seed_status.csv", status_rows, ["seed", "dataset", "path", "candidate_rows", "selected_rows", "selection_rows", "selected_counts"])
    write_csv(out_dir / "table_delta_mean_std.csv", aggregate_rows, ["metric", "seeds", "mean", "std", "min", "max"])

    payload = {"seeds": len(seed_rows), "delta_mean_std": aggregate_rows}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
