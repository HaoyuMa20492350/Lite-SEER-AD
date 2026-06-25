from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]
DATASET_INPUTS = {
    "mvtec15": {
        "ours": Path("tables/feature_pixelraw_mvtec15/table_main_mvtec5.csv"),
        "baseline": Path("tables/mvtec15_baselines/table_main_mvtec15.csv"),
        "ours_filter": "feature_tuned_crv",
    },
    "visa": {
        "ours": Path("tables/feature_pixelraw_visa/table_main_visa.csv"),
        "baseline": Path("tables/visa_baselines/table_main_visa.csv"),
        "ours_filter": "feature_tuned_crv",
    },
    "mpdd": {
        "ours": Path("tables/feature_pixelraw_mpdd/table_main_mpdd.csv"),
        "baseline": Path("tables/mpdd_baselines/table_main_mpdd.csv"),
        "ours_filter": "feature_tuned_crv",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a feature-first cross-dataset paper package.")
    p.add_argument("--out", default="tables/feature_paper_package")
    return p.parse_args()


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


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = finite([to_float(row, key) for row in rows])
    if not vals:
        return None
    return sum(vals) / len(vals)


def normalize_ours(dataset: str, rows: list[dict[str, Any]], ablation: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if str(row.get("ablation", "")) != ablation:
            continue
        item = dict(row)
        item["dataset"] = dataset
        item["source"] = "ours_feature"
        item["method"] = "lite_seer_ad_feature"
        out.append(item)
    return out


def normalize_baselines(dataset: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["dataset"] = dataset
        item["source"] = "baseline"
        item["method"] = str(item.get("method", "") or item.get("ablation", "") or "baseline")
        out.append(item)
    return out


def method_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("source", "")), str(row.get("method", ""))), []).append(row)
    out = []
    for (dataset, source, method), items in sorted(grouped.items()):
        row = {"dataset": dataset, "source": source, "method": method, "categories": len({item.get("category", "") for item in items})}
        for metric in METRICS:
            row[metric] = mean(items, metric)
        out.append(row)
    return out


def best_baselines(baselines: list[dict[str, Any]], dataset: str, category: str) -> dict[str, dict[str, Any]]:
    candidates = [row for row in baselines if row.get("dataset") == dataset and row.get("category") == category]
    out = {}
    for metric in METRICS:
        vals = [row for row in candidates if math.isfinite(to_float(row, metric))]
        if vals:
            out[metric] = max(vals, key=lambda row: to_float(row, metric))
    return out


def category_deltas(ours: list[dict[str, Any]], baselines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in sorted(ours, key=lambda item: (str(item.get("dataset", "")), str(item.get("category", "")))):
        dataset = str(row.get("dataset", ""))
        category = str(row.get("category", ""))
        best = best_baselines(baselines, dataset, category)
        out = {"dataset": dataset, "category": category, "method": "lite_seer_ad_feature"}
        for metric in METRICS:
            ours_value = to_float(row, metric)
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
        vals = [row for row in delta_rows if row.get(f"win_{metric}") in {True, False}]
        out[metric] = {
            "wins": sum(1 for row in vals if bool(row.get(f"win_{metric}"))),
            "total": len(vals),
            "mean_delta": mean(delta_rows, f"delta_{metric}"),
        }
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    all_rows: list[dict[str, Any]] = []
    ours_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for dataset, spec in DATASET_INPUTS.items():
        ours = normalize_ours(dataset, read_csv(spec["ours"]), str(spec["ours_filter"]))
        baselines = normalize_baselines(dataset, read_csv(spec["baseline"]))
        ours_rows.extend(ours)
        baseline_rows.extend(baselines)
        all_rows.extend(ours)
        all_rows.extend(baselines)
        status_rows.append(
            {
                "dataset": dataset,
                "ours_path": str(spec["ours"]),
                "baseline_path": str(spec["baseline"]),
                "ours_categories": len({row.get("category", "") for row in ours}),
                "baseline_categories": len({row.get("category", "") for row in baselines}),
                "baseline_methods": len({row.get("method", "") for row in baselines}),
                "ours_rows": len(ours),
                "baseline_rows": len(baselines),
                "complete": bool(ours and baselines),
            }
        )

    main_fields = [
        "dataset",
        "source",
        "method",
        "run",
        "category",
        "ablation",
        *METRICS,
        "fprr",
        "image_score_mode",
        "image_score_source",
        "pixel_heatmap_source",
    ]
    write_csv(out_dir / "table_main_cross_dataset.csv", all_rows, main_fields)
    write_csv(out_dir / "table_mean_by_dataset_method.csv", method_means(all_rows), ["dataset", "source", "method", "categories", *METRICS])
    delta_rows = category_deltas(ours_rows, baseline_rows)
    delta_fields = ["dataset", "category", "method"]
    for metric in METRICS:
        delta_fields.extend([f"ours_{metric}", f"best_baseline_{metric}", f"best_baseline_{metric}_method", f"delta_{metric}", f"win_{metric}"])
    write_csv(out_dir / "table_category_deltas.csv", delta_rows, delta_fields)
    write_csv(out_dir / "table_run_status.csv", status_rows, list(status_rows[0].keys()) if status_rows else ["dataset"])
    summary = {
        "inputs": {dataset: {key: str(value) for key, value in spec.items()} for dataset, spec in DATASET_INPUTS.items()},
        "wins": win_summary(delta_rows),
        "ours_means": [row for row in method_means(ours_rows) if row.get("method") == "lite_seer_ad_feature"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "status": status_rows, "wins": summary["wins"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
