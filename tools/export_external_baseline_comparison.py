from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "pixel_ap",
    "dice",
    "oracle_dice",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare strict Lite-SEER-AD MVTec results with all selected "
            "pinned external baseline artifacts."
        )
    )
    parser.add_argument(
        "--lite-table",
        default=(
            "tables/strict_fixed_threshold_paper/"
            "table_strict_mean_by_category.csv"
        ),
    )
    parser.add_argument(
        "--external-root",
        default="baselines/external_outputs/mvtec15",
    )
    parser.add_argument("--methods", default="patchcore,padim")
    parser.add_argument(
        "--out",
        default="tables/external_baseline_comparison",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--worst-categories", type=int, default=5)
    return parser.parse_args()


def split_methods(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    if not methods:
        raise ValueError("At least one external method is required")
    return methods


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_sign_p(wins: int, losses: int) -> float | None:
    observations = wins + losses
    if observations == 0:
        return None
    tail = min(wins, losses)
    probability = sum(
        math.comb(observations, index)
        for index in range(tail + 1)
    ) / (2**observations)
    return min(1.0, 2.0 * probability)


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256(
        ":".join((str(seed), *parts)).encode("utf-8")
    ).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_inference(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    results = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        for metric in METRICS:
            deltas = np.asarray(
                [float(row[f"delta_{metric}"]) for row in method_rows],
                dtype=np.float64,
            )
            rng = stable_rng(seed, method, metric)
            indices = rng.integers(
                0,
                len(deltas),
                size=(samples, len(deltas)),
            )
            means = np.mean(deltas[indices], axis=1)
            wins = int(np.sum(deltas > 0))
            losses = int(np.sum(deltas < 0))
            results.append(
                {
                    "method": method,
                    "display_method": method_rows[0]["display_method"],
                    "metric": metric,
                    "categories": len(deltas),
                    "mean_delta": float(np.mean(deltas)),
                    "median_delta": float(np.median(deltas)),
                    "ci95_low": float(np.quantile(means, 0.025)),
                    "ci95_high": float(np.quantile(means, 0.975)),
                    "wins": wins,
                    "losses": losses,
                    "ties": int(np.sum(deltas == 0)),
                    "sign_test_p": exact_sign_p(wins, losses),
                    "bootstrap_samples": samples,
                    "bootstrap_seed": seed,
                }
            )
    for metric in METRICS:
        metric_rows = [row for row in results if row["metric"] == metric]
        raw = [
            float(row["sign_test_p"])
            if row["sign_test_p"] is not None
            else 1.0
            for row in metric_rows
        ]
        for row, adjusted in zip(metric_rows, holm_adjust(raw)):
            row["sign_test_p_holm_within_metric"] = adjusted
            row["bootstrap_ci_excludes_zero"] = (
                float(row["ci95_low"]) > 0
                or float(row["ci95_high"]) < 0
            )
    return results


def failure_rows(
    rows: list[dict[str, Any]],
    *,
    worst_categories: int,
) -> list[dict[str, Any]]:
    failures = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        for metric in METRICS:
            ordered = sorted(
                method_rows,
                key=lambda row: float(row[f"delta_{metric}"]),
            )
            for rank, row in enumerate(
                ordered[: max(1, worst_categories)],
                start=1,
            ):
                delta = float(row[f"delta_{metric}"])
                failures.append(
                    {
                        "method": method,
                        "display_method": row["display_method"],
                        "metric": metric,
                        "rank": rank,
                        "category": row["category"],
                        "lite_value": float(row[f"lite_{metric}"]),
                        "external_value": float(row[f"external_{metric}"]),
                        "delta_lite_minus_external": delta,
                        "lite_loses": delta < 0,
                    }
                )
    return failures


def supplement_markdown(
    inference: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> str:
    lines = [
        "# External Baseline Paired Analysis",
        "",
        "All tests use the 15 matched MVTec AD categories. Confidence intervals",
        "are percentile paired-bootstrap intervals over categories. Sign tests",
        "are exact and two-sided; Holm correction is applied across the seven",
        "methods separately for each metric.",
        "",
        "| Method | Metric | Mean delta | 95% CI | W/L/T | Sign p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in inference:
        if row["metric"] == "oracle_dice":
            continue
        lines.append(
            "| {display_method} | {metric} | {mean_delta:+.4f} | "
            "[{ci95_low:+.4f}, {ci95_high:+.4f}] | "
            "{wins}/{losses}/{ties} | {sign_test_p:.4g} | "
            "{sign_test_p_holm_within_metric:.4g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Largest Category Losses",
            "",
            "| Method | Metric | Category | Delta |",
            "|---|---|---|---:|",
        ]
    )
    for row in failures:
        if (
            row["rank"] == 1
            and row["metric"] != "oracle_dice"
            and row["lite_loses"]
        ):
            lines.append(
                f"| {row['display_method']} | {row['metric']} | "
                f"{row['category']} | "
                f"{row['delta_lite_minus_external']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "Positive delta favors Lite-SEER-AD. Statistical significance does",
            "not by itself establish practical or universal superiority.",
            "",
        ]
    )
    return "\n".join(lines)


def load_external_method(
    external_root: Path,
    method: str,
) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted((external_root / method).glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance_path = path.with_name("provenance.json")
        if provenance_path.exists():
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            payload = {**provenance, **payload}
        category = str(payload["category"])
        rows[category] = payload
    if not rows:
        raise FileNotFoundError(
            f"No external metrics found for {method}: {external_root / method}"
        )
    return rows


def build_comparison(
    lite_rows: dict[str, dict[str, str]],
    external: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    categories = set(lite_rows)
    for method, method_rows in external.items():
        mismatch = sorted(categories ^ set(method_rows))
        if mismatch:
            raise ValueError(
                f"Lite and {method} categories do not align: "
                + ", ".join(mismatch)
            )

    rows = []
    summary: dict[str, Any] = {
        "categories": len(categories),
        "methods": list(external),
        "threshold_protocol": "synthetic_normal_fixed_threshold_v1",
        "uses_real_anomaly_labels_for_threshold": False,
        "uses_real_anomaly_masks_for_threshold": False,
        "comparisons": {},
    }
    for method, method_rows in external.items():
        first = next(iter(method_rows.values()))
        method_summary: dict[str, Any] = {
            "display_method": first.get("display_method", method),
            "source_kind": first.get("source_kind", ""),
            "official_implementation": bool(
                first.get("official_implementation", False)
            ),
            "source_commit": first.get("source_commit", ""),
            "metrics": {},
        }
        for category in sorted(categories):
            row: dict[str, Any] = {
                "method": method,
                "display_method": method_summary["display_method"],
                "source_kind": method_summary["source_kind"],
                "category": category,
            }
            for metric in METRICS:
                lite_value = float(lite_rows[category][metric])
                external_value = float(method_rows[category][metric])
                row[f"lite_{metric}"] = lite_value
                row[f"external_{metric}"] = external_value
                row[f"delta_{metric}"] = lite_value - external_value
            rows.append(row)
        method_table = [row for row in rows if row["method"] == method]
        for metric in METRICS:
            lite_values = [
                float(row[f"lite_{metric}"]) for row in method_table
            ]
            external_values = [
                float(row[f"external_{metric}"]) for row in method_table
            ]
            deltas = [float(row[f"delta_{metric}"]) for row in method_table]
            method_summary["metrics"][metric] = {
                "lite_mean": sum(lite_values) / len(lite_values),
                "external_mean": sum(external_values)
                / len(external_values),
                "delta_lite_minus_external": sum(deltas) / len(deltas),
                "lite_category_wins": sum(delta > 0 for delta in deltas),
                "ties": sum(delta == 0 for delta in deltas),
                "external_category_wins": sum(delta < 0 for delta in deltas),
            }
        summary["comparisons"][method] = method_summary
    return rows, summary


def main() -> None:
    args = parse_args()
    lite_rows = {
        row["category"]: row
        for row in read_csv(Path(args.lite_table))
        if row["dataset"] == "mvtec15"
    }
    external_root = Path(args.external_root)
    methods = split_methods(args.methods)
    external = {
        method: load_external_method(external_root, method)
        for method in methods
    }
    rows, summary = build_comparison(lite_rows, external)
    inference = paired_inference(
        rows,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    failures = failure_rows(
        rows,
        worst_categories=args.worst_categories,
    )
    summary["paired_inference"] = {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "sign_test": "two_sided_exact",
        "multiplicity": "holm_across_methods_within_metric",
        "records": inference,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_lite_vs_external_baselines.csv", rows)
    write_csv(out_dir / "table_paired_inference.csv", inference)
    write_csv(out_dir / "table_worst_category_losses.csv", failures)
    (out_dir / "paired_analysis.md").write_text(
        supplement_markdown(inference, failures),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
