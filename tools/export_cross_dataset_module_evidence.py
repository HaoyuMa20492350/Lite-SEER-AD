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

from seer_ad_v2.evaluation.module_evidence import (
    comparison_rows,
    summarize_comparisons,
)


DATASET_TABLES = {
    "mvtec15": Path("tables/feature_mvtec15"),
    "visa": Path("tables/feature_visa"),
    "mpdd": Path("tables/feature_mpdd"),
}
COMPARISONS = [
    ("HN-SEV", "hn_sev_vs_feature_only", "feature_hn_sev", "feature_only"),
    ("CRV", "tuned_crv_vs_hn_sev", "feature_tuned_crv", "feature_hn_sev"),
    ("LC-RDS", "utility_vs_fixed10", "utility_lc_rds", "feature_fixed10"),
    ("LC-RDS", "utility_vs_fixed25", "utility_lc_rds", "feature_fixed25"),
    ("LC-RDS", "utility_vs_rule", "utility_lc_rds", "feature_rule_brds"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export 33-category HN-SEV, CRV, and LC-RDS evidence."
    )
    parser.add_argument(
        "--out",
        default="tables/feature_first_fusion_aggregate_paper_package",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    category_rows: list[dict[str, Any]] = []
    for dataset, root in DATASET_TABLES.items():
        metric_rows = []
        for filename in (
            "table_ablation_hn_sev.csv",
            "table_ablation_crv.csv",
            "table_ablation_lc_rds.csv",
        ):
            metric_rows.extend(read_csv(root / filename))
        efficiency_rows = read_csv(root / "table_efficiency_mvtec5.csv")
        for module, comparison, target, baseline in COMPARISONS:
            category_rows.extend(
                comparison_rows(
                    dataset,
                    metric_rows,
                    efficiency_rows,
                    module=module,
                    comparison=comparison,
                    target=target,
                    baseline=baseline,
                )
            )

    summary_rows = summarize_comparisons(category_rows)
    write_csv(out_dir / "table_module_ablation_by_category.csv", category_rows)
    write_csv(out_dir / "table_module_ablation_cross_dataset.csv", summary_rows)

    overall = {
        (row["module"], row["comparison"]): row
        for row in summary_rows
        if row["dataset"] == "all"
    }
    hn = overall[("HN-SEV", "hn_sev_vs_feature_only")]
    fixed10 = overall[("LC-RDS", "utility_vs_fixed10")]
    fixed25 = overall[("LC-RDS", "utility_vs_fixed25")]
    rule = overall[("LC-RDS", "utility_vs_rule")]
    crv = overall[("CRV", "tuned_crv_vs_hn_sev")]
    expected_categories = 33
    coverage_complete = all(
        row["categories"] == expected_categories
        for row in (hn, crv, fixed10, fixed25, rule)
    )
    hn_positive = (
        hn["negative_delta_fprr"] == expected_categories
        and hn["mean_delta_fprr"] < 0
    )
    lc_high_budget_positive = all(
        row["negative_delta_latency_ms_mean"] == expected_categories
        and row["mean_delta_latency_ms_mean"] < 0
        for row in (fixed25, rule)
    )
    summary = {
        "datasets": list(DATASET_TABLES),
        "categories": expected_categories,
        "comparisons": len(COMPARISONS),
        "coverage_complete": coverage_complete,
        "hn_sev": {
            "mean_delta_fprr": hn["mean_delta_fprr"],
            "categories_with_lower_fprr": hn["negative_delta_fprr"],
            "positive_repeatable_evidence": hn_positive,
        },
        "crv": {
            "mean_delta_sdr": crv["mean_delta_sdr_mean"],
            "categories_with_positive_sdr": crv["positive_delta_sdr_mean"],
            "claim": (
                "Visualization only. Cross-dataset SDR-GT alignment is "
                "evaluated separately."
            ),
        },
        "lc_rds": {
            "fixed10": {
                "mean_delta_latency_ms": fixed10["mean_delta_latency_ms_mean"],
                "categories_faster": fixed10[
                    "negative_delta_latency_ms_mean"
                ],
            },
            "fixed25": {
                "mean_delta_latency_ms": fixed25["mean_delta_latency_ms_mean"],
                "categories_faster": fixed25[
                    "negative_delta_latency_ms_mean"
                ],
            },
            "rule": {
                "mean_delta_latency_ms": rule["mean_delta_latency_ms_mean"],
                "categories_faster": rule["negative_delta_latency_ms_mean"],
            },
            "positive_repeatable_high_budget_evidence": lc_high_budget_positive,
            "claim": (
                "LC-RDS is consistently faster than fixed25 and rule-based "
                "repair, but not every fixed10 run."
            ),
        },
    }
    (out_dir / "module_evidence_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(category_rows)} category-comparison rows and "
        f"{len(summary_rows)} summaries to {out_dir}"
    )


if __name__ == "__main__":
    main()
