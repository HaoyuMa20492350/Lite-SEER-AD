"""Export the strict fixed-threshold policy bundle used by the main paper."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_threshold_bundle(csv_path: Path) -> dict[str, object]:
    rows = _read_rows(csv_path)
    required = {
        "dataset",
        "category",
        "split_seed",
        "threshold",
        "threshold_protocol",
        "selected_candidate",
        "normal_pixel_fpr",
        "heldout_run",
        "uses_real_anomaly_labels_for_threshold",
        "uses_real_anomaly_masks_for_threshold",
    }
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["category"])].append(row)

    policies = []
    for (dataset, category), group in sorted(grouped.items()):
        thresholds = {float(row["threshold"]) for row in group}
        candidates = {row["split_seed"]: row["selected_candidate"] for row in group}
        per_seed = [
            {
                "split_seed": row["split_seed"],
                "threshold": float(row["threshold"]),
                "selected_candidate": row["selected_candidate"],
                "normal_pixel_fpr": float(row["normal_pixel_fpr"]),
                "heldout_run": row["heldout_run"],
            }
            for row in sorted(group, key=lambda item: item["split_seed"])
        ]
        policies.append(
            {
                "dataset": dataset,
                "category": category,
                "threshold_protocol": group[0]["threshold_protocol"],
                "threshold": next(iter(thresholds)) if len(thresholds) == 1 else None,
                "thresholds_are_seed_consistent": len(thresholds) == 1,
                "max_normal_pixel_fpr": max(float(row["normal_pixel_fpr"]) for row in group),
                "selected_candidate_by_seed": candidates,
                "uses_real_anomaly_labels": any(
                    _as_bool(row["uses_real_anomaly_labels_for_threshold"]) for row in group
                ),
                "uses_real_anomaly_masks": any(
                    _as_bool(row["uses_real_anomaly_masks_for_threshold"]) for row in group
                ),
                "per_seed": per_seed,
            }
        )

    return {
        "schema": "synthetic_normal_fixed_threshold_v1_bundle",
        "source_csv": csv_path.as_posix(),
        "policy_count": len(policies),
        "evaluated_run_count": len(rows),
        "uses_real_anomaly_labels": any(policy["uses_real_anomaly_labels"] for policy in policies),
        "uses_real_anomaly_masks": any(policy["uses_real_anomaly_masks"] for policy in policies),
        "policies": policies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("tables/strict_fixed_threshold/strict_selected_metrics.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json"),
    )
    args = parser.parse_args()

    bundle = build_threshold_bundle(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {bundle['policy_count']} policies to {args.out}")


if __name__ == "__main__":
    main()
