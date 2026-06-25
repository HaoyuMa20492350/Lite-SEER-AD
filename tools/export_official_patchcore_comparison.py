from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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
            "Compare strict Lite-SEER-AD MVTec results with the pinned "
            "PatchCore-Official artifacts."
        )
    )
    parser.add_argument(
        "--lite-table",
        default="tables/strict_fixed_threshold_paper/table_strict_mean_by_category.csv",
    )
    parser.add_argument(
        "--official-root",
        default="baselines/external_outputs/mvtec15/patchcore",
    )
    parser.add_argument(
        "--out",
        default="tables/official_patchcore_comparison",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    lite_rows = {
        row["category"]: row
        for row in _read_csv(Path(args.lite_table))
        if row["dataset"] == "mvtec15"
    }
    official_root = Path(args.official_root)
    official_rows = {}
    for path in sorted(official_root.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        official_rows[str(payload["category"])] = payload
    missing = sorted(set(lite_rows) ^ set(official_rows))
    if missing:
        raise ValueError(
            "Lite and official PatchCore categories do not align: "
            + ", ".join(missing)
        )
    rows = []
    for category in sorted(lite_rows):
        row: dict[str, Any] = {"category": category}
        for metric in METRICS:
            lite_value = float(lite_rows[category][metric])
            official_value = float(official_rows[category][metric])
            row[f"lite_{metric}"] = lite_value
            row[f"patchcore_official_{metric}"] = official_value
            row[f"delta_{metric}"] = lite_value - official_value
        rows.append(row)
    summary: dict[str, Any] = {
        "categories": len(rows),
        "patchcore_source_commit": next(iter(official_rows.values()))[
            "source_commit"
        ],
        "threshold_protocol": "synthetic_normal_fixed_threshold_v1",
        "uses_real_anomaly_labels_for_threshold": False,
        "uses_real_anomaly_masks_for_threshold": False,
        "metrics": {},
    }
    for metric in METRICS:
        lite_values = [float(row[f"lite_{metric}"]) for row in rows]
        official_values = [
            float(row[f"patchcore_official_{metric}"]) for row in rows
        ]
        deltas = [float(row[f"delta_{metric}"]) for row in rows]
        summary["metrics"][metric] = {
            "lite_mean": sum(lite_values) / len(lite_values),
            "patchcore_official_mean": sum(official_values)
            / len(official_values),
            "delta_lite_minus_patchcore": sum(deltas) / len(deltas),
            "lite_category_wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "patchcore_category_wins": sum(delta < 0 for delta in deltas),
        }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "table_lite_vs_patchcore_official.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
