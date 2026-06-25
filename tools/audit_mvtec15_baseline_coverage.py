from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.registry import REQUIRED_MVTEC15_BASELINES
from tools.run_mvtec15_baselines import MVTEC15_CATEGORIES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit required MVTec15 baseline coverage in exported baseline tables.")
    p.add_argument("--tables", default="tables/mvtec15_baselines")
    p.add_argument("--categories", default=",".join(MVTEC15_CATEGORIES))
    p.add_argument("--required-methods", default=",".join(REQUIRED_MVTEC15_BASELINES))
    p.add_argument("--out", default=None)
    p.add_argument("--fail-on-missing", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables)
    out_dir = Path(args.out) if args.out else tables_dir
    methods = split_csv(args.required_methods)
    categories = split_csv(args.categories)
    rows = _read_csv(tables_dir / "table_main_mvtec15.csv")
    present: dict[tuple[str, str], str] = {}
    for row in rows:
        method = str(row.get("method", "")).strip()
        category = str(row.get("category", "")).strip()
        if method and category:
            present[(method, category)] = str(row.get("run", ""))

    coverage_rows = []
    by_method = {}
    for method in methods:
        method_present = 0
        for category in categories:
            key = (method, category)
            found = key in present
            method_present += int(found)
            coverage_rows.append({"method": method, "category": category, "present": found, "run": present.get(key, "")})
        by_method[method] = {"present": method_present, "total": len(categories), "missing": len(categories) - method_present}

    missing = [row for row in coverage_rows if not bool(row["present"])]
    summary = {
        "tables": str(tables_dir),
        "required_methods": methods,
        "categories": categories,
        "total_required": len(methods) * len(categories),
        "present": len(coverage_rows) - len(missing),
        "missing": len(missing),
        "complete": not missing,
        "by_method": by_method,
    }
    _write_csv(out_dir / "baseline_coverage.csv", coverage_rows, ["method", "category", "present", "run"])
    (out_dir / "baseline_coverage.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.fail_on_missing and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
