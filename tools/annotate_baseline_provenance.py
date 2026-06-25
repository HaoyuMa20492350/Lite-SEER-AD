from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.registry import BASELINES


PROVENANCE_FIELDS = [
    "method_id",
    "display_method",
    "implementation_variant",
    "official_implementation",
    "source_path",
    "source_url",
    "source_commit",
    "reference_key",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate existing baseline runs and tables with provenance."
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument(
        "--tables-root",
        action="append",
        default=[],
        help="Baseline table directory. May be repeated.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _method(run_dir: Path, metrics: dict[str, Any]) -> str:
    args = _read_json(run_dir / "run_args.json")
    payload = args.get("args", {}) if isinstance(args, dict) else {}
    return str(
        payload.get("method", metrics.get("method_id", metrics.get("method", "")))
    )


def _provenance(method: str, commit: str) -> dict[str, Any]:
    spec = BASELINES[method]
    return {
        "method_id": method,
        "display_method": spec.display_name,
        "implementation_variant": spec.implementation_variant,
        "official_implementation": spec.official_implementation,
        "source_path": spec.source_path,
        "source_url": "",
        "source_commit": commit if spec.local_runner else "",
        "reference_key": spec.reference_key,
    }


def _write_metric_csv(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


def _annotate_runs(root: Path, commit: str) -> int:
    count = 0
    for metrics_path in root.glob("*/metrics.json"):
        run_dir = metrics_path.parent
        metrics = _read_json(metrics_path)
        method = _method(run_dir, metrics)
        if method not in BASELINES:
            continue
        metrics.update(_provenance(method, commit))
        metrics["method"] = method
        metrics_path.write_text(
            json.dumps(metrics, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        _write_metric_csv(run_dir / "metrics.csv", metrics)
        count += 1
    return count


def _annotate_table(path: Path, commit: str) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        existing_fields = list(reader.fieldnames or [])
    changed = 0
    for row in rows:
        method = str(row.get("method_id") or row.get("method") or "")
        if method not in BASELINES:
            continue
        for key, value in _provenance(method, commit).items():
            if not row.get(key):
                row[key] = value
        changed += 1
    fields = existing_fields + [
        field for field in PROVENANCE_FIELDS if field not in existing_fields
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return changed


def main() -> None:
    args = parse_args()
    commit = _commit()
    run_count = _annotate_runs(Path(args.runs_root), commit)
    table_roots = args.tables_root or [
        "tables/mvtec15_baselines",
        "tables/visa_baselines",
        "tables/mpdd_baselines",
    ]
    table_count = 0
    row_count = 0
    for root_value in table_roots:
        for path in Path(root_value).glob("*.csv"):
            if "table_" not in path.name:
                continue
            row_count += _annotate_table(path, commit)
            table_count += 1
    print(
        json.dumps(
            {
                "annotated_runs": run_count,
                "annotated_tables": table_count,
                "annotated_table_rows": row_count,
                "source_commit": commit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
