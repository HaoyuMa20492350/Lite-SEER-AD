from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "mvtec": Path("tables/fulltest_mvtec15"),
    "visa": Path("tables/fulltest_visa"),
    "mpdd": Path("tables/fulltest_mpdd"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine verified full-test results into paper-ready tables."
    )
    parser.add_argument(
        "--coverage-audit",
        default="tables/full_test_coverage_audit.json",
    )
    parser.add_argument(
        "--out",
        default="tables/fulltest_paper_package",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["dataset"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tagged(dataset: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"dataset": dataset, **row} for row in rows]


def main() -> None:
    args = parse_args()
    audit_path = REPO_ROOT / args.coverage_audit
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("complete"):
        raise RuntimeError(
            f"Full-test coverage is incomplete; refusing to export paper tables: {audit_path}"
        )

    main_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for dataset, source in SOURCES.items():
        root = REPO_ROOT / source
        main_rows.extend(tagged(dataset, read_csv(root / "table_main_mvtec5.csv")))
        efficiency_rows.extend(
            tagged(dataset, read_csv(root / "table_efficiency_mvtec5.csv"))
        )
        evidence_rows.extend(tagged(dataset, read_csv(root / "evidence_summary.csv")))
        seen: set[str] = set()
        for name in (
            "table_ablation_hn_sev.csv",
            "table_ablation_crv.csv",
            "table_ablation_lc_rds.csv",
        ):
            for row in read_csv(root / name):
                key = str(row.get("run", ""))
                if key in seen:
                    continue
                seen.add(key)
                ablation_rows.append({"dataset": dataset, **row})

    out = REPO_ROOT / args.out
    write_csv(out / "table_main_cross_dataset.csv", main_rows)
    write_csv(out / "table_ablation_cross_dataset.csv", ablation_rows)
    write_csv(out / "table_efficiency_cross_dataset.csv", efficiency_rows)
    write_csv(out / "table_module_evidence_cross_dataset.csv", evidence_rows)
    (out / "coverage_summary.json").write_text(
        json.dumps(
            {
                "categories": audit["total_categories"],
                "test_images": audit["total_test_images"],
                "settings": len(audit["ablations"]),
                "image_records": audit["total_expected_image_records"],
                "coverage_audit": str(Path(args.coverage_audit)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote verified full-test paper tables to {out}")


if __name__ == "__main__":
    main()
