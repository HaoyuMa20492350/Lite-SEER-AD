"""Run missing HN-SEV exact input-ablation commands in resumable batches."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMAND_TABLE = (
    REPO_ROOT
    / "tables"
    / "hn_sev_input_ablation"
    / "table_missing_exact_ablation_commands.csv"
)
DEFAULT_REPORT_DIR = REPO_ROOT / "tables" / "hn_sev_input_ablation"
DEFAULT_SUMMARY = REPO_ROOT / "tables" / "hn_sev_input_ablation" / "summary.json"


def split_filter(value: str | None) -> set[str] | None:
    if value is None or not value.strip():
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selected_rows(
    rows: list[dict[str, str]],
    *,
    datasets: set[str] | None = None,
    categories: set[str] | None = None,
    requirements: set[str] | None = None,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if datasets is not None and row.get("dataset") not in datasets:
            continue
        if categories is not None and row.get("category") not in categories:
            continue
        if requirements is not None and row.get("requirement") not in requirements:
            continue
        selected.append(row)
    return selected


def output_ready(row: dict[str, str], root: Path) -> bool:
    train_run_name = row.get("train_run_name") or ""
    eval_run_name = row.get("eval_run_name") or ""
    if not train_run_name or not eval_run_name:
        return False
    train_dir = root / "runs" / train_run_name
    eval_dir = root / "runs" / eval_run_name
    return (
        (train_dir / "hn_sev.pt").exists()
        and (eval_dir / "predictions.npz").exists()
        and (eval_dir / "eval_metrics.json").exists()
    )


def run_shell_command(command: str, *, dry_run: bool, cwd: Path) -> tuple[int | None, float]:
    start = time.perf_counter()
    if dry_run:
        return None, 0.0
    completed = subprocess.run(command, cwd=cwd, shell=True, check=False)
    return int(completed.returncode), time.perf_counter() - start


def run_rows(
    rows: list[dict[str, str]],
    *,
    root: Path = REPO_ROOT,
    dry_run: bool = False,
    resume: bool = True,
    max_runs: int | None = None,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    executed = 0
    for row in rows:
        if max_runs is not None and executed >= max_runs:
            break
        if resume and output_ready(row, root):
            report.append(
                {
                    "dataset": row.get("dataset"),
                    "category": row.get("category"),
                    "requirement": row.get("requirement"),
                    "target_ablation": row.get("target_ablation"),
                    "train_run_name": row.get("train_run_name"),
                    "eval_run_name": row.get("eval_run_name"),
                    "status": "skipped_existing",
                    "train_returncode": None,
                    "infer_returncode": None,
                    "evaluate_returncode": None,
                    "elapsed_seconds": 0.0,
                }
            )
            continue
        executed += 1
        train_code, train_seconds = run_shell_command(
            row.get("train_command", ""),
            dry_run=dry_run,
            cwd=root,
        )
        infer_code: int | None = None
        infer_seconds = 0.0
        eval_code: int | None = None
        eval_seconds = 0.0
        status = "dry_run" if dry_run else "passed"
        if train_code not in {0, None}:
            status = "failed_train"
        else:
            infer_code, infer_seconds = run_shell_command(
                row.get("infer_command", ""),
                dry_run=dry_run,
                cwd=root,
            )
            if infer_code not in {0, None}:
                status = "failed_infer"
            else:
                eval_code, eval_seconds = run_shell_command(
                    row.get("evaluate_command", ""),
                    dry_run=dry_run,
                    cwd=root,
                )
                if eval_code not in {0, None}:
                    status = "failed_evaluate"
        item = {
            "dataset": row.get("dataset"),
            "category": row.get("category"),
            "requirement": row.get("requirement"),
            "target_ablation": row.get("target_ablation"),
            "train_run_name": row.get("train_run_name"),
            "eval_run_name": row.get("eval_run_name"),
            "status": status,
            "train_returncode": train_code,
            "infer_returncode": infer_code,
            "evaluate_returncode": eval_code,
            "elapsed_seconds": train_seconds + infer_seconds + eval_seconds,
        }
        report.append(item)
        if fail_fast and status in {"failed_train", "failed_infer", "failed_evaluate"}:
            break
    return report


def summarize(
    report: list[dict[str, Any]],
    selected_count: int,
    *,
    audit_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in report:
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
    audit_summary = audit_summary or {}
    return {
        "schema": "lite-seer-ad-hn-sev-exact-ablation-runner-v1",
        "selected_commands": selected_count,
        "reported_commands": len(report),
        "counts": counts,
        "audit_categories": audit_summary.get("categories"),
        "complete_exact_categories": audit_summary.get("complete_exact_categories"),
        "complete_metric_categories": audit_summary.get("complete_metric_categories"),
        "ready_for_audit_refresh": bool(report)
        and all(
            row.get("status") in {"passed", "skipped_existing", "dry_run"}
            for row in report
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report_dir: Path, report: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "table_hn_sev_exact_ablation_runner_report.csv", report)
    (report_dir / "hn_sev_exact_ablation_runner_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-table", type=Path, default=DEFAULT_COMMAND_TABLE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--audit-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset filter.")
    parser.add_argument("--categories", default=None, help="Comma-separated category filter.")
    parser.add_argument("--requirements", default=None, help="Comma-separated requirement filter.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.command_table)
    selected = selected_rows(
        rows,
        datasets=split_filter(args.datasets),
        categories=split_filter(args.categories),
        requirements=split_filter(args.requirements),
    )
    report = run_rows(
        selected,
        root=REPO_ROOT,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        max_runs=args.max_runs,
        fail_fast=args.fail_fast,
    )
    summary = summarize(
        report,
        len(selected),
        audit_summary=read_json(args.audit_summary),
    )
    write_report(args.report_dir, report, summary)
    print(
        "HN-SEV exact ablation runner: "
        f"selected={len(selected)} reported={len(report)} counts={summary['counts']}"
    )
    if any(
        row.get("status") in {"failed_train", "failed_infer", "failed_evaluate"}
        for row in report
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
