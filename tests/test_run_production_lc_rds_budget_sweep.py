from __future__ import annotations

import csv
from pathlib import Path

from tools.run_production_lc_rds_budget_sweep import (
    read_rows,
    run_rows,
    selected_rows,
    summarize,
    write_report,
)


def write_command_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "dataset": "mvtec15",
            "category": "bottle",
            "budget_ms": "10",
            "checkpoint_ready": "True",
            "run_name": "production_lc_rds_budget_mvtec15_bottle_budget10_utility_lc_rds",
            "infer_command": "python infer.py --run-name production_lc_rds_budget_mvtec15_bottle_budget10_utility_lc_rds",
            "evaluate_command": "python evaluate.py --pred_dir runs/x --out runs/x/eval_metrics.json",
        },
        {
            "dataset": "mvtec15",
            "category": "bottle",
            "budget_ms": "25",
            "checkpoint_ready": "False",
            "run_name": "production_lc_rds_budget_mvtec15_bottle_budget25_utility_lc_rds",
            "infer_command": "python infer.py --run-name production_lc_rds_budget_mvtec15_bottle_budget25_utility_lc_rds",
            "evaluate_command": "python evaluate.py --pred_dir runs/y --out runs/y/eval_metrics.json",
        },
        {
            "dataset": "visa",
            "category": "candle",
            "budget_ms": "10",
            "checkpoint_ready": "True",
            "run_name": "production_lc_rds_budget_visa_candle_budget10_utility_lc_rds",
            "infer_command": "python infer.py --run-name production_lc_rds_budget_visa_candle_budget10_utility_lc_rds",
            "evaluate_command": "python evaluate.py --pred_dir runs/z --out runs/z/eval_metrics.json",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_selected_rows_filters_and_skips_missing_checkpoints(tmp_path: Path) -> None:
    table = tmp_path / "commands.csv"
    write_command_table(table)
    rows = read_rows(table)

    selected = selected_rows(
        rows,
        datasets={"mvtec15"},
        categories={"bottle"},
        budgets={"10", "25"},
        require_ready=True,
    )

    assert [row["budget_ms"] for row in selected] == ["10"]


def test_run_rows_dry_run_respects_max_runs(tmp_path: Path) -> None:
    table = tmp_path / "commands.csv"
    write_command_table(table)
    rows = selected_rows(read_rows(table), require_ready=True)

    report = run_rows(rows, root=tmp_path, dry_run=True, max_runs=1)
    summary = summarize(
        report,
        selected_count=len(rows),
        deployment_summary={"expected_budget_runs": 3, "budget_runs": 1, "missing_budget_runs": 2},
    )

    assert len(report) == 1
    assert report[0]["status"] == "dry_run"
    assert summary["selected_commands"] == 2
    assert summary["completed_budget_runs"] == 1
    assert summary["missing_budget_runs"] == 2
    assert summary["counts"] == {"dry_run": 1}


def test_run_rows_resume_skips_existing_outputs(tmp_path: Path) -> None:
    table = tmp_path / "commands.csv"
    write_command_table(table)
    rows = selected_rows(read_rows(table), datasets={"mvtec15"}, budgets={"10"})
    run_dir = tmp_path / "runs" / rows[0]["run_name"]
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.npz").write_bytes(b"placeholder")
    (run_dir / "eval_metrics.json").write_text("{}", encoding="utf-8")

    report = run_rows(rows, root=tmp_path, dry_run=True, resume=True)

    assert report[0]["status"] == "skipped_existing"


def test_write_report_creates_runner_outputs(tmp_path: Path) -> None:
    report = [{"run_name": "x", "status": "dry_run"}]
    summary = summarize(report, selected_count=1)

    write_report(tmp_path, report, summary)

    assert (tmp_path / "table_production_budget_runner_report.csv").is_file()
    assert (tmp_path / "production_budget_runner_summary.json").is_file()
