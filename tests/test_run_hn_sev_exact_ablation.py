from __future__ import annotations

import csv
from pathlib import Path

from tools.run_hn_sev_exact_ablation import (
    output_ready,
    run_rows,
    selected_rows,
    summarize,
)


def row(name: str = "synthetic_only_sev") -> dict[str, str]:
    return {
        "dataset": "mvtec15",
        "category": "bottle",
        "requirement": "synthetic_only",
        "target_ablation": name,
        "train_run_name": f"hn_sev_exact_mvtec15_bottle_{name}",
        "eval_run_name": f"hn_sev_exact_eval_mvtec15_bottle_{name}",
        "train_command": "python train_hn_sev.py",
        "infer_command": "python infer.py",
        "evaluate_command": "python evaluate.py",
    }


def test_selected_rows_filters_requirements() -> None:
    rows = [row("synthetic_only_sev"), {**row("clean_normal_sev"), "requirement": "clean_normal_added"}]

    selected = selected_rows(rows, requirements={"clean_normal_added"})

    assert len(selected) == 1
    assert selected[0]["target_ablation"] == "clean_normal_sev"


def test_output_ready_requires_train_and_eval_artifacts(tmp_path: Path) -> None:
    item = row()
    train_dir = tmp_path / "runs" / item["train_run_name"]
    eval_dir = tmp_path / "runs" / item["eval_run_name"]
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (train_dir / "hn_sev.pt").write_bytes(b"x")
    (eval_dir / "predictions.npz").write_bytes(b"x")

    assert output_ready(item, tmp_path) is False

    (eval_dir / "eval_metrics.json").write_text("{}", encoding="utf-8")
    assert output_ready(item, tmp_path) is True


def test_run_rows_dry_run_and_resume(tmp_path: Path) -> None:
    item = row()

    dry_report = run_rows([item], root=tmp_path, dry_run=True)
    assert dry_report[0]["status"] == "dry_run"

    train_dir = tmp_path / "runs" / item["train_run_name"]
    eval_dir = tmp_path / "runs" / item["eval_run_name"]
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (train_dir / "hn_sev.pt").write_bytes(b"x")
    (eval_dir / "predictions.npz").write_bytes(b"x")
    (eval_dir / "eval_metrics.json").write_text("{}", encoding="utf-8")

    report = run_rows([item], root=tmp_path, dry_run=False)
    assert report[0]["status"] == "skipped_existing"


def test_summarize_marks_ready_for_refresh() -> None:
    summary = summarize(
        [{"status": "passed"}, {"status": "skipped_existing"}],
        2,
        audit_summary={"categories": 33, "complete_exact_categories": 1},
    )

    assert summary["counts"] == {"passed": 1, "skipped_existing": 1}
    assert summary["ready_for_audit_refresh"] is True
    assert summary["audit_categories"] == 33
