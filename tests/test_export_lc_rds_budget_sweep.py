from __future__ import annotations

from pathlib import Path
import json

from tools.export_lc_rds_budget_sweep import (
    BUDGETS_MS,
    REQUIRED_ACTIONS,
    build_sweep,
    write_outputs,
)


def write_roi_run(root: Path) -> None:
    run_dir = root / "runs/feature_fixedpixel_mvtec15_bottle_feature_pixel_policy"
    run_dir.mkdir(parents=True)
    rows = [
        {
            "image_index": 0,
            "roi_id": 0,
            "scheduler_action": "repair-10",
            "repair_gain": 0.5,
            "area_ratio": 0.10,
            "nfe": 10,
        },
        {
            "image_index": 1,
            "roi_id": 0,
            "scheduler_action": "skip",
            "repair_gain": 0.0,
            "area_ratio": 0.05,
            "nfe": 0,
        },
    ]
    (run_dir / "roi_budget.json").write_text(json.dumps(rows), encoding="utf-8")


def test_build_sweep_measures_all_required_actions() -> None:
    action_samples, action_rows, budget_rows, _, _, summary = build_sweep(warmups=1, repeats=2)

    assert {row["action"] for row in action_rows} == set(REQUIRED_ACTIONS)
    assert {row["action"] for row in action_samples} == set(REQUIRED_ACTIONS)
    assert [row["budget_ms"] for row in budget_rows] == BUDGETS_MS
    assert summary["evidence_level"] == "measured_synthetic_action_budget_sweep_v1"
    assert summary["action_coverage_ready"] is True
    assert summary["release_gate_passed"] is False


def test_budget_sweep_respects_budget_accounting() -> None:
    _, _, budget_rows, _, _, summary = build_sweep(warmups=1, repeats=2)

    assert summary["max_budget_violation_rate"] == 0.0
    assert all(float(row["latency_ms"]) <= float(row["budget_ms"]) for row in budget_rows)
    assert all("gain_per_ms" in row for row in budget_rows)


def test_build_sweep_replays_roi_logs_with_measured_action_latency(tmp_path: Path) -> None:
    write_roi_run(tmp_path)

    _, _, _, category_rows, roi_budget_rows, summary = build_sweep(
        root=tmp_path,
        warmups=1,
        repeats=2,
    )

    assert len(category_rows) == len(BUDGETS_MS)
    assert [row["budget_ms"] for row in roi_budget_rows] == BUDGETS_MS
    assert summary["roi_measured_budget_replay_ready"] is True
    assert summary["roi_measured_categories"] == 1
    assert summary["roi_measured_images"] == 2
    assert summary["max_roi_measured_budget_violation_rate"] == 0.0


def test_write_outputs_creates_sweep_tables(tmp_path: Path) -> None:
    write_roi_run(tmp_path)
    out_dir = tmp_path / "tables/lc_rds_budget_sweep"

    summary = write_outputs(tmp_path, out_dir, warmups=1, repeats=2)

    assert summary["action_coverage_ready"] is True
    assert summary["roi_measured_budget_replay_ready"] is True
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "table_action_latency_samples.csv").is_file()
    assert (out_dir / "table_action_latency_summary.csv").is_file()
    assert (out_dir / "table_budget_sweep.csv").is_file()
    assert (out_dir / "table_category_roi_measured_budget_sweep.csv").is_file()
    assert (out_dir / "table_roi_measured_budget_sweep.csv").is_file()
