from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.export_lc_rds_budget_audit import (
    build_audit,
    parse_budget_v2_config,
    replay_budget,
    write_outputs,
)


def write_roi_run(root: Path, dataset: str, category: str) -> None:
    run_dir = root / "runs" / f"feature_fixedpixel_{dataset}_{category}_feature_pixel_policy"
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
            "image_index": 0,
            "roi_id": 1,
            "scheduler_action": "repair-25",
            "repair_gain": 0.4,
            "area_ratio": 0.20,
            "nfe": 25,
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
    with (run_dir / "efficiency.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "latency_ms_mean", "value": "42.0"})
        writer.writerow({"metric": "nfe_mean", "value": "10.0"})
        writer.writerow({"metric": "repaired_area_ratio_mean", "value": "0.1"})


def write_budget_v2_config(root: Path) -> None:
    path = root / "configs/scheduler/lc_rds_budget_v2.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "schema: lc_rds_budget_v2",
                "actions:",
                "  - skip",
                "  - repair5",
                "  - repair10",
                "  - repair25",
                "  - native_refine",
                "budgets_ms: [10, 25, 50, 75, 100, 150]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_budget_v2_config_canonicalizes_actions(tmp_path: Path) -> None:
    write_budget_v2_config(tmp_path)

    actions, budgets = parse_budget_v2_config(tmp_path)

    assert actions == ["skip", "repair-5", "repair-10", "repair-25", "native-refine"]
    assert budgets == [10, 25, 50, 75, 100, 150]


def test_replay_budget_respects_latency_budget() -> None:
    rows = [
        {"scheduler_action": "repair-10", "repair_gain": 0.5, "area_ratio": 0.1},
        {"scheduler_action": "repair-25", "repair_gain": 0.8, "area_ratio": 0.2},
    ]

    tight = replay_budget(rows, 25.0)
    loose = replay_budget(rows, 150.0)

    assert tight["latency_ms"] == 0.0
    assert tight["repair_gain"] == 0.0
    assert loose["latency_ms"] == 115.0
    assert loose["repair_gain"] == 1.3
    assert loose["budget_violation"] == 0.0


def test_build_audit_summarizes_runs_and_keeps_release_gate_false(tmp_path: Path) -> None:
    write_budget_v2_config(tmp_path)
    write_roi_run(tmp_path, "mvtec15", "bottle")

    run_rows, category_rows, budget_rows, action_rows, summary = build_audit(tmp_path)

    assert len(run_rows) == 1
    assert len(category_rows) == 6
    assert len(action_rows) == 5
    assert [row["budget_ms"] for row in budget_rows] == [10, 25, 50, 75, 100, 150]
    assert summary["categories"] == 1
    assert summary["evidence_level"] == "offline_replay_from_roi_logs_v1"
    assert summary["release_gate_passed"] is False
    assert summary["action_space"]["action_space_ready"] is True
    assert "repair-5" in summary["action_space"]["missing_observed_actions"]
    assert "measured synchronized" in summary["required_for_release"][0]


def test_write_outputs_creates_budget_tables(tmp_path: Path) -> None:
    write_budget_v2_config(tmp_path)
    write_roi_run(tmp_path, "visa", "pcb1")
    out_dir = tmp_path / "tables/lc_rds_budget_audit"

    summary = write_outputs(tmp_path, out_dir)

    assert summary["runs"] == 1
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "table_observed_runs.csv").is_file()
    assert (out_dir / "table_category_budget_replay.csv").is_file()
    assert (out_dir / "table_budget_summary.csv").is_file()
    assert (out_dir / "table_action_space.csv").is_file()
