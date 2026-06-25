from __future__ import annotations

import json
from pathlib import Path

from tools.export_deployment_readiness import build_rows, build_summary, write_outputs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_deployment_repo(
    root: Path,
    *,
    measured_budget: bool = False,
    energy: bool = False,
    production_components: bool = False,
    production_multi_action: bool = False,
) -> None:
    write_json(
        root / "tables/deployment_latency/summary.json",
        {
            "latency_protocol": "synchronized_batch_latency_v1",
            "latency_batch_size": 1,
            "latency_warmups": 10,
            "latency_repeats": 50,
            "components": ["io", "detector", "verifier", "scheduler", "repair", "end_to_end"],
            "latency_ms_p95": 1.2,
            "latency_ms_p99": 1.5,
            "gpu_memory_mb": 12.0,
            "energy_joules": 0.7 if energy else None,
            "hardware": {
                "platform": "test",
                "cpu_count": 4,
                "device": "cpu",
                "gpu_name": None,
            },
        },
    )
    write_json(
        root / "tables/lc_rds_budget_audit/summary.json",
        {
            "evidence_level": "offline_replay_from_roi_logs_v1",
            "budgets_ms": [10, 25, 50, 75, 100, 150],
            "max_budget_violation_rate": 0.0,
        },
    )
    write_json(
        root / "tables/lc_rds_budget_sweep/summary.json",
        {
            "evidence_level": "measured_budget_sweep_v1"
            if measured_budget
            else "measured_synthetic_action_budget_sweep_v1",
            "release_gate_passed": measured_budget,
            "budgets_ms": [10, 25, 50, 75, 100, 150],
            "action_coverage_ready": True,
            "max_budget_violation_rate": 0.0,
            "roi_measured_budget_replay_ready": True,
            "roi_measured_categories": 33,
            "roi_measured_observed_actions": ["repair-10"],
            "max_roi_measured_budget_violation_rate": 0.0,
        },
    )
    if production_components:
        write_json(
            root / "tables/deployment_production_latency/summary.json",
            {
                "evidence_level": "production_inference_component_latency_v1",
                "production_component_latency_ready": True,
                "multi_action_budget_sweep_ready": production_multi_action,
                "release_gate_passed": production_multi_action,
                "categories": 33,
                "images": 100,
                "component_latency_protocol": "per_image_synchronized_component_breakdown_v1",
                "observed_actions": ["skip", "repair-10"]
                if not production_multi_action
                else ["skip", "repair-5", "repair-10", "repair-25", "native-refine"],
                "max_budget_violation_rate": 0.0,
                "budgets_ms": [10, 25, 50, 75, 100, 150],
                "cross_hardware_ready": False,
                "hardware_profiles": 1,
                "energy_measurement_ready": False,
            },
        )


def test_deployment_readiness_marks_smoke_ready_but_production_blocked(
    tmp_path: Path,
) -> None:
    make_deployment_repo(tmp_path)

    rows = build_rows(tmp_path)
    summary = build_summary(rows)

    assert summary["smoke_protocol_ready"] is True
    assert summary["production_deployment_ready"] is False
    assert "production:real_pipeline_components" in summary["blocking_requirements"]
    assert "production:energy_measurement" in summary["blocking_requirements"]
    assert "budget:synthetic_action_sweep" not in summary["blocking_requirements"]
    assert "budget:roi_measured_budget_replay" not in summary["blocking_requirements"]


def test_write_outputs_creates_readiness_tables(tmp_path: Path) -> None:
    make_deployment_repo(tmp_path)
    out_dir = tmp_path / "tables/deployment_readiness"

    summary = write_outputs(tmp_path, out_dir)

    assert summary["release_gate_passed"] is False
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "table_deployment_readiness.csv").is_file()


def test_deployment_readiness_uses_production_component_latency_summary(
    tmp_path: Path,
) -> None:
    make_deployment_repo(tmp_path, production_components=True)

    rows = build_rows(tmp_path)
    summary = build_summary(rows)
    statuses = {row["requirement"]: row["status"] for row in rows}

    assert statuses["production:real_pipeline_components"] == "pass"
    assert statuses["production:measured_lc_rds_budget_sweep"] == "blocked"
    assert summary["production_deployment_ready"] is False
    assert "production:real_pipeline_components" not in summary["blocking_requirements"]


def test_summary_can_pass_when_rows_are_all_pass() -> None:
    rows = [
        {"requirement": "protocol", "status": "pass", "gate": "smoke"},
        {"requirement": "production", "status": "pass", "gate": "production"},
    ]

    summary = build_summary(rows)

    assert summary["smoke_protocol_ready"] is True
    assert summary["production_deployment_ready"] is True
