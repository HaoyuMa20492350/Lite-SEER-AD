from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.export_production_deployment_latency import build_outputs, write_outputs


def write_run(
    root: Path,
    name: str,
    category: str,
    actions: list[str],
    *,
    latency_budget_ms: float | None = None,
) -> None:
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "run_args.json").write_text(
        json.dumps(
            {
                "args": {
                    "config": "configs/mvtec.yaml",
                    "category": category,
                    "ablation": "utility_lc_rds",
                    "latency_budget_ms": latency_budget_ms,
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        "lc_rds:\n  latency_budget_ms: 60.0\n",
        encoding="utf-8",
    )
    with (run_dir / "component_latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "detector_latency_ms",
                "hn_sev_latency_ms",
                "repair_latency_ms",
                "end_to_end_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "index": 0,
                "detector_latency_ms": 10.0,
                "hn_sev_latency_ms": 1.0,
                "repair_latency_ms": 20.0,
                "end_to_end_latency_ms": 31.0,
            }
        )
        writer.writerow(
            {
                "index": 1,
                "detector_latency_ms": 12.0,
                "hn_sev_latency_ms": 2.0,
                "repair_latency_ms": 0.0,
                "end_to_end_latency_ms": 14.0,
            }
        )
    rows = []
    for idx, action in enumerate(actions):
        rows.append(
            {
                "image_index": idx % 2,
                "scheduler_action": action,
                "action_latency_ms": 20.0 if action != "skip" else 0.0,
                "cumulative_spent_ms": 20.0 if action != "skip" else 0.0,
            }
        )
    (run_dir / "roi_budget.json").write_text(json.dumps(rows), encoding="utf-8")


def test_build_outputs_marks_component_ready_without_overclaiming_budget_sweep(tmp_path: Path) -> None:
    write_run(tmp_path, "fulltest_mvtec15_bottle_utility_lc_rds", "bottle", ["skip", "repair-10"])

    component_rows, budget_rows, coverage_rows, missing_commands, summary = build_outputs(
        tmp_path,
        min_categories=1,
    )

    assert len(component_rows) == 1
    assert [row["budget_ms"] for row in budget_rows] == [10, 25, 50, 75, 100, 150]
    assert {row["action"] for row in coverage_rows if row["observed"]} == {"skip", "repair-10"}
    assert len(missing_commands) == 6
    assert {row["budget_ms"] for row in missing_commands} == {10, 25, 50, 75, 100, 150}
    assert all("--latency-budget-ms" in row["infer_command"] for row in missing_commands)
    assert summary["production_component_latency_ready"] is True
    assert summary["budget_sweep_coverage_ready"] is False
    assert summary["missing_budget_runs"] == 6
    assert summary["multi_action_budget_sweep_ready"] is False
    assert summary["release_gate_passed"] is False


def test_write_outputs_creates_production_latency_tables(tmp_path: Path) -> None:
    write_run(tmp_path, "fulltest_mvtec15_bottle_utility_lc_rds", "bottle", ["skip", "repair-10"])
    out_dir = tmp_path / "tables/deployment_production_latency"

    summary = write_outputs(tmp_path, out_dir, min_categories=1)

    assert summary["categories"] == 1
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "table_production_component_latency.csv").is_file()
    assert (out_dir / "table_production_budget_replay.csv").is_file()
    assert (out_dir / "table_production_action_coverage.csv").is_file()
    assert (out_dir / "table_missing_production_budget_commands.csv").is_file()


def test_build_outputs_counts_existing_budget_run_prefix(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "fulltest_mvtec15_bottle_utility_lc_rds",
        "bottle",
        ["skip", "repair-10"],
    )
    write_run(
        tmp_path,
        "production_lc_rds_budget_mvtec15_bottle_budget10_utility_lc_rds",
        "bottle",
        ["skip", "repair-5"],
        latency_budget_ms=10.0,
    )

    _, budget_rows, _, missing_commands, summary = build_outputs(tmp_path, min_categories=1)

    assert len(missing_commands) == 5
    assert {row["budget_ms"] for row in missing_commands} == {25, 50, 75, 100, 150}
    assert summary["budget_runs"] == 1
    assert summary["missing_budget_runs"] == 5
    assert next(row for row in budget_rows if row["budget_ms"] == 10)["runs"] == 1


def test_build_outputs_reads_hardware_and_energy_sidecars(tmp_path: Path) -> None:
    write_run(tmp_path, "fulltest_mvtec15_bottle_utility_lc_rds", "bottle", ["skip"])
    write_run(
        tmp_path,
        "production_lc_rds_budget_mvtec15_bottle_budget10_utility_lc_rds",
        "bottle",
        ["skip"],
        latency_budget_ms=10.0,
    )
    deployment_latency = tmp_path / "tables/deployment_latency"
    deployment_latency.mkdir(parents=True)
    (deployment_latency / "summary.json").write_text(
        json.dumps(
            {
                "hardware": {
                    "platform": "Windows",
                    "processor": "cpu-a",
                    "cpu_count": 8,
                    "device": "cuda",
                    "gpu_name": "gpu-a",
                    "python": "3.12",
                    "torch": "2.4",
                }
            }
        ),
        encoding="utf-8",
    )
    profile_dir = tmp_path / "tables/deployment_production_latency/hardware_profiles"
    profile_dir.mkdir(parents=True)
    (profile_dir / "second_gpu.json").write_text(
        json.dumps(
            {
                "platform": "Linux",
                "processor": "cpu-b",
                "cpu_count": 16,
                "device": "cuda",
                "gpu_name": "gpu-b",
                "python": "3.12",
                "torch": "2.4",
            }
        ),
        encoding="utf-8",
    )
    energy_dir = tmp_path / "tables/deployment_production_latency/energy_measurements"
    energy_dir.mkdir(parents=True)
    (energy_dir / "bottle_budget10.json").write_text(
        json.dumps({"energy_joules": 12.5, "protocol": "nvidia_smi_power_poll_v1"}),
        encoding="utf-8",
    )

    *_, summary = build_outputs(tmp_path, min_categories=1)

    assert summary["hardware_profiles"] == 2
    assert summary["cross_hardware_ready"] is True
    assert summary["energy_measurements"] == 1
    assert summary["energy_measurement_ready"] is True
