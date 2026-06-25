"""Export deployment-readiness gates from latency and LC-RDS audits.

This audit does not re-measure latency. It converts the existing smoke latency
table and LC-RDS budget audit into a requirement-by-requirement readiness table
so deployment claims cannot rely on a single aggregate number.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_COMPONENTS = ["io", "detector", "verifier", "scheduler", "repair", "end_to_end"]
REQUIRED_BUDGETS = [10, 25, 50, 75, 100, 150]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def status_row(
    requirement: str,
    status: str,
    evidence: str,
    detail: str,
    gate: str = "smoke",
) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "status": status,
        "gate": gate,
        "evidence": evidence,
        "detail": detail,
    }


def build_rows(root: Path) -> list[dict[str, Any]]:
    latency = read_json(root / "tables/deployment_latency/summary.json")
    budget = read_json(root / "tables/lc_rds_budget_audit/summary.json")
    budget_sweep = read_json(root / "tables/lc_rds_budget_sweep/summary.json")
    production = read_json(root / "tables/deployment_production_latency/summary.json")
    hardware = latency.get("hardware", {}) or {}
    rows = [
        status_row(
            "protocol:synchronized_batch_latency_v1",
            "pass" if latency.get("latency_protocol") == "synchronized_batch_latency_v1" else "fail",
            "tables/deployment_latency/summary.json",
            str(latency.get("latency_protocol")),
        ),
        status_row(
            "protocol:batch_size_1",
            "pass" if latency.get("latency_batch_size") == 1 else "fail",
            "tables/deployment_latency/summary.json",
            f"batch={latency.get('latency_batch_size')}",
        ),
        status_row(
            "protocol:warmups_and_measurements_recorded",
            "pass"
            if int(latency.get("latency_warmups", 0)) > 0 and int(latency.get("latency_repeats", 0)) > 0
            else "fail",
            "tables/deployment_latency/summary.json",
            f"warmups={latency.get('latency_warmups')}; repeats={latency.get('latency_repeats')}",
        ),
        status_row(
            "protocol:component_breakdown_shape",
            "pass" if latency.get("components") == REQUIRED_COMPONENTS else "fail",
            "tables/deployment_latency/table_component_latency.csv",
            " ".join(latency.get("components", []) or []),
        ),
        status_row(
            "metric:p95_p99_present",
            "pass" if "latency_ms_p95" in latency and "latency_ms_p99" in latency else "fail",
            "tables/deployment_latency/summary.json",
            f"p95={latency.get('latency_ms_p95')}; p99={latency.get('latency_ms_p99')}",
        ),
        status_row(
            "metric:gpu_memory_recorded",
            "pass" if "gpu_memory_mb" in latency else "fail",
            "tables/deployment_latency/summary.json",
            f"gpu_memory_mb={latency.get('gpu_memory_mb')}",
        ),
        status_row(
            "hardware:primary_hardware_metadata",
            "pass" if hardware.get("platform") and "cpu_count" in hardware else "fail",
            "tables/deployment_latency/summary.json",
            f"device={hardware.get('device')}; gpu={hardware.get('gpu_name')}",
        ),
        status_row(
            "budget:offline_six_budget_replay",
            "pass"
            if budget.get("evidence_level") == "offline_replay_from_roi_logs_v1"
            and budget.get("budgets_ms") == REQUIRED_BUDGETS
            else "fail",
            "tables/lc_rds_budget_audit/summary.json",
            f"evidence={budget.get('evidence_level')}; budgets={budget.get('budgets_ms')}",
        ),
        status_row(
            "budget:violation_rate_recorded",
            "pass" if "max_budget_violation_rate" in budget else "fail",
            "tables/lc_rds_budget_audit/summary.json",
            f"max_budget_violation_rate={budget.get('max_budget_violation_rate')}",
        ),
        status_row(
            "budget:synthetic_action_sweep",
            "pass"
            if budget_sweep.get("evidence_level") == "measured_synthetic_action_budget_sweep_v1"
            and budget_sweep.get("budgets_ms") == REQUIRED_BUDGETS
            and budget_sweep.get("action_coverage_ready") is True
            and float(budget_sweep.get("max_budget_violation_rate", 1.0)) <= 0.01
            else "fail",
            "tables/lc_rds_budget_sweep/summary.json",
            (
                f"evidence={budget_sweep.get('evidence_level')}; "
                f"action_coverage_ready={budget_sweep.get('action_coverage_ready')}; "
                f"max_budget_violation_rate={budget_sweep.get('max_budget_violation_rate')}"
            ),
        ),
        status_row(
            "budget:roi_measured_budget_replay",
            "pass"
            if budget_sweep.get("roi_measured_budget_replay_ready") is True
            and int(budget_sweep.get("roi_measured_categories", 0)) >= 33
            and float(budget_sweep.get("max_roi_measured_budget_violation_rate", 1.0)) <= 0.01
            else "fail",
            "tables/lc_rds_budget_sweep/summary.json",
            (
                f"categories={budget_sweep.get('roi_measured_categories')}; "
                f"observed_actions={budget_sweep.get('roi_measured_observed_actions')}; "
                f"max_violation={budget_sweep.get('max_roi_measured_budget_violation_rate')}"
            ),
        ),
        status_row(
            "production:real_pipeline_components",
            "pass" if production.get("production_component_latency_ready") is True else "blocked",
            "tables/deployment_production_latency/summary.json",
            (
                f"categories={production.get('categories')}; "
                f"images={production.get('images')}; "
                f"protocol={production.get('component_latency_protocol')}"
            )
            if production
            else "missing production inference component-latency summary",
            gate="production",
        ),
        status_row(
            "production:measured_lc_rds_budget_sweep",
            "pass"
            if production.get("multi_action_budget_sweep_ready") is True
            and production.get("release_gate_passed") is True
            else "blocked",
            "tables/deployment_production_latency/summary.json",
            (
                f"observed_actions={production.get('observed_actions')}; "
                f"missing_budget_runs={production.get('missing_budget_runs')}; "
                f"max_budget_violation_rate={production.get('max_budget_violation_rate')}; "
                f"budgets={production.get('budgets_ms')}"
            )
            if production
            else f"current synthetic evidence={budget_sweep.get('evidence_level')}",
            gate="production",
        ),
        status_row(
            "production:cross_hardware",
            "pass" if production.get("cross_hardware_ready") is True else "blocked",
            "tables/deployment_production_latency/summary.json",
            (
                f"hardware_profiles={production.get('hardware_profiles')}"
                if production
                else "production hardware profiles are not summarized"
            ),
            gate="production",
        ),
        status_row(
            "production:energy_measurement",
            "pass"
            if production.get("energy_measurement_ready") is True or latency.get("energy_joules") is not None
            else "pending",
            "tables/deployment_production_latency/summary.json",
            (
                "production energy measurement is ready"
                if production.get("energy_measurement_ready") is True
                else "energy_joules is not measured"
                if latency.get("energy_joules") is None
                else f"energy_joules={latency.get('energy_joules')}"
            ),
            gate="production",
        ),
    ]
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    smoke_rows = [row for row in rows if row["gate"] == "smoke"]
    production_rows = [row for row in rows if row["gate"] == "production"]
    smoke_ready = all(row["status"] == "pass" for row in smoke_rows)
    production_ready = smoke_ready and all(row["status"] == "pass" for row in production_rows)
    blockers = [row["requirement"] for row in rows if row["status"] != "pass"]
    return {
        "schema": "lite-seer-ad-deployment-readiness-v1",
        "smoke_protocol_ready": smoke_ready,
        "production_deployment_ready": production_ready,
        "release_gate_passed": production_ready,
        "release_gate_reason": (
            "Smoke protocol is ready, but these deployment blockers remain: "
            + ", ".join(blockers)
            if smoke_ready and not production_ready
            else "Deployment readiness gate passed."
            if production_ready
            else "Smoke deployment protocol is incomplete."
        ),
        "counts": counts,
        "blocking_requirements": blockers,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["requirement", "status", "gate", "evidence", "detail"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(root: Path, out_dir: Path) -> dict[str, Any]:
    rows = build_rows(root)
    summary = build_summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_deployment_readiness.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("tables/deployment_readiness"))
    args = parser.parse_args()
    summary = write_outputs(args.root, args.out_dir)
    print(
        f"Wrote deployment readiness to {args.out_dir} "
        f"(production_deployment_ready={summary['production_deployment_ready']})"
    )


if __name__ == "__main__":
    main()
