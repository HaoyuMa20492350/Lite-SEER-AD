from __future__ import annotations

import json
from pathlib import Path

from tools.export_completion_gap_matrix import build_rows, build_summary, write_outputs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n", encoding="utf-8")


def make_minimal_repo(root: Path) -> None:
    write_json(
        root / "tables/feature_first_plan_acceptance/summary.json",
        {
            "complete": True,
            "counts": {"pass": 19, "fail": 0, "blocked": 0, "pending_run": 0},
        },
    )
    write_json(
        root / "tables/feature_first_fusion_aggregate_paper_package/summary.json",
        {"category_selection_agreement": 1.0},
    )
    write_json(
        root
        / "tables/feature_first_fusion_aggregate_paper_package/module_evidence_summary.json",
        {
            "hn_sev": {
                "positive_repeatable_evidence": True,
                "categories_with_lower_fprr": 33,
            },
            "lc_rds": {
                "positive_repeatable_high_budget_evidence": True,
                "fixed25": {"categories_faster": 33},
                "rule": {"categories_faster": 33},
            },
            "crv": {"claim": "Visualization only."},
        },
    )
    write_json(
        root
        / "tables/feature_first_fusion_aggregate_paper_package/repair_quality_summary.json",
        {"claim_decision": "downgrade_to_visualization_only"},
    )
    write_json(
        root / "tables/image_score_aggregation_mvtec15/summary.json",
        {
            "adoption_decision": "retain_current_top5",
            "uses_real_anomaly_labels_for_selection": False,
            "uses_real_anomaly_masks_for_selection": False,
        },
    )
    write_json(
        root / "tables/external_baseline_comparison/summary.json",
        {
            "methods": [
                "patchcore",
                "padim",
                "uniad",
                "draem",
                "ddad",
                "rd4ad",
                "simplenet",
            ],
            "uses_real_anomaly_labels_for_threshold": False,
            "uses_real_anomaly_masks_for_threshold": False,
        },
    )
    write_json(
        root / "artifacts/predictions_manifest.json",
        {
            "all_predictions_present": True,
            "all_threshold_policies_present": True,
        },
    )
    write_json(root / "tables/diffusionad_compute_plan/summary.json", {})
    touch(root / "artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json")
    touch(root / "artifacts/manifest.json")
    touch(root / "docs/claim_freeze_2026_06_21.md")
    (root / "docs/claim_freeze_2026_06_21.md").write_text(
        "Feature-first Lite-SEER-AD\n", encoding="utf-8"
    )
    touch(root / "paper/manuscript.md")
    (root / "paper/manuscript.md").write_text(
        "Lite-SEER-AD: Label-Free Feature-First Industrial Anomaly Localization\n",
        encoding="utf-8",
    )
    for path in [
        "requirements-lock.txt",
        "environment.yml",
        "Dockerfile",
        ".github/workflows/ci.yml",
        "CITATION.cff",
        ".zenodo.json",
    ]:
        touch(root / path)


def row_by_dimension(rows: list[dict], dimension: str) -> dict:
    return next(row for row in rows if row["dimension"] == dimension)


def test_gap_matrix_marks_completed_and_blocking_dimensions(tmp_path: Path) -> None:
    make_minimal_repo(tmp_path)

    rows = build_rows(tmp_path)
    summary = build_summary(rows, "2026-06-21")

    assert row_by_dimension(rows, "主检测器与 label-free 像素策略")["status"] == "complete"
    assert row_by_dimension(rows, "CRV")["status"] == "complete_with_declared_limit"
    assert row_by_dimension(rows, "HN-SEV")["status"] == "partial"
    assert row_by_dimension(rows, "扩散修复必要性")["status"] == "pending"
    assert "HN-SEV" in summary["blocking_p0_dimensions"]
    assert "扩散修复必要性" in summary["blocking_p0_dimensions"]
    assert summary["default_100_ready"] is False


def test_hn_sev_claim_bounded_evidence_is_not_a_default_blocker(tmp_path: Path) -> None:
    make_minimal_repo(tmp_path)
    write_json(
        tmp_path / "tables/hn_sev_retention_calibration/summary.json",
        {
            "evidence_level": "roi_mask_retention_calibration_v1",
            "categories": 33,
            "release_gate_passed": False,
            "overall": {
                "roi_rows_resolved": 5091,
                "tp_retention": 0.16,
                "roi_recall_before_hn_sev": 0.87,
                "roi_recall_after_hn_sev": 0.15,
                "background_suppression_rate": 0.93,
            },
        },
    )
    write_json(
        tmp_path / "tables/hn_sev_input_ablation/summary.json",
        {
            "evidence_level": "mixed_table_run_metadata_coverage_v2",
            "categories": 33,
            "release_gate_passed": True,
        },
    )
    (tmp_path / "docs/results_limitations_draft.md").write_text(
        "HN-SEV currently has a recall-safety limitation.\n",
        encoding="utf-8",
    )

    rows = build_rows(tmp_path)
    summary = build_summary(rows, "2026-06-21")
    hn_row = row_by_dimension(rows, "HN-SEV")

    assert hn_row["status"] == "complete_with_declared_limit"
    assert hn_row["current_completion_percent"] == 100
    assert hn_row["blocking_for_default_100"] is False
    assert "HN-SEV" not in summary["blocking_p0_dimensions"]


def test_deployment_energy_measurement_removes_energy_gap_only(tmp_path: Path) -> None:
    make_minimal_repo(tmp_path)
    write_json(
        tmp_path / "tables/deployment_latency/summary.json",
        {
            "evidence_level": "synchronized_component_smoke_v1",
            "latency_protocol": "synchronized_batch_latency_v1",
            "latency_ms_p95": 1.0,
            "latency_ms_p99": 2.0,
            "budget_violation_rate": 0.0,
            "hardware": {"platform": "Windows"},
        },
    )
    write_json(
        tmp_path / "tables/deployment_readiness/summary.json",
        {"smoke_protocol_ready": True, "production_deployment_ready": False},
    )
    write_json(
        tmp_path / "tables/deployment_production_latency/summary.json",
        {
            "production_component_latency_ready": True,
            "categories": 33,
            "component_latency_protocol": "per_image_synchronized_component_breakdown_v1",
            "multi_action_budget_sweep_ready": True,
            "max_budget_violation_rate": 0.0,
            "energy_measurement_ready": True,
            "hardware_profiles": 1,
            "cross_hardware_ready": False,
        },
    )

    rows = build_rows(tmp_path)
    deployment = row_by_dimension(rows, "效率与部署")

    assert deployment["status"] == "partial"
    assert deployment["current_completion_percent"] == 96
    assert "跨硬件验证" in deployment["missing_to_100"]
    assert "能耗测量" not in deployment["missing_to_100"]


def test_failure_taxonomy_closes_weak_category_as_declared_limit(tmp_path: Path) -> None:
    make_minimal_repo(tmp_path)
    write_json(
        tmp_path / "tables/failure_taxonomy/summary.json",
        {
            "release_gate_passed": True,
            "all_required_weak_categories_covered": True,
            "covered_weak_categories": 9,
            "required_weak_categories": 9,
        },
    )

    rows = build_rows(tmp_path)
    weak = row_by_dimension(rows, "弱类别")

    assert weak["status"] == "complete_with_declared_limit"
    assert weak["current_completion_percent"] == 100
    assert "failure taxonomy" in weak["missing_to_100"]


def test_write_outputs_creates_json_csv_and_markdown(tmp_path: Path) -> None:
    make_minimal_repo(tmp_path)
    out_dir = tmp_path / "tables/completion_gap_matrix"

    summary = write_outputs(tmp_path, out_dir, "2026-06-21")

    assert summary["schema"] == "lite-seer-ad-completion-gap-matrix-v1"
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "rows.json").is_file()
    assert (out_dir / "table_completion_gap_matrix.csv").is_file()
    markdown = (out_dir / "completion_gap_matrix.md").read_text(encoding="utf-8")
    assert "Lite-SEER-AD completion gap matrix" in markdown
    assert "HN-SEV" in markdown
