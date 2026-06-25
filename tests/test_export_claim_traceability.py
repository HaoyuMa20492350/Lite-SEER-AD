from __future__ import annotations

import json
from pathlib import Path

from tools.export_claim_traceability import build_rows, build_summary, write_outputs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_traceability_repo(root: Path, *, include_manifest_paths: bool = True) -> None:
    write_json(
        root / "artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json",
        {
            "schema": "synthetic_normal_fixed_threshold_v1_bundle",
            "policy_count": 33,
            "evaluated_run_count": 99,
            "uses_real_anomaly_labels": False,
            "uses_real_anomaly_masks": False,
            "policies": [{"max_normal_pixel_fpr": 0.004} for _ in range(33)],
        },
    )
    write_json(
        root / "tables/feature_first_fusion_aggregate_paper_package/summary.json",
        {
            "seeds": [7, 13, 23],
            "category_selection_agreement": 1.0,
            "paired_bootstrap_positive_ci": [
                {
                    "dataset": "all",
                    "metric": "aupro",
                    "categories": 33,
                    "mean_delta": 0.0584,
                    "ci95_low": 0.032,
                    "ci95_high": 0.0918,
                    "wins": 30,
                },
                {
                    "dataset": "all",
                    "metric": "pixel_ap",
                    "categories": 33,
                    "mean_delta": 0.0618,
                    "ci95_low": 0.0419,
                    "ci95_high": 0.0821,
                    "wins": 31,
                },
            ],
        },
    )
    write_json(
        root / "tables/external_baseline_comparison/summary.json",
        {
            "paired_inference": {
                "records": [
                    {
                        "method": "padim",
                        "metric": "pixel_auroc",
                        "mean_delta": 0.0189,
                        "ci95_low": 0.0106,
                        "ci95_high": 0.028,
                        "sign_test_p_holm_within_metric": 0.0068,
                    }
                ]
            }
        },
    )
    write_json(
        root / "tables/feature_first_fusion_aggregate_paper_package/module_evidence_summary.json",
        {
            "hn_sev": {"mean_delta_fprr": -0.9187, "categories_with_lower_fprr": 33},
            "lc_rds": {
                "fixed25": {"mean_delta_latency_ms": -95.04, "categories_faster": 33},
                "rule": {"mean_delta_latency_ms": -85.05, "categories_faster": 33},
                "fixed10": {"mean_delta_latency_ms": -0.51, "categories_faster": 16},
            },
        },
    )
    write_json(
        root / "tables/feature_first_fusion_aggregate_paper_package/repair_quality_summary.json",
        {
            "claim_decision": "downgrade_to_visualization_only",
            "sdr_gt": {"sdr_gt_fraction_spearman": -0.1235},
        },
    )
    write_json(
        root / "tables/diffusionad_compute_plan/summary.json",
        {
            "optimizer_steps": 663000,
            "estimated_total_gpu_hours": 3801.4,
            "categories": 15,
        },
    )
    write_json(
        root / "tables/image_score_aggregation_mvtec15/summary.json",
        {
            "adoption_decision": "retain_current_top5",
            "current_mean_image_auroc": 0.9278,
            "selected_mean_image_auroc": 0.9263,
            "mean_delta": -0.0015,
        },
    )
    write_json(
        root / "artifacts/predictions_manifest.json",
        {
            "all_predictions_present": True,
            "all_threshold_policies_present": True,
            "entry_count": 99,
        },
    )
    paths = [
        "artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json",
        "tables/feature_first_fusion_aggregate_paper_package/summary.json",
        "tables/external_baseline_comparison/summary.json",
        "tables/feature_first_fusion_aggregate_paper_package/module_evidence_summary.json",
        "tables/feature_first_fusion_aggregate_paper_package/repair_quality_summary.json",
        "tables/diffusionad_compute_plan/summary.json",
        "tables/image_score_aggregation_mvtec15/summary.json",
        "artifacts/predictions_manifest.json",
    ]
    manifest_paths = paths if include_manifest_paths else paths[:-1]
    write_json(
        root / "artifacts/manifest.json",
        {
            "schema": "lite-seer-ad-artifact-manifest-v1",
            "files": [
                {"path": path, "sha256": f"{index:064x}", "bytes": 10}
                for index, path in enumerate(manifest_paths, start=1)
            ],
        },
    )


def test_traceability_passes_when_claims_match_manifest(tmp_path: Path) -> None:
    make_traceability_repo(tmp_path)

    rows = build_rows(tmp_path)
    summary = build_summary(rows)

    assert len(rows) == 11
    assert summary["local_traceability_ready"] is True
    assert all(row["status"] == "pass" for row in rows)


def test_traceability_fails_when_evidence_missing_from_manifest(tmp_path: Path) -> None:
    make_traceability_repo(tmp_path, include_manifest_paths=False)

    rows = build_rows(tmp_path)
    summary = build_summary(rows)

    assert summary["local_traceability_ready"] is False
    assert "prediction_arrays_99_runs" in summary["failed_claims"]


def test_write_outputs_creates_traceability_artifacts(tmp_path: Path) -> None:
    make_traceability_repo(tmp_path)
    out_dir = tmp_path / "tables/claim_traceability"

    summary = write_outputs(tmp_path, out_dir)

    assert summary["claims_passed"] == 11
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "rows.json").is_file()
    assert (out_dir / "table_claim_traceability.csv").is_file()
    assert (out_dir / "claim_traceability.md").is_file()
