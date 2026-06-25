from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ["aupro", "pixel_ap", "dice"]
CANONICAL_PREDICTION_KEYS = {
    "detection_heatmaps",
    "verification_heatmaps",
    "image_score_heatmaps",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    out_dir = Path("tables/feature_first_plan_acceptance")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    gate_roots = {
        "mvtec15": Path("tables/synthetic_gate_fusion_aggregate_mvtec15"),
        "visa": Path("tables/synthetic_gate_fusion_aggregate_visa"),
        "mpdd": Path("tables/synthetic_gate_fusion_aggregate_mpdd"),
    }
    selection_rows = []
    for root in gate_roots.values():
        for seed_dir in root.glob("seed*"):
            selection_rows.extend(read_csv(seed_dir / "selection.csv"))
    no_label_selection = len(selection_rows) == 99 and all(
        not bool_value(row.get("uses_real_anomaly_labels_for_selection"))
        and not bool_value(row.get("uses_real_anomaly_masks_for_selection"))
        and row.get("selection_evidence_seeds") == "7 13 23"
        for row in selection_rows
    )
    rows.append(
        {
            "requirement": "paper_selection_uses_no_real_anomaly_labels_or_masks",
            "status": "pass" if no_label_selection else "fail",
            "value": len(selection_rows),
            "threshold": (
                "99 rows, all label/mask flags false, and evidence seeds 7/13/23"
            ),
            "evidence": "tables/synthetic_gate_fusion_aggregate_*/seed*/selection.csv",
        }
    )

    selected_predictions = sorted(
        path
        for root in gate_roots.values()
        for path in root.rglob("predictions.npz")
        if "selected_runs" in path.parts
    )
    schema_failures = []
    for path in selected_predictions:
        with np.load(path, allow_pickle=False) as artifact:
            if not CANONICAL_PREDICTION_KEYS.issubset(artifact.files):
                schema_failures.append(str(path))
    rows.append(
        {
            "requirement": "predictions_three_heatmap_schema_explicit",
            "status": (
                "pass"
                if len(selected_predictions) == 99 and not schema_failures
                else "fail"
            ),
            "value": (
                f"files={len(selected_predictions)};"
                f"schema_failures={len(schema_failures)}"
            ),
            "threshold": (
                "99 selected predictions with detection_heatmaps, "
                "verification_heatmaps, image_score_heatmaps"
            ),
            "evidence": (
                "tables/synthetic_gate_fusion_aggregate_*/"
                "seed*/selected_runs/*/predictions.npz"
            ),
        }
    )

    freeze = read_json(
        Path("tables/protocol_freeze_20260613_final/protocol_manifest.json")
    )
    frozen_configs = (freeze.get("files", {}) or {}).get("configs", [])
    frozen_evidence = (freeze.get("files", {}) or {}).get("evidence", [])
    freeze_complete = (
        freeze.get("paper_protocol")
        == "normal_plus_synthetic_cross_seed_mean_no_real_anomaly_labels"
        and freeze.get("seeds") == [7, 13, 23]
        and len(freeze.get("candidate_templates", [])) >= 7
        and len(frozen_configs) == 4
        and all(row.get("exists") and row.get("sha256") for row in frozen_configs)
        and len(frozen_evidence) >= 6
        and all(row.get("exists") and row.get("sha256") for row in frozen_evidence)
        and len((freeze.get("git", {}) or {}).get("untracked_files", [])) > 0
    )
    rows.append(
        {
            "requirement": "paper_evidence_environment_and_protocol_frozen",
            "status": "pass" if freeze_complete else "fail",
            "value": (
                f"configs={len(frozen_configs)};"
                f"candidates={len(freeze.get('candidate_templates', []))};"
                f"evidence={len(frozen_evidence)};"
                f"untracked_hashed={len((freeze.get('git', {}) or {}).get('untracked_files', []))}"
            ),
            "threshold": (
                "cross-seed protocol, seeds 7/13/23, >=7 candidates, "
                "4 hashed configs, >=6 hashed evidence files, untracked hashes"
            ),
            "evidence": (
                "tables/protocol_freeze_20260613_final/"
                "protocol_manifest.json"
            ),
        }
    )

    seed_runs = sum(
        len(read_json(root / "synthetic_gate_run_report.json").get("runs", []))
        for root in gate_roots.values()
    )
    rows.append(
        {
            "requirement": "three_datasets_three_seeds_complete",
            "status": "pass" if seed_runs == 9 else "fail",
            "value": seed_runs,
            "threshold": 9,
            "evidence": "tables/synthetic_gate_fusion_aggregate_*",
        }
    )

    stability = read_json(
        Path("tables/feature_first_fusion_aggregate_paper_package/summary.json")
    )
    agreement = as_float(stability.get("category_selection_agreement"))
    rows.append(
        {
            "requirement": "category_selection_stability",
            "status": "pass" if agreement >= 0.95 else "fail",
            "value": agreement,
            "threshold": ">=0.95",
            "evidence": "tables/feature_first_fusion_aggregate_paper_package/summary.json",
        }
    )

    delta_std_rows = read_csv(
        Path("tables/synthetic_gate_fusion_aggregate_stability/table_delta_mean_std.csv")
    )
    for metric in ["pixel_ap", "dice"]:
        row = next(
            (
                item
                for item in delta_std_rows
                if item.get("metric") == f"delta_{metric}"
            ),
            {},
        )
        value = as_float(row.get("std"))
        rows.append(
            {
                "requirement": f"three_seed_delta_std_{metric}",
                "status": "pass" if value <= 0.003 else "fail",
                "value": value,
                "threshold": "<=0.003",
                "evidence": "tables/synthetic_gate_fusion_aggregate_stability/table_delta_mean_std.csv",
            }
        )

    bootstrap = read_csv(
        Path("tables/feature_first_fusion_aggregate_paper_package/table_paired_bootstrap_ci.csv")
    )
    for metric in METRICS:
        row = next(
            (
                item
                for item in bootstrap
                if item.get("dataset") == "all" and item.get("metric") == metric
            ),
            {},
        )
        mean_delta = as_float(row.get("mean_delta"))
        ci_low = as_float(row.get("ci95_low"))
        rows.append(
            {
                "requirement": f"overall_{metric}_mean_and_ci_positive",
                "status": "pass" if mean_delta > 0 and ci_low > 0 else "fail",
                "value": f"mean={mean_delta:.6f};ci_low={ci_low:.6f}",
                "threshold": "mean>0 and ci95_low>0",
                "evidence": "tables/feature_first_fusion_aggregate_paper_package/table_paired_bootstrap_ci.csv",
            }
        )
        if metric in {"pixel_ap", "dice"}:
            wins = int(float(row.get("wins", 0)))
            rows.append(
                {
                    "requirement": f"best_baseline_category_wins_{metric}",
                    "status": "pass" if wins >= 27 else "fail",
                    "value": f"{wins}/33",
                    "threshold": ">=27/33",
                    "evidence": "tables/feature_first_fusion_aggregate_paper_package/table_paired_bootstrap_ci.csv",
                }
            )

    alignment_rows = []
    alignment_paths = []
    for seed in (7, 13, 23):
        path = Path(
            f"tables/synthetic_gate_fusion_aggregate_sota_seed{seed}/table_alignment_status.csv"
        )
        alignment_paths.append(path)
        alignment_rows.extend(read_csv(path))
    alignment_complete = all(path.exists() for path in alignment_paths)
    rows.append(
        {
            "requirement": "same_path_baseline_alignment",
            "status": "pass" if alignment_complete and not alignment_rows else "fail",
            "value": f"files={sum(path.exists() for path in alignment_paths)}/3;issues={len(alignment_rows)}",
            "threshold": "3/3 files and 0 issues",
            "evidence": "tables/synthetic_gate_fusion_aggregate_sota_seed*/table_alignment_status.csv",
        }
    )

    modules = read_csv(
        Path(
            "tables/feature_first_fusion_aggregate_paper_package/"
            "table_module_ablation_mpdd.csv"
        )
    )
    repair_summary = read_json(
        Path(
            "tables/feature_first_fusion_aggregate_paper_package/"
            "repair_quality_summary.json"
        )
    )
    repair_quality = repair_summary.get("quality", {}) or {}
    repair_sdr = repair_summary.get("sdr_gt", {}) or {}
    module_summary = read_json(
        Path(
            "tables/feature_first_fusion_aggregate_paper_package/"
            "module_evidence_summary.json"
        )
    )
    repair_report_complete = (
        repair_summary.get("categories") == 33
        and repair_summary.get("datasets") == ["mvtec15", "visa", "mpdd"]
        and repair_summary.get("missing_categories") == []
        and repair_summary.get("evidence_scope")
        == "independent_module_run_128_not_main_detector"
        and int(repair_quality.get("image_count", 0)) > 0
        and int(repair_sdr.get("roi_count", 0)) > 0
        and np.isfinite(as_float(repair_quality.get("anomaly_mean_ssim")))
        and np.isfinite(as_float(repair_sdr.get("sdr_gt_fraction_spearman")))
        and repair_summary.get("claim_decision")
        in {
            "retain_gt_aligned_repair_diagnostic",
            "downgrade_to_visualization_only",
        }
    )
    rows.append(
        {
            "requirement": "cross_dataset_repair_quality_and_sdr_gt_report",
            "status": "pass" if repair_report_complete else "fail",
            "value": (
                f"categories={repair_summary.get('categories', 0)}/33;"
                f"images={repair_quality.get('image_count', 0)};"
                f"rois={repair_sdr.get('roi_count', 0)};"
                f"sdr_gt_spearman="
                f"{as_float(repair_sdr.get('sdr_gt_fraction_spearman')):.6f};"
                f"decision={repair_summary.get('claim_decision', '')}"
            ),
            "threshold": (
                "33 categories, 3 datasets, image/ROI evidence, finite repair "
                "quality and SDR-GT metrics, explicit claim decision"
            ),
            "evidence": (
                "tables/feature_first_fusion_aggregate_paper_package/"
                "repair_quality_summary.json"
            ),
        }
    )
    hn = next((row for row in modules if row.get("module") == "HN-SEV"), {})
    crv = next((row for row in modules if row.get("module") == "CRV"), {})
    lc = [row for row in modules if row.get("module") == "LC-RDS"]
    rows.append(
        {
            "requirement": "hn_sev_independent_positive_evidence",
            "status": (
                "pass"
                if (
                    module_summary.get("coverage_complete")
                    and module_summary.get("hn_sev", {}).get(
                        "positive_repeatable_evidence"
                    )
                )
                else "fail"
            ),
            "value": (
                f"mean_delta_fprr="
                f"{as_float(module_summary.get('hn_sev', {}).get('mean_delta_fprr')):.6f};"
                f"lower={module_summary.get('hn_sev', {}).get('categories_with_lower_fprr', 0)}/33"
            ),
            "threshold": "33/33 categories covered and all have FPRR delta<0",
            "evidence": (
                "tables/feature_first_fusion_aggregate_paper_package/"
                "module_evidence_summary.json"
            ),
        }
    )
    crv_detection_positive = (
        as_float(crv.get("mean_delta_pixel_ap")) > 0
        or as_float(crv.get("mean_delta_dice")) > 0
    )
    crv_module_summary = module_summary.get("crv", {}) or {}
    crv_diagnostic_positive = (
        module_summary.get("coverage_complete")
        and as_float(crv_module_summary.get("mean_delta_sdr")) > 0
        and int(crv_module_summary.get("categories_with_positive_sdr", 0)) == 33
    )
    crv_gt_aligned = bool(
        repair_summary.get("correlation_supports_gt_alignment")
    )
    crv_explicitly_downgraded = (
        repair_summary.get("claim_decision")
        == "downgrade_to_visualization_only"
    )
    rows.append(
        {
            "requirement": "crv_detection_gain_or_explanation_downgrade",
            "status": (
                "pass"
                if (
                    crv_detection_positive
                    or (crv_diagnostic_positive and crv_gt_aligned)
                    or (repair_report_complete and crv_explicitly_downgraded)
                )
                else "fail"
            ),
            "value": (
                f"detection={crv_detection_positive};"
                f"diagnostic={crv_diagnostic_positive};"
                f"gt_aligned={crv_gt_aligned};"
                f"decision={repair_summary.get('claim_decision', '')}"
            ),
            "threshold": (
                "detection gain, GT-aligned diagnostic evidence, or complete "
                "cross-dataset evidence with explicit visualization-only downgrade"
            ),
            "evidence": (
                "tables/feature_first_fusion_aggregate_paper_package/"
                "module_evidence_summary.json;"
                "tables/feature_first_fusion_aggregate_paper_package/"
                "repair_quality_summary.json"
            ),
        }
    )
    lc_summary = module_summary.get("lc_rds", {}) or {}
    lc_pass = bool(
        module_summary.get("coverage_complete")
        and lc_summary.get("positive_repeatable_high_budget_evidence")
    )
    rows.append(
        {
            "requirement": "lc_rds_independent_latency_evidence",
            "status": "pass" if lc_pass else "fail",
            "value": (
                f"fixed25={as_float((lc_summary.get('fixed25') or {}).get('mean_delta_latency_ms')):.6f},"
                f"{(lc_summary.get('fixed25') or {}).get('categories_faster', 0)}/33;"
                f"rule={as_float((lc_summary.get('rule') or {}).get('mean_delta_latency_ms')):.6f},"
                f"{(lc_summary.get('rule') or {}).get('categories_faster', 0)}/33;"
                f"fixed10={as_float((lc_summary.get('fixed10') or {}).get('mean_delta_latency_ms')):.6f},"
                f"{(lc_summary.get('fixed10') or {}).get('categories_faster', 0)}/33"
            ),
            "threshold": (
                "33/33 categories covered; faster than fixed25 and rule in "
                "33/33 categories; fixed10 limitation reported"
            ),
            "evidence": (
                "tables/feature_first_fusion_aggregate_paper_package/"
                "module_evidence_summary.json"
            ),
        }
    )

    few_shot = read_json(Path("tables/fewshot_mvtec/protocol.json"))
    rows.append(
        {
            "requirement": "few_shot_8_16_32_three_seeds",
            "status": "pass" if few_shot.get("runs_completed") == 135 else "fail",
            "value": few_shot.get("runs_completed", 0),
            "threshold": 135,
            "evidence": "tables/fewshot_mvtec",
        }
    )

    discovery = read_json(
        Path("tables/mvtec_ad2_feature_first_readiness/discovery.json")
    )
    public_protocol = read_json(Path("tables/mvtec_ad2_feature_first/protocol.json"))
    submission_protocol = read_json(
        Path(
            "submissions/mvtec_ad2_seed7_model256_metadata/"
            "submission_protocol.json"
        )
    )
    public_complete = public_protocol.get("runs_completed") == 24
    checker_status = (
        submission_protocol.get("official_checker", {}) or {}
    ).get("status")
    archive_value = submission_protocol.get("archive")
    archive_complete = bool(archive_value) and Path(str(archive_value)).is_file()
    submission_complete = (
        submission_protocol.get("files") == 4090
        and bool_value(submission_protocol.get("full_official_submission"))
        and checker_status == "passed"
        and archive_complete
    )
    local_ad2_archived = (
        bool(discovery.get("ready"))
        and public_complete
        and submission_complete
    )
    rows.append(
        {
            "requirement": "optional_mvtec_ad2_local_artifacts_archived",
            "status": "pass" if local_ad2_archived else "fail",
            "value": (
                f"discovered={bool(discovery.get('ready'))};"
                f"public={public_protocol.get('runs_completed', 0)}/24;"
                f"submission_files={submission_protocol.get('files', 0)};"
                f"checker={checker_status};"
                f"archive={archive_complete};"
                "main_paper_gate=false"
            ),
            "threshold": (
                "optional asset only: dataset discovered, 24 public runs, "
                "4090 private samples, checker pass, archive; no server result"
            ),
            "evidence": (
                "tables/mvtec_ad2_feature_first_readiness/discovery.json;"
                "tables/mvtec_ad2_feature_first/protocol.json;"
                "submissions/mvtec_ad2_seed7_model256_metadata/"
                "submission_protocol.json"
            ),
        }
    )

    fields = ["requirement", "status", "value", "threshold", "evidence"]
    with (out_dir / "table_acceptance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ["pass", "fail", "blocked", "pending_run"]
    }
    payload = {
        "requirements": len(rows),
        "counts": counts,
        "complete": counts["fail"] == 0 and counts["blocked"] == 0 and counts["pending_run"] == 0,
        "failed": [row for row in rows if row["status"] == "fail"],
        "blocked": [row for row in rows if row["status"] == "blocked"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
