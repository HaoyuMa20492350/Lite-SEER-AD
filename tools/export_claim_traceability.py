"""Export claim-to-artifact traceability for paper-facing numbers.

The public-reproduction gate requires each retained paper number to point back
to an auditable artifact, regeneration command, protocol/seed boundary, and a
SHA256 entry in the release manifest. This exporter validates a curated claim
registry against the current JSON artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    manuscript_anchor: str
    evidence_path: str
    regenerate_command: str
    protocol_or_seed: str
    extractor: Callable[[Path], tuple[bool, str, str]]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_entries(root: Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(root / "artifacts/manifest.json")
    return {
        str(entry.get("path", "")).replace("\\", "/"): entry
        for entry in manifest.get("files", [])
    }


def rounded(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "nan"


def get_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def paired_record(summary: dict[str, Any], method: str, metric: str) -> dict[str, Any]:
    for row in (summary.get("paired_inference", {}) or {}).get("records", []):
        if row.get("method") == method and row.get("metric") == metric:
            return row
    return {}


def main_delta(metric: str) -> Callable[[Path], tuple[bool, str, str]]:
    def _extract(path: Path) -> tuple[bool, str, str]:
        data = read_json(path)
        rows = data.get("paired_bootstrap_positive_ci", []) or []
        row = next(
            (
                item
                for item in rows
                if item.get("dataset") == "all" and item.get("metric") == metric
            ),
            {},
        )
        ok = bool(row) and row.get("categories") == 33 and row.get("ci95_low", 0.0) > 0.0
        value = (
            f"delta={rounded(row.get('mean_delta'))}; "
            f"ci=[{rounded(row.get('ci95_low'))}, {rounded(row.get('ci95_high'))}]; "
            f"wins={row.get('wins')}/33"
        )
        return ok, value, "33-category paired BCa/sign-test row"

    return _extract


def threshold_bundle(path: Path) -> tuple[bool, str, str]:
    data = read_json(path)
    policies = data.get("policies", []) or []
    max_fpr = max((float(row.get("max_normal_pixel_fpr", 1.0)) for row in policies), default=1.0)
    ok = (
        data.get("schema") == "synthetic_normal_fixed_threshold_v1_bundle"
        and data.get("policy_count") == 33
        and data.get("evaluated_run_count") == 99
        and data.get("uses_real_anomaly_labels") is False
        and data.get("uses_real_anomaly_masks") is False
        and max_fpr <= 0.005
    )
    value = f"policies={data.get('policy_count')}; runs={data.get('evaluated_run_count')}; max_fpr={max_fpr:.6f}"
    return ok, value, "fixed threshold policy bundle"


def policy_stability(path: Path) -> tuple[bool, str, str]:
    data = read_json(path)
    ok = data.get("category_selection_agreement") == 1.0 and data.get("seeds") == [7, 13, 23]
    value = f"agreement={rounded(data.get('category_selection_agreement'), 4)}; seeds={data.get('seeds')}"
    return ok, value, "cross-seed synthetic policy stability"


def hn_sev(path: Path) -> tuple[bool, str, str]:
    data = read_json(path).get("hn_sev", {}) or {}
    ok = data.get("categories_with_lower_fprr") == 33 and data.get("mean_delta_fprr", 1.0) < 0.0
    value = f"mean_delta_fprr={rounded(data.get('mean_delta_fprr'))}; wins={data.get('categories_with_lower_fprr')}/33"
    return ok, value, "module evidence summary"


def lc_rds(path: Path) -> tuple[bool, str, str]:
    data = read_json(path).get("lc_rds", {}) or {}
    fixed25 = data.get("fixed25", {}) or {}
    rule = data.get("rule", {}) or {}
    fixed10 = data.get("fixed10", {}) or {}
    ok = fixed25.get("categories_faster") == 33 and rule.get("categories_faster") == 33
    value = (
        f"fixed25_delta_ms={rounded(fixed25.get('mean_delta_latency_ms'), 2)}; "
        f"rule_delta_ms={rounded(rule.get('mean_delta_latency_ms'), 2)}; "
        f"fixed10_wins={fixed10.get('categories_faster')}/33"
    )
    return ok, value, "module evidence summary with fixed10 limitation"


def crv_negative(path: Path) -> tuple[bool, str, str]:
    data = read_json(path)
    sdr = (data.get("sdr_gt", {}) or {}).get("sdr_gt_fraction_spearman")
    ok = data.get("claim_decision") == "downgrade_to_visualization_only" and float(sdr) < 0.0
    value = f"sdr_gt_spearman={rounded(sdr)}; decision={data.get('claim_decision')}"
    return ok, value, "repair quality and SDR-GT negative finding"


def diffusionad_compute(path: Path) -> tuple[bool, str, str]:
    data = read_json(path)
    ok = data.get("optimizer_steps") == 663000 and data.get("categories") == 15
    value = (
        f"steps={data.get('optimizer_steps')}; "
        f"gpu_hours={rounded(data.get('estimated_total_gpu_hours'), 1)}; "
        f"categories={data.get('categories')}"
    )
    return ok, value, "measured compute extrapolation"


def image_agg_negative(path: Path) -> tuple[bool, str, str]:
    data = read_json(path)
    ok = data.get("adoption_decision") == "retain_current_top5" and data.get("mean_delta", 1.0) < 0.0
    value = (
        f"current={rounded(data.get('current_mean_image_auroc'))}; "
        f"selected={rounded(data.get('selected_mean_image_auroc'))}; "
        f"delta={rounded(data.get('mean_delta'))}"
    )
    return ok, value, "11-mode aggregation negative audit"


def padim_pixel_auroc(path: Path) -> tuple[bool, str, str]:
    data = read_json(path)
    row = paired_record(data, "padim", "pixel_auroc")
    ok = bool(row) and row.get("ci95_low", -1.0) > 0.0 and row.get("sign_test_p_holm_within_metric", 1.0) < 0.05
    value = (
        f"delta={rounded(row.get('mean_delta'))}; "
        f"ci=[{rounded(row.get('ci95_low'))}, {rounded(row.get('ci95_high'))}]; "
        f"holm_p={row.get('sign_test_p_holm_within_metric')}"
    )
    return ok, value, "paired external baseline inference"


def prediction_manifest(path: Path) -> tuple[bool, str, str]:
    data = read_json(path)
    ok = (
        data.get("all_predictions_present") is True
        and data.get("all_threshold_policies_present") is True
        and data.get("entry_count") == 99
    )
    value = f"entries={data.get('entry_count')}; predictions={data.get('all_predictions_present')}; thresholds={data.get('all_threshold_policies_present')}"
    return ok, value, "selected prediction arrays and threshold policies"


CLAIMS = [
    ClaimSpec(
        "threshold_label_free_33x99",
        "0.5% normal-pixel FPR cap; 33 policies; 99 held-out runs",
        "artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json",
        "python scripts/release/export_fixed_threshold_bundle.py",
        "selection seeds 7/13/23; no real anomaly labels or masks",
        threshold_bundle,
    ),
    ClaimSpec(
        "policy_stability_100_percent",
        "Candidate agreement is 100% across held-out seeds",
        "tables/feature_first_fusion_aggregate_paper_package/summary.json",
        "python tools/export_feature_first_paper_package.py",
        "seeds 7/13/23",
        policy_stability,
    ),
    ClaimSpec(
        "main_aupro_delta",
        "AUPRO improves by 0.0584 with positive 95% CI",
        "tables/feature_first_fusion_aggregate_paper_package/summary.json",
        "python tools/export_feature_first_paper_package.py",
        "33 categories; paired BCa bootstrap",
        main_delta("aupro"),
    ),
    ClaimSpec(
        "main_pixel_ap_delta",
        "Pixel AP improves by 0.0618 with positive 95% CI",
        "tables/feature_first_fusion_aggregate_paper_package/summary.json",
        "python tools/export_feature_first_paper_package.py",
        "33 categories; paired BCa bootstrap",
        main_delta("pixel_ap"),
    ),
    ClaimSpec(
        "external_padim_pixel_auroc",
        "PaDiM Pixel AUROC robust paired advantage",
        "tables/external_baseline_comparison/summary.json",
        "python tools/export_external_baseline_comparison.py",
        "15 MVTec categories; Holm-adjusted sign test",
        padim_pixel_auroc,
    ),
    ClaimSpec(
        "hn_sev_fprr_33_categories",
        "HN-SEV lowers FPRR in all 33/33 categories",
        "tables/feature_first_fusion_aggregate_paper_package/module_evidence_summary.json",
        "python tools/export_feature_first_paper_package.py",
        "independent module evidence; not detector AP/Dice",
        hn_sev,
    ),
    ClaimSpec(
        "lc_rds_latency_boundaries",
        "LC-RDS faster than fixed25/rule; not universal fixed10 fastest",
        "tables/feature_first_fusion_aggregate_paper_package/module_evidence_summary.json",
        "python tools/export_feature_first_paper_package.py",
        "33 categories; fixed10 limitation retained",
        lc_rds,
    ),
    ClaimSpec(
        "crv_negative_alignment",
        "CRV downgraded because SDR-GT Spearman is negative",
        "tables/feature_first_fusion_aggregate_paper_package/repair_quality_summary.json",
        "python tools/export_feature_first_paper_package.py",
        "CRV visualization/post-hoc audit only",
        crv_negative,
    ),
    ClaimSpec(
        "diffusionad_compute_plan",
        "DiffusionAD full requires 663,000 optimizer steps and about 3,801 GPU hours",
        "tables/diffusionad_compute_plan/summary.json",
        "python tools/export_diffusionad_compute_plan.py",
        "15 categories; 3000 epochs/category author configuration",
        diffusionad_compute,
    ),
    ClaimSpec(
        "image_aggregation_negative_audit",
        "11-mode image aggregation search is retained as a negative audit",
        "tables/image_score_aggregation_mvtec15/summary.json",
        "python tools/select_image_score_aggregation.py",
        "normal/synthetic selection; held-out labels audit only",
        image_agg_negative,
    ),
    ClaimSpec(
        "prediction_arrays_99_runs",
        "Selected prediction arrays and threshold policies cover all 99 main runs",
        "artifacts/predictions_manifest.json",
        "python scripts/release/build_prediction_array_manifest.py --out artifacts/predictions_manifest.json",
        "33 categories x seeds 7/13/23",
        prediction_manifest,
    ),
]


def build_rows(root: Path) -> list[dict[str, Any]]:
    manifest = manifest_entries(root)
    commit = get_commit(root)
    rows = []
    for spec in CLAIMS:
        path = root / spec.evidence_path
        ok, value, detail = spec.extractor(path)
        manifest_entry = manifest.get(spec.evidence_path)
        in_manifest = manifest_entry is not None
        rows.append(
            {
                "claim_id": spec.claim_id,
                "status": "pass" if ok and in_manifest else "fail",
                "value": value,
                "manuscript_anchor": spec.manuscript_anchor,
                "evidence_path": spec.evidence_path,
                "evidence_sha256": manifest_entry.get("sha256", "") if manifest_entry else "",
                "in_manifest": in_manifest,
                "regenerate_command": spec.regenerate_command,
                "protocol_or_seed": spec.protocol_or_seed,
                "commit": commit,
                "detail": detail,
            }
        )
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row["claim_id"] for row in rows if row["status"] != "pass"]
    return {
        "schema": "lite-seer-ad-claim-traceability-v1",
        "claims": len(rows),
        "claims_passed": len(rows) - len(failed),
        "local_traceability_ready": not failed,
        "release_gate_passed": not failed,
        "release_gate_reason": (
            "Every curated paper-facing numeric claim resolves to an evidence artifact with a SHA256 manifest entry."
            if not failed
            else "Some curated paper-facing numeric claims are missing evidence or SHA256 manifest entries."
        ),
        "failed_claims": failed,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "claim_id",
        "status",
        "value",
        "manuscript_anchor",
        "evidence_path",
        "evidence_sha256",
        "in_manifest",
        "regenerate_command",
        "protocol_or_seed",
        "commit",
        "detail",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Lite-SEER-AD Claim Traceability",
        "",
        f"- Claims: `{summary['claims']}`",
        f"- Passed: `{summary['claims_passed']}`",
        f"- Local traceability ready: `{summary['local_traceability_ready']}`",
        "",
        "| Claim | Status | Value | Evidence |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['claim_id']}` | {row['status']} | {row['value']} | `{row['evidence_path']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(root: Path, out_dir: Path) -> dict[str, Any]:
    rows = build_rows(root)
    summary = build_summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_claim_traceability.csv", rows)
    (out_dir / "rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "claim_traceability.md", rows, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("tables/claim_traceability"))
    args = parser.parse_args()
    summary = write_outputs(args.root.resolve(), args.out_dir)
    print(
        f"Wrote claim traceability for {summary['claims']} claims to {args.out_dir} "
        f"(local_traceability_ready={summary['local_traceability_ready']})"
    )


if __name__ == "__main__":
    main()
