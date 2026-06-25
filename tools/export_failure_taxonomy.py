"""Export a post-hoc taxonomy for retained weak categories.

This audit does not select methods or thresholds. It reads frozen held-out
results and records why the remaining weak categories are still risky, what
label-free candidate family was selected, and which follow-up would be needed
to turn the limitation into a performance fix.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]
DEFAULT_SPECS = [
    {
        "dataset": "mvtec15",
        "category": "grid",
        "family": "periodic texture",
        "failure_mode": "structured-grid defects can become image-score misses even when pixel localization improves",
        "recommended_action": "frequency/spatial calibration must stay inside the label-free gate",
    },
    {
        "dataset": "mvtec15",
        "category": "screw",
        "family": "small high-frequency object",
        "failure_mode": "tiny defects and reflective edges leave low pixel AP/Dice margin",
        "recommended_action": "native-resolution ROI refinement or screw-specific highpass candidates",
    },
    {
        "dataset": "mvtec15",
        "category": "pill",
        "family": "color and local contrast",
        "failure_mode": "small color defects can be missed by the frozen threshold",
        "recommended_action": "color/local-contrast candidates evaluated by synthetic-normal evidence",
    },
    {
        "dataset": "mvtec15",
        "category": "capsule",
        "family": "color and local contrast",
        "failure_mode": "capsule defects remain sensitive to threshold and local contrast",
        "recommended_action": "capsule color branch or local-contrast normalization inside the same gate",
    },
    {
        "dataset": "mvtec15",
        "category": "hazelnut",
        "family": "shape and surface cut",
        "failure_mode": "large cuts can be detected but mask quality still trails strong baselines",
        "recommended_action": "boundary-aware postprocess or high-resolution ROI candidate, selected label-free",
    },
    {
        "dataset": "mvtec15",
        "category": "transistor",
        "family": "multi-part object layout",
        "failure_mode": "normal multi-object layout creates feature ambiguity and AUPRO risk",
        "recommended_action": "layout-aware normality features without held-out GT selection",
    },
    {
        "dataset": "visa",
        "category": "capsules",
        "family": "tiny low-contrast defects",
        "failure_mode": "small capsule defects remain a localization failure despite high-resolution selection",
        "recommended_action": "native-resolution ROI candidate and local color contrast normalization",
    },
    {
        "dataset": "visa",
        "category": "fryum",
        "family": "texture and edge clutter",
        "failure_mode": "irregular texture and edges can reduce mask quality against strong baselines",
        "recommended_action": "texture-aware highpass candidate kept under label-free gate selection",
    },
    {
        "dataset": "mpdd",
        "category": "bracket_white",
        "family": "low-contrast white object",
        "failure_mode": "white-on-bright defects are low-contrast and remain low Dice/Pixel AP",
        "recommended_action": "local contrast normalization or native-resolution ROI branch",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset", "")), str(row.get("category", ""))


def best_baseline(
    rows: list[dict[str, str]],
    dataset: str,
    category: str,
    metric: str,
) -> tuple[str, float]:
    candidates = [
        row
        for row in rows
        if row.get("dataset") == dataset
        and row.get("category") == category
        and row.get("source") == "baseline"
    ]
    if not candidates:
        return "", float("nan")
    best = max(candidates, key=lambda row: as_float(row.get(metric)))
    return str(best.get("method", "")), as_float(best.get(metric))


def severity(row: dict[str, Any]) -> str:
    image_auroc = as_float(row.get("image_auroc"))
    pixel_ap = as_float(row.get("pixel_ap"))
    dice = as_float(row.get("dice"))
    dice_gap = -as_float(row.get("dice_delta_vs_best"))
    ap_gap = -as_float(row.get("pixel_ap_delta_vs_best"))
    if dice < 0.10 or pixel_ap < 0.05 or image_auroc < 0.70:
        return "critical"
    if dice < 0.25 or pixel_ap < 0.15 or dice_gap > 0.20 or ap_gap > 0.20:
        return "major"
    return "moderate"


def panel_case_categories(panel_summary: dict[str, Any]) -> set[str]:
    return {
        str(case.get("category"))
        for case in panel_summary.get("cases", []) or []
        if isinstance(case, dict) and case.get("category")
    }


def build_taxonomy_rows(
    table_rows: list[dict[str, str]],
    specs: list[dict[str, str]] | None = None,
    panel_categories: set[str] | None = None,
) -> list[dict[str, Any]]:
    specs = specs or DEFAULT_SPECS
    panel_categories = panel_categories or set()
    ours = {
        key(row): row
        for row in table_rows
        if row.get("source") == "ours_selected"
    }
    taxonomy_rows: list[dict[str, Any]] = []
    for spec in specs:
        dataset = spec["dataset"]
        category = spec["category"]
        row = ours.get((dataset, category), {})
        output: dict[str, Any] = {
            "dataset": dataset,
            "category": category,
            "family": spec["family"],
            "failure_mode": spec["failure_mode"],
            "recommended_action": spec["recommended_action"],
            "covered_by_failure_panel": category in panel_categories,
            "selected_candidate": row.get("selected_candidate", ""),
            "run": row.get("run", ""),
        }
        for metric in METRICS:
            output[metric] = as_float(row.get(metric))
        for metric in ("dice", "pixel_ap", "aupro", "image_auroc"):
            method, value = best_baseline(table_rows, dataset, category, metric)
            output[f"best_baseline_{metric}_method"] = method
            output[f"best_baseline_{metric}"] = value
            output[f"{metric}_delta_vs_best"] = (
                output[metric] - value
                if method
                else float("nan")
            )
        output["taxonomy_status"] = "covered" if row else "missing_result"
        output["severity"] = severity(output) if row else "missing"
        taxonomy_rows.append(output)
    return taxonomy_rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = len(rows)
    covered = sum(1 for row in rows if row["taxonomy_status"] == "covered")
    severities: dict[str, int] = {}
    for row in rows:
        severities[row["severity"]] = severities.get(row["severity"], 0) + 1
    return {
        "schema": "lite-seer-ad-failure-taxonomy-v1",
        "taxonomy_role": "post_hoc_failure_analysis_only",
        "required_weak_categories": required,
        "covered_weak_categories": covered,
        "all_required_weak_categories_covered": covered == required,
        "release_gate_passed": covered == required,
        "release_gate_reason": (
            "All retained weak categories are covered by the post-hoc failure taxonomy."
            if covered == required
            else "Some retained weak categories are missing frozen held-out result rows."
        ),
        "severity_counts": severities,
        "uses_real_anomaly_labels_for_method_selection": False,
        "uses_real_anomaly_masks_for_method_selection": False,
        "uses_real_anomaly_masks_for_taxonomy": True,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Weak Category Failure Taxonomy",
        "",
        f"- Covered weak categories: `{summary['covered_weak_categories']}/{summary['required_weak_categories']}`",
        f"- Role: `{summary['taxonomy_role']}`",
        f"- Method selection uses real anomaly labels: `{summary['uses_real_anomaly_labels_for_method_selection']}`",
        f"- Method selection uses real anomaly masks: `{summary['uses_real_anomaly_masks_for_method_selection']}`",
        "",
        "| Dataset | Category | Severity | Selected candidate | Pixel AP | Dice | Failure mode | Next action |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {category} | {severity} | {selected_candidate} | "
            "{pixel_ap:.4f} | {dice:.4f} | {failure_mode} | {recommended_action} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    root: Path,
    out_dir: Path,
    sota_table: Path,
    failure_panel_summary: Path,
) -> dict[str, Any]:
    table_rows = read_csv(root / sota_table)
    panel_summary = read_json(root / failure_panel_summary)
    rows = build_taxonomy_rows(
        table_rows,
        panel_categories=panel_case_categories(panel_summary),
    )
    summary = build_summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_failure_taxonomy.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "analysis.md", summary, rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--sota-table",
        type=Path,
        default=Path("tables/heldout_sota_comparison_post/table_heldout_sota_cross_dataset.csv"),
    )
    parser.add_argument(
        "--failure-panel-summary",
        type=Path,
        default=Path("tables/failure_case_panel_mvtec15/summary.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("tables/failure_taxonomy"))
    args = parser.parse_args()
    summary = write_outputs(args.root, args.out_dir, args.sota_table, args.failure_panel_summary)
    print(
        f"Wrote failure taxonomy to {args.out_dir} "
        f"(release_gate_passed={summary['release_gate_passed']})"
    )


if __name__ == "__main__":
    main()
