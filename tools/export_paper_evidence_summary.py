from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]
DEFAULT_OURS = "lite_seer_ad_crv035"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a paper-facing evidence summary from current experiment tables.")
    p.add_argument("--comparison-dir", default="tables/mvtec15_comparison")
    p.add_argument("--ours-dir", default="tables/mvtec15_ours")
    p.add_argument("--baseline-dir", default="tables/mvtec15_baselines")
    p.add_argument("--next-mvtec5-dir", default="tables/next_mvtec5")
    p.add_argument("--out", default="docs/paper_evidence_summary.md")
    p.add_argument("--ours-method", default=DEFAULT_OURS)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def group_means(rows: list[dict[str, str]], key: str, fields: list[str]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = str(row.get(key, ""))
        if not group:
            continue
        for field in fields:
            value = fnum(row.get(field))
            if value == value:
                values[group][field].append(value)
    return {
        group: {field: mean(vals) for field, vals in by_field.items() if vals}
        for group, by_field in values.items()
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def dataset_categories(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def method_summary(main_rows: list[dict[str, str]]) -> str:
    means = group_means(main_rows, "method", METRICS)
    rows = []
    for method, vals in sorted(means.items(), key=lambda item: item[1].get("pixel_ap", -1), reverse=True):
        rows.append([method, *(fmt(vals.get(metric)) for metric in METRICS)])
    return markdown_table(["method", *METRICS], rows)


def metric_means(rows: list[dict[str, str]], fields: list[str]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for field in fields:
            value = fnum(row.get(field))
            if value == value:
                values[field].append(value)
    return {field: mean(vals) for field, vals in values.items() if vals}


def ours_vs_best_baseline(main_rows: list[dict[str, str]], ours_method: str) -> tuple[str, dict[str, Any]]:
    by_category: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in main_rows:
        category = row.get("category")
        method = row.get("method")
        if category and method:
            by_category[category][method] = row

    rows = []
    wins = 0
    deltas: list[float] = []
    excluded = {ours_method, "lite_seer_ad_crv05"}
    for category in sorted(by_category):
        methods = by_category[category]
        ours = methods.get(ours_method)
        if not ours:
            continue
        baseline_items = [(m, fnum(r.get("pixel_ap"))) for m, r in methods.items() if m not in excluded]
        baseline_items = [(m, v) for m, v in baseline_items if v == v]
        if not baseline_items:
            continue
        best_method, best_value = max(baseline_items, key=lambda item: item[1])
        ours_value = fnum(ours.get("pixel_ap"))
        delta = ours_value - best_value
        wins += int(delta > 0)
        deltas.append(delta)
        rows.append([category, fmt(ours_value), best_method, fmt(best_value), fmt(delta)])
    summary = {"categories": len(rows), "wins": wins, "mean_delta": mean(deltas) if deltas else None}
    return markdown_table(["category", "ours_pixel_ap", "best_baseline", "best_pixel_ap", "delta"], rows), summary


def write_summary(args: argparse.Namespace) -> None:
    comparison_dir = Path(args.comparison_dir)
    ours_dir = Path(args.ours_dir)
    baseline_dir = Path(args.baseline_dir)
    next_mvtec5_dir = Path(args.next_mvtec5_dir)

    main_rows = read_csv(comparison_dir / "table_main_ours_vs_baselines.csv")
    eff_rows = read_csv(comparison_dir / "table_efficiency_ours_vs_baselines.csv")
    coverage = read_json(baseline_dir / "baseline_coverage.json")
    mvtec15_gate = read_json(ours_dir / "gate_summary.json")
    mvtec5_gate = read_json(next_mvtec5_dir / "gate_summary.json")
    crv_decision = read_json(comparison_dir / "crv_weight_decision.json")
    crv5_decision = read_json(next_mvtec5_dir / "crv_weight_decision.json")
    visa_smoke_report = read_json(Path("runs/visa_smoke_report.json"))
    mpdd_smoke_report = read_json(Path("runs/mpdd_smoke_report.json"))
    visa_mini_report = read_json(Path("runs/visa_mini_report.json"))
    visa_mini_main = read_csv(Path("tables/visa_mini/table_main_visa.csv"))
    visa_mini_gate = read_json(Path("tables/visa_mini/gate_summary.json"))
    visa_gate_report = read_json(Path("runs/visa_gate_report.json"))
    visa_gate_main = read_csv(Path("tables/visa_gate/table_main_visa.csv"))
    visa_gate_gate = read_json(Path("tables/visa_gate/gate_summary.json"))
    mpdd_mini_report = read_json(Path("runs/mpdd_mini_report.json"))
    mpdd_mini_main = read_csv(Path("tables/mpdd_mini/table_main_mpdd.csv"))
    mpdd_mini_gate = read_json(Path("tables/mpdd_mini/gate_summary.json"))
    mpdd_gate_report = read_json(Path("runs/mpdd_gate_report.json"))
    mpdd_gate_main = read_csv(Path("tables/mpdd_gate/table_main_mpdd.csv"))
    mpdd_gate_gate = read_json(Path("tables/mpdd_gate/gate_summary.json"))
    visa_baseline_smoke_report = read_json(Path("runs/visa_baseline_smoke_report.json"))
    mpdd_baseline_smoke_report = read_json(Path("runs/mpdd_baseline_smoke_report.json"))
    visa_baseline_smoke_coverage = read_json(Path("tables/visa_baseline_smoke/baseline_coverage.json"))
    mpdd_baseline_smoke_coverage = read_json(Path("tables/mpdd_baseline_smoke/baseline_coverage.json"))
    visa_baseline_report = read_json(Path("runs/visa_baseline_report.json"))
    mpdd_baseline_report = read_json(Path("runs/mpdd_baseline_report.json"))
    visa_baseline_coverage = read_json(Path("tables/visa_baselines/baseline_coverage.json"))
    mpdd_baseline_coverage = read_json(Path("tables/mpdd_baselines/baseline_coverage.json"))
    paper_package_status = read_csv(Path("tables/paper_package/table_run_status.csv"))
    paper_package_coverage = read_csv(Path("tables/paper_package/table_baseline_coverage.csv"))

    method_table = method_summary(main_rows)
    win_table, win_summary = ours_vs_best_baseline(main_rows, args.ours_method)
    eff_means = group_means(eff_rows, "method", ["latency_ms_mean", "fps", "nfe_mean"])
    eff_rows_md = []
    for method, vals in sorted(eff_means.items(), key=lambda item: item[1].get("fps", 0), reverse=True):
        eff_rows_md.append([method, fmt(vals.get("latency_ms_mean"), 2), fmt(vals.get("fps"), 2), fmt(vals.get("nfe_mean"), 2)])
    eff_table = markdown_table(["method", "latency_ms", "fps", "nfe"], eff_rows_md)

    visa_root = Path("SEER-AD-dataset/VisA")
    mpdd_root = Path("SEER-AD-dataset/MPDD/official/MPDD/MPDD")
    visa_categories = [c for c in dataset_categories(visa_root) if c != "split_csv"]
    mpdd_categories = dataset_categories(mpdd_root)
    visa_smoke_ok = bool(visa_smoke_report) and not visa_smoke_report.get("failures") and all(
        bool(row.get("ok")) for row in visa_smoke_report.get("schema", [])
    )
    mpdd_smoke_ok = bool(mpdd_smoke_report) and not mpdd_smoke_report.get("failures") and all(
        bool(row.get("ok")) for row in mpdd_smoke_report.get("schema", [])
    )
    visa_mini_ok = bool(visa_mini_report) and not visa_mini_report.get("failures") and all(
        bool(row.get("ok")) for row in visa_mini_report.get("schema", [])
    )
    visa_gate_ok = bool(visa_gate_report) and not visa_gate_report.get("failures") and all(
        bool(row.get("ok")) for row in visa_gate_report.get("schema", [])
    )
    mpdd_mini_ok = bool(mpdd_mini_report) and not mpdd_mini_report.get("failures") and all(
        bool(row.get("ok")) for row in mpdd_mini_report.get("schema", [])
    )
    mpdd_gate_ok = bool(mpdd_gate_report) and not mpdd_gate_report.get("failures") and all(
        bool(row.get("ok")) for row in mpdd_gate_report.get("schema", [])
    )
    visa_baseline_smoke_ok = bool(visa_baseline_smoke_report) and not visa_baseline_smoke_report.get("failures") and all(
        bool(row.get("ok")) for row in visa_baseline_smoke_report.get("schema", [])
    )
    mpdd_baseline_smoke_ok = bool(mpdd_baseline_smoke_report) and not mpdd_baseline_smoke_report.get("failures") and all(
        bool(row.get("ok")) for row in mpdd_baseline_smoke_report.get("schema", [])
    )
    visa_baseline_ok = bool(visa_baseline_report) and not visa_baseline_report.get("failures") and all(
        bool(row.get("ok")) for row in visa_baseline_report.get("schema", [])
    )
    mpdd_baseline_ok = bool(mpdd_baseline_report) and not mpdd_baseline_report.get("failures") and all(
        bool(row.get("ok")) for row in mpdd_baseline_report.get("schema", [])
    )
    paper_package_ok = bool(paper_package_status) and all(
        str(row.get("schema_ok", "")).lower() == "true" and int(float(row.get("failures", 1))) == 0
        for row in paper_package_status
    )
    paper_coverage_ok = bool(paper_package_coverage) and all(
        str(row.get("complete", "")).lower() == "true" and int(float(row.get("missing", 1))) == 0
        for row in paper_package_coverage
    )
    visa_mini_categories = sorted({row.get("category", "") for row in visa_mini_main if row.get("category")})
    visa_gate_categories = sorted({row.get("category", "") for row in visa_gate_main if row.get("category")})
    mpdd_mini_categories = sorted({row.get("category", "") for row in mpdd_mini_main if row.get("category")})
    mpdd_gate_categories = sorted({row.get("category", "") for row in mpdd_gate_main if row.get("category")})
    visa_mini_means = metric_means(visa_mini_main, METRICS)
    visa_gate_means = metric_means(visa_gate_main, METRICS)
    mpdd_mini_means = metric_means(mpdd_mini_main, METRICS)
    mpdd_gate_means = metric_means(mpdd_gate_main, METRICS)

    lines = [
        "# Lite-SEER-AD v2 Paper Evidence Summary",
        "",
        "## Current State",
        "",
        f"- MVTec15 plan audit: `complete=true` was previously verified by `tables/plan_audit/plan_audit.json`.",
        f"- Baseline coverage: `{coverage.get('present', 'n/a')}/{coverage.get('total_required', 'n/a')}`, complete=`{coverage.get('complete', 'n/a')}`.",
        f"- MVTec5 gate: HN-SEV `{mvtec5_gate.get('module_passes', {}).get('hn_sev', 'n/a')}/{mvtec5_gate.get('module_total', {}).get('hn_sev', 'n/a')}`, CRV `{mvtec5_gate.get('module_passes', {}).get('crv', 'n/a')}/{mvtec5_gate.get('module_total', {}).get('crv', 'n/a')}`, LC-RDS `{mvtec5_gate.get('module_passes', {}).get('lc_rds', 'n/a')}/{mvtec5_gate.get('module_total', {}).get('lc_rds', 'n/a')}`.",
        f"- MVTec15 gate: HN-SEV `{mvtec15_gate.get('module_passes', {}).get('hn_sev', 'n/a')}/{mvtec15_gate.get('module_total', {}).get('hn_sev', 'n/a')}`, CRV `{mvtec15_gate.get('module_passes', {}).get('crv', 'n/a')}/{mvtec15_gate.get('module_total', {}).get('crv', 'n/a')}`, LC-RDS `{mvtec15_gate.get('module_passes', {}).get('lc_rds', 'n/a')}/{mvtec15_gate.get('module_total', {}).get('lc_rds', 'n/a')}`.",
        "",
        "## MVTec15 Reading",
        "",
        method_table,
        "",
        f"- Against the best baseline by category, `{args.ours_method}` wins `{win_summary['wins']}/{win_summary['categories']}` categories on Pixel AP.",
        f"- Mean Pixel AP delta versus the best category baseline is `{fmt(win_summary['mean_delta'])}`.",
        "- This does not support a全面 SOTA claim. The paper should emphasize repair-aware verification, module evidence, and low-budget regional diffusion reasoning.",
        "",
        win_table,
        "",
        "## Efficiency Reading",
        "",
        eff_table,
        "",
        "- Current Lite-SEER-AD runs are slower than memory/statistical baselines and many local baselines on MVTec15.",
        "- LC-RDS should therefore be presented as a budget-allocation contribution inside the repair-aware diffusion pipeline, not as an overall fastest-method claim.",
        "",
        "## Protocol Decisions",
        "",
        f"- Default CRV weight for paper-scale MVTec15 comparison: `{crv_decision.get('recommended_crv_weight', 0.35)}`.",
        f"- MVTec5 CRV search recommended `{crv5_decision.get('recommended', {}).get('crv_weight', 'n/a')}`; keep it as weight-sensitivity evidence, not the default for the full paper protocol.",
        "- Default narrative: repair-aware anomaly verification + low-budget diffusion scheduling, not full-metric SOTA.",
        "- Default next dataset order: VisA first, MPDD second.",
        "",
        "## Dataset Smoke Status",
        "",
        f"- VisA smoke: complete=`{visa_smoke_ok}`, runs=`{len(visa_smoke_report.get('schema', [])) if visa_smoke_report else 0}`, failures=`{len(visa_smoke_report.get('failures', [])) if visa_smoke_report else 'n/a'}`.",
        f"- MPDD smoke: complete=`{mpdd_smoke_ok}`, runs=`{len(mpdd_smoke_report.get('schema', [])) if mpdd_smoke_report else 0}`, failures=`{len(mpdd_smoke_report.get('failures', [])) if mpdd_smoke_report else 'n/a'}`.",
        "- These smoke runs validate data loading, masks, standard outputs, qualitative artifacts, and summary export only. They are not paper-strength contribution evidence.",
        "",
        "## VisA Mini Status",
        "",
        f"- VisA mini core pass: complete=`{visa_mini_ok}`, schema_runs=`{len(visa_mini_report.get('schema', [])) if visa_mini_report else 0}`, failures=`{len(visa_mini_report.get('failures', [])) if visa_mini_report else 'n/a'}`.",
        f"- Covered categories: `{len(visa_mini_categories)}`/`{len(visa_categories)}` detected VisA categories: `{', '.join(visa_mini_categories)}`.",
        f"- Full-model mean metrics: image AUROC `{fmt(visa_mini_means.get('image_auroc'))}`, pixel AUROC `{fmt(visa_mini_means.get('pixel_auroc'))}`, AUPRO `{fmt(visa_mini_means.get('aupro'))}`, Pixel AP `{fmt(visa_mini_means.get('pixel_ap'))}`, Dice `{fmt(visa_mini_means.get('dice'))}`.",
        f"- Mini module gate: HN-SEV `{visa_mini_gate.get('module_passes', {}).get('hn_sev', 'n/a')}/{visa_mini_gate.get('module_total', {}).get('hn_sev', 'n/a')}`, CRV `{visa_mini_gate.get('module_passes', {}).get('crv', 'n/a')}/{visa_mini_gate.get('module_total', {}).get('crv', 'n/a')}`, prototype `{visa_mini_gate.get('module_passes', {}).get('prototype', 'n/a')}/{visa_mini_gate.get('module_total', {}).get('prototype', 'n/a')}`, LC-RDS `{visa_mini_gate.get('module_passes', {}).get('lc_rds', 'n/a')}/{visa_mini_gate.get('module_total', {}).get('lc_rds', 'n/a')}`.",
        "- This mini pass used the core ablation set: `full,residual_only,no_sev,no_crv,rule_brds`. Prototype and full LC-RDS gates are therefore intentionally incomplete.",
        "",
        "## VisA Gate Status",
        "",
        f"- VisA gate pass: complete=`{visa_gate_ok}`, schema_runs=`{len(visa_gate_report.get('schema', [])) if visa_gate_report else 0}`, failures=`{len(visa_gate_report.get('failures', [])) if visa_gate_report else 'n/a'}`.",
        f"- Covered categories: `{len(visa_gate_categories)}`/`{len(visa_categories)}` detected VisA categories: `{', '.join(visa_gate_categories)}`.",
        f"- Full-model mean metrics from this pass: image AUROC `{fmt(visa_gate_means.get('image_auroc'))}`, pixel AUROC `{fmt(visa_gate_means.get('pixel_auroc'))}`, AUPRO `{fmt(visa_gate_means.get('aupro'))}`, Pixel AP `{fmt(visa_gate_means.get('pixel_ap'))}`, Dice `{fmt(visa_gate_means.get('dice'))}`.",
        f"- Gate-specific module evidence: prototype `{visa_gate_gate.get('module_passes', {}).get('prototype', 'n/a')}/{visa_gate_gate.get('module_total', {}).get('prototype', 'n/a')}`, LC-RDS `{visa_gate_gate.get('module_passes', {}).get('lc_rds', 'n/a')}/{visa_gate_gate.get('module_total', {}).get('lc_rds', 'n/a')}`.",
        "- This pass complements the VisA mini core pass. It intentionally does not retest HN-SEV/CRV comparisons, so the gate summary should be read by module family rather than as a single all-module ready flag.",
        "",
        "## MPDD Mini Status",
        "",
        f"- MPDD mini core pass: complete=`{mpdd_mini_ok}`, schema_runs=`{len(mpdd_mini_report.get('schema', [])) if mpdd_mini_report else 0}`, failures=`{len(mpdd_mini_report.get('failures', [])) if mpdd_mini_report else 'n/a'}`.",
        f"- Covered categories: `{len(mpdd_mini_categories)}`/`{len(mpdd_categories)}` detected MPDD categories: `{', '.join(mpdd_mini_categories)}`.",
        f"- Full-model mean metrics: image AUROC `{fmt(mpdd_mini_means.get('image_auroc'))}`, pixel AUROC `{fmt(mpdd_mini_means.get('pixel_auroc'))}`, AUPRO `{fmt(mpdd_mini_means.get('aupro'))}`, Pixel AP `{fmt(mpdd_mini_means.get('pixel_ap'))}`, Dice `{fmt(mpdd_mini_means.get('dice'))}`.",
        f"- Mini module gate: HN-SEV `{mpdd_mini_gate.get('module_passes', {}).get('hn_sev', 'n/a')}/{mpdd_mini_gate.get('module_total', {}).get('hn_sev', 'n/a')}`, CRV `{mpdd_mini_gate.get('module_passes', {}).get('crv', 'n/a')}/{mpdd_mini_gate.get('module_total', {}).get('crv', 'n/a')}`, prototype `{mpdd_mini_gate.get('module_passes', {}).get('prototype', 'n/a')}/{mpdd_mini_gate.get('module_total', {}).get('prototype', 'n/a')}`, LC-RDS `{mpdd_mini_gate.get('module_passes', {}).get('lc_rds', 'n/a')}/{mpdd_mini_gate.get('module_total', {}).get('lc_rds', 'n/a')}`.",
        "- MPDD now has all-category core evidence. Prototype and full LC-RDS still need a gate-specific pass if MPDD module claims are kept symmetrical with VisA.",
        "",
        "## MPDD Gate Status",
        "",
        f"- MPDD gate pass: complete=`{mpdd_gate_ok}`, schema_runs=`{len(mpdd_gate_report.get('schema', [])) if mpdd_gate_report else 0}`, failures=`{len(mpdd_gate_report.get('failures', [])) if mpdd_gate_report else 'n/a'}`.",
        f"- Covered categories: `{len(mpdd_gate_categories)}`/`{len(mpdd_categories)}` detected MPDD categories: `{', '.join(mpdd_gate_categories)}`.",
        f"- Full-model mean metrics from this pass: image AUROC `{fmt(mpdd_gate_means.get('image_auroc'))}`, pixel AUROC `{fmt(mpdd_gate_means.get('pixel_auroc'))}`, AUPRO `{fmt(mpdd_gate_means.get('aupro'))}`, Pixel AP `{fmt(mpdd_gate_means.get('pixel_ap'))}`, Dice `{fmt(mpdd_gate_means.get('dice'))}`.",
        f"- Gate-specific module evidence: prototype `{mpdd_gate_gate.get('module_passes', {}).get('prototype', 'n/a')}/{mpdd_gate_gate.get('module_total', {}).get('prototype', 'n/a')}`, LC-RDS `{mpdd_gate_gate.get('module_passes', {}).get('lc_rds', 'n/a')}/{mpdd_gate_gate.get('module_total', {}).get('lc_rds', 'n/a')}`.",
        "- This completes the same mini/core plus gate-specific evidence pattern now used for VisA.",
        "",
        "## Baseline Expansion Status",
        "",
        "- The MVTec15 baseline runner now supports non-MVTec config/category discovery and writes dataset-specific table aliases such as `table_main_visa.csv` and `table_main_mpdd.csv`.",
        f"- VisA baseline smoke: complete=`{visa_baseline_smoke_ok}`, methods=`{visa_baseline_smoke_coverage.get('present', 'n/a')}/{visa_baseline_smoke_coverage.get('total_required', 'n/a')}`, failures=`{len(visa_baseline_smoke_report.get('failures', [])) if visa_baseline_smoke_report else 'n/a'}`.",
        f"- MPDD baseline smoke: complete=`{mpdd_baseline_smoke_ok}`, methods=`{mpdd_baseline_smoke_coverage.get('present', 'n/a')}/{mpdd_baseline_smoke_coverage.get('total_required', 'n/a')}`, failures=`{len(mpdd_baseline_smoke_report.get('failures', [])) if mpdd_baseline_smoke_report else 'n/a'}`.",
        f"- VisA full baseline table: complete=`{visa_baseline_ok}`, coverage=`{visa_baseline_coverage.get('present', 'n/a')}/{visa_baseline_coverage.get('total_required', 'n/a')}`, failures=`{len(visa_baseline_report.get('failures', [])) if visa_baseline_report else 'n/a'}`.",
        f"- MPDD full baseline table: complete=`{mpdd_baseline_ok}`, coverage=`{mpdd_baseline_coverage.get('present', 'n/a')}/{mpdd_baseline_coverage.get('total_required', 'n/a')}`, failures=`{len(mpdd_baseline_report.get('failures', [])) if mpdd_baseline_report else 'n/a'}`.",
        "- Smoke baselines used tiny samples and `--allow-random-weights`; full baseline tables were rerun across all categories without random weights.",
        "",
        "## Paper Package Status",
        "",
        f"- Cross-dataset package exported to `tables/paper_package`: run_status_ok=`{paper_package_ok}`, baseline_coverage_ok=`{paper_coverage_ok}`.",
        "- Package artifacts: `table_main_cross_dataset.csv`, `table_efficiency_cross_dataset.csv`, `table_mean_by_dataset_method.csv`, `table_category_deltas.csv`, `table_module_gates.csv`, `table_qualitative_case_index.csv`, `fig_pareto_pixel_ap_fps.png`, `table_run_status.csv`, `table_baseline_coverage.csv`, and `failure_analysis_notes.md`.",
        "- Results/limitations draft: `docs/results_limitations_draft.md`.",
        "",
        "## Next Experiments",
        "",
        f"- VisA root available: `{visa_root.as_posix()}` with `{len(visa_categories)}` detected categories: `{', '.join(visa_categories)}`.",
        f"- MPDD root available: `{mpdd_root.as_posix()}` with `{len(mpdd_categories)}` detected categories: `{', '.join(mpdd_categories)}`.",
        "- Next analysis step: inspect `tables/paper_package/table_mean_by_dataset_method.csv` and category-level deltas to write the actual Results/Limitations narrative.",
        "- Next figure step: build Pareto and qualitative failure panels from current MVTec15, VisA, and MPDD runs.",
        "",
        "## Acceptance Gates",
        "",
        "- VisA: full model all 12 categories, at least 6 baselines, main/efficiency/HN-SEV/CRV/LC-RDS tables, and qualitative cases.",
        "- MPDD: full model all 6 categories, baseline coverage aligned with VisA, reflective/metal false-positive analysis, SDR evidence, and failure cases.",
        "- Paper package: merged MVTec/VisA/MPDD tables, Pareto plots, module ablations, and honest failure analysis.",
        "",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    write_summary(parse_args())


if __name__ == "__main__":
    main()
