from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MVTEC5_CATEGORIES = ["bottle", "cable", "capsule", "metal_nut", "zipper"]
MVTEC15_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
REQUIRED_METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]
REQUIRED_PLAN_METRICS = ["fprr", "sdr_mean", "rdc", "pareto_area"]
REQUIRED_EFFICIENCY = ["latency_ms_mean", "fps", "nfe_mean", "repaired_area_ratio_mean"]
REQUIRED_TABLES = ["table_ablation_hn_sev.csv", "table_ablation_crv.csv", "table_ablation_lc_rds.csv"]
REQUIRED_VISUAL_FILES = [
    "input.png",
    "reconstruction.png",
    "residual.png",
    "candidate_roi.png",
    "verified_roi.png",
    "final_mask.png",
    "repair.png",
    "ground_truth.png",
]
REQUIRED_BASELINES = ["patchcore", "padim", "simplenet", "draem", "rd4ad", "uniad", "diffusionad", "ddad"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Lite-SEER-AD v2 plan completion against current artifacts.")
    p.add_argument("--runs", default="runs")
    p.add_argument("--mvtec5-tables", default="tables/next_mvtec5")
    p.add_argument("--mvtec15-tables", default="tables/mvtec15_ours")
    p.add_argument("--baseline-tables", default="tables/mvtec15_baselines")
    p.add_argument("--out", default="tables/plan_audit")
    p.add_argument("--fail-on-incomplete", action="store_true")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in {"", None} else float("nan")
    except Exception:
        return float("nan")


def _quality(row: dict[str, Any]) -> float:
    vals = [_float(row, key) for key in ["pixel_ap", "dice", "aupro"]]
    vals = [value for value in vals if math.isfinite(value)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _row(section: str, requirement: str, passed: bool, evidence: str, detail: str = "") -> dict[str, Any]:
    return {
        "section": section,
        "requirement": requirement,
        "status": "pass" if passed else "fail",
        "evidence": evidence,
        "detail": detail,
    }


def _gate_rows(name: str, path: Path, require_prototype: bool = True) -> list[dict[str, Any]]:
    gate = _load_json(path)
    rows = [_row(name, "gate_summary.json exists", bool(gate), str(path))]
    rows.append(_row(name, "ready_for_mvtec15 is true", bool(gate.get("ready_for_mvtec15")), str(path), json.dumps(gate.get("chain_ready", {}))))
    module_ready = gate.get("module_ready", {})
    for module in ["hn_sev", "crv", "lc_rds"]:
        rows.append(_row(name, f"{module} module ready", bool(module_ready.get(module)), str(path), json.dumps(gate.get("module_passes", {}))))
    if require_prototype:
        rows.append(_row(name, "prototype sub-branch ready", bool(module_ready.get("prototype")), str(path), json.dumps(gate.get("module_passes", {}))))
    return rows


def _table_rows(section: str, table_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for name in REQUIRED_TABLES:
        path = table_dir / name
        data = _read_csv(path)
        rows.append(_row(section, f"{name} exists and has rows", bool(data), str(path), f"rows={len(data)}"))
    recon = table_dir / "table_reconstruction_sweep.csv"
    recon_rows = _read_csv(recon)
    steps_by_cat: dict[str, set[int]] = {}
    for row in recon_rows:
        try:
            steps_by_cat.setdefault(str(row.get("category", "")), set()).add(int(float(row.get("reconstruction_steps", ""))))
        except Exception:
            pass
    complete_sweep = all(steps_by_cat.get(cat, set()) >= {1, 5, 10} for cat in MVTEC5_CATEGORIES)
    rows.append(_row(section, "reconstruction_steps sweep covers 1/5/10 for MVTec5", complete_sweep, str(recon), json.dumps({k: sorted(v) for k, v in steps_by_cat.items()})))
    return rows


def _metric_contract_rows(section: str, table_dir: Path, main_table_name: str) -> list[dict[str, Any]]:
    path = table_dir / main_table_name
    rows = _read_csv(path)
    out = [_row(section, f"{main_table_name} exists and has rows", bool(rows), str(path), f"rows={len(rows)}")]
    for key in REQUIRED_METRICS + REQUIRED_PLAN_METRICS:
        finite = [row for row in rows if math.isfinite(_float(row, key))]
        out.append(_row(section, f"main table has finite {key}", bool(rows) and len(finite) == len(rows), str(path), f"finite={len(finite)}/{len(rows)}"))
    eff_path = table_dir / main_table_name.replace("table_main", "table_efficiency")
    eff_rows = _read_csv(eff_path)
    for key in REQUIRED_EFFICIENCY:
        finite = [row for row in eff_rows if math.isfinite(_float(row, key))]
        out.append(_row(section, f"efficiency table has finite {key}", bool(eff_rows) and len(finite) == len(eff_rows), str(eff_path), f"finite={len(finite)}/{len(eff_rows)}"))
    return out


def _visual_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows = []
    for category in MVTEC5_CATEGORIES:
        run_dir = runs_root / f"next_mvtec5_{category}_full" / "images"
        complete = 0
        incomplete: list[str] = []
        for case_dir in sorted([path for path in run_dir.glob("*") if path.is_dir()]):
            missing = [name for name in REQUIRED_VISUAL_FILES if not (case_dir / name).exists()]
            if missing:
                incomplete.append(f"{case_dir.name}:{'|'.join(missing)}")
            else:
                complete += 1
        rows.append(
            _row(
                "visuals",
                f"{category} has at least 5 complete qualitative cases",
                complete >= 5,
                str(run_dir),
                f"complete={complete}, first_incomplete={incomplete[:3]}",
            )
        )
    return rows


def _residual_quality_rows(table_dir: Path) -> list[dict[str, Any]]:
    path = table_dir / "table_reconstruction_sweep.csv"
    rows = _read_csv(path)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(str(row.get("category", "")), []).append(row)
    out = []
    selected: dict[str, dict[str, Any]] = {}
    for category in MVTEC5_CATEGORIES:
        candidates = by_category.get(category, [])
        finite = [row for row in candidates if math.isfinite(_quality(row))]
        if finite:
            best = max(finite, key=_quality)
            selected[category] = {"run": best.get("run", ""), "steps": best.get("reconstruction_steps", ""), "quality": _quality(best)}
    out.append(_row("residual_quality", "best reconstruction setting selected for every MVTec5 category", len(selected) == len(MVTEC5_CATEGORIES), str(path), json.dumps(selected)))
    return out


def _baseline_rows(path: Path) -> list[dict[str, Any]]:
    coverage = _load_json(path)
    rows = [_row("mvtec15_baselines", "baseline_coverage.json exists", bool(coverage), str(path))]
    rows.append(_row("mvtec15_baselines", "required baseline coverage complete", bool(coverage.get("complete")), str(path), f"present={coverage.get('present')}/{coverage.get('total_required')}, missing={coverage.get('missing')}"))
    by_method = coverage.get("by_method", {})
    for method in REQUIRED_BASELINES:
        info = by_method.get(method, {})
        rows.append(_row("mvtec15_baselines", f"{method} covers all 15 classes", int(info.get("missing", 15)) == 0, str(path), json.dumps(info)))
    return rows


def _narrative_rows() -> list[dict[str, Any]]:
    paths = [Path("README.md"), Path("docs/implementation_map.md"), Path("docs/literature_matrix.md")]
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in paths if path.exists())
    sota_claim = False
    for line in text.splitlines():
        if re.search(r"\bSOTA\b|state[- ]of[- ]the[- ]art|全面", line, flags=re.IGNORECASE) is None:
            continue
        if re.search(
            r"\b(no|not|cannot|can't|avoid|without|should not|must not|non-sota)\b|"
            r"不能|不可|不应|不得|尚未|未达到|不宣称",
            line,
            flags=re.IGNORECASE,
        ):
            continue
        if re.search(
            r"(achiev|establish|outperform|surpass|new|broad|overall|is\s+(a\s+)?sota|"
            r"sota\s+(method|detector|system|performance)|全面.*(领先|超越|达到))",
            line,
            flags=re.IGNORECASE,
        ):
            sota_claim = True
            break
    positioning = re.search(r"repair-aware|修复感知|low-budget|低算力|budget", text, flags=re.IGNORECASE) is not None
    return [
        _row("paper_positioning", "no broad SOTA claim in docs", not sota_claim, "README.md; docs/", "searched SOTA/state-of-the-art/全面"),
        _row("paper_positioning", "repair-aware or low-budget positioning stated", positioning, "README.md; docs/", "searched repair-aware/low-budget/低算力/budget"),
    ]


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs)
    mvtec5 = Path(args.mvtec5_tables)
    mvtec15 = Path(args.mvtec15_tables)
    baseline = Path(args.baseline_tables)
    audit_rows: list[dict[str, Any]] = []
    audit_rows.extend(_gate_rows("mvtec5_gate", mvtec5 / "gate_summary.json"))
    audit_rows.extend(_table_rows("mvtec5_tables", mvtec5))
    audit_rows.extend(_metric_contract_rows("mvtec5_metrics", mvtec5, "table_main_mvtec5.csv"))
    audit_rows.extend(_visual_rows(runs_root))
    audit_rows.extend(_residual_quality_rows(mvtec5))
    audit_rows.extend(_gate_rows("mvtec15_ours_gate", mvtec15 / "gate_summary.json"))
    audit_rows.extend(_metric_contract_rows("mvtec15_ours_metrics", mvtec15, "table_main_mvtec15.csv"))
    audit_rows.extend(_baseline_rows(baseline / "baseline_coverage.json"))
    audit_rows.extend(_narrative_rows())

    failed = [row for row in audit_rows if row["status"] != "pass"]
    summary = {
        "total": len(audit_rows),
        "passed": len(audit_rows) - len(failed),
        "failed": len(failed),
        "complete": not failed,
        "failed_requirements": [{key: row[key] for key in ["section", "requirement", "detail"]} for row in failed],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "plan_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "requirement", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(audit_rows)
    (out / "plan_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.fail_on_incomplete and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
