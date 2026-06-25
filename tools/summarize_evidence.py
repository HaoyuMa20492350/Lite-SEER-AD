from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


MAIN_FIELDS = [
    "run",
    "dataset",
    "category",
    "ablation",
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "aupro_proxy",
    "pixel_ap",
    "f1",
    "iou",
    "dice",
    "fprr",
    "rdc",
    "sdr_mean",
    "pareto_area",
    "image_score_mode",
    "image_score_source",
    "pixel_heatmap_source",
]
EFF_FIELDS = ["run", "category", "ablation", "latency_ms_mean", "fps", "nfe_mean", "repaired_area_ratio_mean", "local_region_ratio_mean", "gpu_memory_mb"]
HN_ABLATIONS = {"full", "no_sev", "residual_only", "synthetic_only_sev", "no_prototype"}
CRV_ABLATIONS = {"full", "no_crv", "repair_visualization_only"}
LC_ABLATIONS = {"full", "fixed10", "fixed25", "rule_brds", "learned_lc_rds"}
FEATURE_MAIN_ABLATIONS = {"full", "feature_tuned_crv", "feature_hn_sev_crv", "feature_first", "feature_pixel_policy", "utility_lc_rds"}
HN_ABLATIONS |= {"feature_only", "feature_hn_sev", "feature_tuned_crv", "feature_hn_sev_crv"}
CRV_ABLATIONS |= {"feature_hn_sev", "feature_tuned_crv", "feature_hn_sev_crv", "feature_first", "feature_pixel_policy"}
LC_ABLATIONS |= {
    "feature_tuned_crv",
    "feature_hn_sev_crv",
    "feature_first",
    "feature_pixel_policy",
    "feature_fixed10",
    "feature_fixed25",
    "feature_rule_brds",
    "utility_lc_rds",
}
CRV_DELTA_EPS = 1e-3
PROTOTYPE_QUALITY_EPS = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize HN-SEV, CRV, and LC-RDS evidence from mini experiment runs.")
    p.add_argument("--runs", default="runs")
    p.add_argument("--prefix", default="mini_mvtec")
    p.add_argument("--exclude-prefix", default=None, help="Comma-separated run prefixes to exclude from the summary.")
    p.add_argument("--include-ablations", default=None, help="Comma-separated ablations to include in the summary.")
    p.add_argument("--out", default="tables/mini_mvtec")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metric_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row.get("metric", ""): row.get("value", "") for row in csv.DictReader(f) if row.get("metric")}


def _float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _quality(row: dict[str, Any]) -> float:
    vals = [_float(row, "pixel_ap"), _float(row, "dice"), _float(row, "aupro")]
    vals = [v for v in vals if _finite(v)]
    if not vals:
        vals = [_float(row, "aupro_proxy")]
        vals = [v for v in vals if _finite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _hn_evidence_row(category: str, full: dict[str, Any], baseline_name: str, baseline: dict[str, Any]) -> dict[str, Any]:
    fprr_drop = _float(baseline, "fprr") - _float(full, "fprr")
    quality_delta = _quality(full) - _quality(baseline)
    if _finite(fprr_drop):
        passed = fprr_drop > 0.0 and quality_delta >= -0.01
    else:
        passed = quality_delta > 0.0
    return {
        "module": "hn_sev",
        "category": category,
        "comparison": f"full_vs_{baseline_name}",
        "primary_delta": fprr_drop,
        "quality_delta": quality_delta,
        "passed": passed,
    }


def _prototype_evidence_row(category: str, full: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    fprr_drop = _float(baseline, "fprr") - _float(full, "fprr")
    quality_delta = _quality(full) - _quality(baseline)
    if _finite(fprr_drop):
        passed = (fprr_drop > 0.0 and quality_delta >= -0.01) or (
            quality_delta > PROTOTYPE_QUALITY_EPS and fprr_drop >= -0.01
        )
    else:
        passed = quality_delta > PROTOTYPE_QUALITY_EPS
    return {
        "module": "prototype",
        "category": category,
        "comparison": "full_vs_no_prototype",
        "primary_delta": fprr_drop,
        "quality_delta": quality_delta,
        "passed": passed,
    }


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _split_prefixes(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _rows(runs_root: Path, prefix: str, exclude_prefix: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    excluded = _split_prefixes(exclude_prefix)
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(prefix + "_")]):
        if any(run_dir.name.startswith(item + "_") for item in excluded):
            continue
        if not (run_dir / "metrics.json").exists() or not (run_dir / "run_args.json").exists():
            continue
        metrics = _load_json(run_dir / "metrics.json")
        efficiency = _load_metric_csv(run_dir / "efficiency.csv")
        args = _load_json(run_dir / "run_args.json")
        cfg = _load_yaml(run_dir / "config.yaml")
        arg_payload = args.get("args", {}) if isinstance(args, dict) else {}
        row = {
            "run": run_dir.name,
            "dataset": (cfg.get("dataset", {}) or {}).get("name", ""),
            "category": arg_payload.get("category") or (cfg.get("dataset", {}) or {}).get("category", ""),
            "ablation": arg_payload.get("ablation", "full"),
        }
        row.update(metrics)
        row.update(efficiency)
        rows.append(row)
    return rows


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _by_category_ablation(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        category = str(row.get("category", ""))
        ablation = str(row.get("ablation", "full"))
        out.setdefault(category, {})[ablation] = row
    return out


def _evidence(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    grouped = _by_category_ablation(rows)
    module_passes = {"hn_sev": 0, "prototype": 0, "crv": 0, "lc_rds": 0}
    module_total = {"hn_sev": 0, "prototype": 0, "crv": 0, "lc_rds": 0}
    feature_first_gate = any(str(row.get("ablation", "")).startswith("feature_") or str(row.get("ablation", "")) == "utility_lc_rds" for row in rows)

    for category, variants in grouped.items():
        full = (
            variants.get("full")
            or variants.get("feature_tuned_crv")
            or variants.get("feature_hn_sev_crv")
            or variants.get("feature_first")
            or variants.get("utility_lc_rds")
        )
        if not full:
            continue

        hn_primary = next((name for name in ["feature_only", "synthetic_only_sev", "no_sev", "residual_only"] if name in variants), None)
        if hn_primary:
            module_total["hn_sev"] += 1
            primary_row = _hn_evidence_row(category, full, hn_primary, variants[hn_primary])
            module_passes["hn_sev"] += int(bool(primary_row["passed"]))
            evidence.append(primary_row)
            for baseline_name in ["no_sev", "residual_only"]:
                if baseline_name in variants and baseline_name != hn_primary:
                    evidence.append(_hn_evidence_row(category, full, baseline_name, variants[baseline_name]))

        prototype_baseline = variants.get("no_prototype")
        if prototype_baseline:
            module_total["prototype"] += 1
            prototype_row = _prototype_evidence_row(category, full, prototype_baseline)
            module_passes["prototype"] += int(bool(prototype_row["passed"]))
            evidence.append(prototype_row)

        crv_category_seen = False
        crv_category_passed = False
        for baseline_name in ["feature_hn_sev", "no_crv", "repair_visualization_only"]:
            baseline = variants.get(baseline_name)
            if not baseline:
                continue
            crv_category_seen = True
            deltas = {
                "pixel_ap_delta": _float(full, "pixel_ap") - _float(baseline, "pixel_ap"),
                "dice_delta": _float(full, "dice") - _float(baseline, "dice"),
                "aupro_delta": _float(full, "aupro") - _float(baseline, "aupro"),
                "aupro_proxy_delta": _float(full, "aupro_proxy") - _float(baseline, "aupro_proxy"),
            }
            quality_delta = _quality(full) - _quality(baseline)
            passed = any(v > CRV_DELTA_EPS for v in deltas.values() if _finite(v)) or quality_delta > CRV_DELTA_EPS
            crv_category_passed = crv_category_passed or passed
            evidence.append(
                {
                    "module": "crv",
                    "category": category,
                    "comparison": f"full_vs_{baseline_name}",
                    "primary_delta": max([v for v in deltas.values() if _finite(v)] or [float("nan")]),
                    "quality_delta": quality_delta,
                    "passed": passed,
                    **deltas,
                }
            )
        if crv_category_seen:
            module_total["crv"] += 1
            module_passes["crv"] += int(crv_category_passed)

        lc_rows = [
            variants[name]
            for name in [
                "full",
                "feature_tuned_crv",
                "fixed10",
                "fixed25",
                "rule_brds",
                "learned_lc_rds",
                "feature_hn_sev_crv",
                "feature_first",
                "feature_fixed10",
                "feature_fixed25",
                "feature_rule_brds",
                "utility_lc_rds",
            ]
            if name in variants
        ]
        if len(lc_rows) >= 3:
            module_total["lc_rds"] += 1
            scored = []
            for row in lc_rows:
                latency = _float(row, "latency_ms_mean")
                quality = _quality(row)
                score = quality / max(latency, 1e-8) if _finite(quality) and _finite(latency) and latency > 0 else float("nan")
                scored.append((str(row.get("ablation", "")), score, quality, latency, _float(row, "nfe_mean")))
            scored_finite = [s for s in scored if _finite(s[1])]
            best = max(scored_finite, key=lambda x: x[1]) if scored_finite else ("", float("nan"), float("nan"), float("nan"), float("nan"))
            passed = best[0] in {
                "full",
                "learned_lc_rds",
                "rule_brds",
                "feature_tuned_crv",
                "feature_hn_sev_crv",
                "feature_first",
                "feature_rule_brds",
                "utility_lc_rds",
            }
            module_passes["lc_rds"] += int(passed)
            evidence.append(
                {
                    "module": "lc_rds",
                    "category": category,
                    "comparison": "latency_nfe_accuracy_proxy",
                    "primary_delta": best[1],
                    "quality_delta": best[2],
                    "latency_ms_mean": best[3],
                    "nfe_mean": best[4],
                    "best_ablation": best[0],
                    "passed": passed,
                }
            )

    module_ready = {key: module_total[key] >= 3 and module_passes[key] >= 3 for key in module_total}
    prototype_ready = module_ready["prototype"] or feature_first_gate
    chain_ready = {
        "hn_sev": module_ready["hn_sev"] and prototype_ready,
        "crv": module_ready["crv"],
        "lc_rds": module_ready["lc_rds"],
    }
    gate = {
        "module_passes": module_passes,
        "module_total": module_total,
        "module_ready": module_ready,
        "feature_first_gate": feature_first_gate,
        "prototype_required": not feature_first_gate,
        "chain_ready": chain_ready,
        "ready_for_mvtec15": all(chain_ready.values()),
    }
    return evidence, gate


def main() -> None:
    args = parse_args()
    rows = _rows(Path(args.runs), args.prefix, args.exclude_prefix)
    if args.include_ablations:
        included = set(_split_prefixes(args.include_ablations))
        rows = [row for row in rows if str(row.get("ablation", "")) in included]
    out = Path(args.out)
    _write(out / "table_main_mvtec5.csv", [r for r in rows if r.get("ablation") in FEATURE_MAIN_ABLATIONS], MAIN_FIELDS)
    _write(out / "table_efficiency_mvtec5.csv", rows, EFF_FIELDS)
    _write(out / "table_ablation_hn_sev.csv", [r for r in rows if r.get("ablation") in HN_ABLATIONS], MAIN_FIELDS)
    _write(out / "table_ablation_crv.csv", [r for r in rows if r.get("ablation") in CRV_ABLATIONS], MAIN_FIELDS)
    _write(out / "table_ablation_lc_rds.csv", [r for r in rows if r.get("ablation") in LC_ABLATIONS], MAIN_FIELDS)
    evidence, gate = _evidence(rows)
    evidence_fields = sorted({key for row in evidence for key in row.keys()})
    _write(out / "evidence_summary.csv", evidence, evidence_fields or ["module", "category", "passed"])
    (out / "gate_summary.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(f"Summarized {len(rows)} runs to {out}; ready_for_mvtec15={gate['ready_for_mvtec15']}")


if __name__ == "__main__":
    main()
