from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MAIN_FIELDS = ["method", "category", "image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]
EFF_FIELDS = ["method", "category", "latency_ms_mean", "fps", "nfe_mean"]
CRV_FIELDS = ["category", "quality_035", "quality_05", "quality_delta_05_minus_035", "pixel_ap_035", "pixel_ap_05", "dice_035", "dice_05", "aupro_035", "aupro_05"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export MVTec15 ours-vs-baseline comparison tables.")
    p.add_argument("--ours", default="tables/mvtec15_ours")
    p.add_argument("--ours-crv05", default="tables/mvtec15_ours_crv05")
    p.add_argument("--baselines", default="tables/mvtec15_baselines")
    p.add_argument("--out", default="tables/mvtec15_comparison")
    return p.parse_args()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, Any], key: str) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in {"", None} else float("nan")
    except Exception:
        return float("nan")


def _quality(row: dict[str, Any]) -> float:
    vals = [_float(row, "pixel_ap"), _float(row, "dice"), _float(row, "aupro")]
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _method_rows(rows: list[dict[str, Any]], method_name: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        new = dict(row)
        new["method"] = method_name
        out.append(new)
    return out


def _eff_method_rows(rows: list[dict[str, Any]], method_name: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("ablation") not in {"", None, "full"} and method_name.startswith("lite_seer"):
            continue
        new = dict(row)
        new["method"] = method_name
        out.append(new)
    return out


def _by_category(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("category", "")): row for row in rows if row.get("category")}


def _crv_weight_rows(ours035: list[dict[str, Any]], ours05: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = _by_category(ours035)
    cand = _by_category(ours05)
    rows = []
    for category in sorted(set(base) & set(cand)):
        b = base[category]
        c = cand[category]
        q035 = _quality(b)
        q05 = _quality(c)
        rows.append(
            {
                "category": category,
                "quality_035": q035,
                "quality_05": q05,
                "quality_delta_05_minus_035": q05 - q035,
                "pixel_ap_035": b.get("pixel_ap", ""),
                "pixel_ap_05": c.get("pixel_ap", ""),
                "dice_035": b.get("dice", ""),
                "dice_05": c.get("dice", ""),
                "aupro_035": b.get("aupro", ""),
                "aupro_05": c.get("aupro", ""),
            }
        )
    deltas = [_float(row, "quality_delta_05_minus_035") for row in rows]
    positive = sum(1 for v in deltas if v > 0)
    summary = {
        "categories": len(rows),
        "crv05_quality_positive_categories": positive,
        "mean_quality_delta_05_minus_035": sum(deltas) / len(deltas) if deltas else None,
        "recommended_crv_weight": 0.5 if deltas and positive >= 8 and sum(deltas) / len(deltas) > 0 else 0.35,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    ours_dir = Path(args.ours)
    crv05_dir = Path(args.ours_crv05)
    baseline_dir = Path(args.baselines)
    out = Path(args.out)

    ours_main = _read_csv(ours_dir / "table_main_mvtec15.csv")
    crv05_main = _read_csv(crv05_dir / "table_main_mvtec15.csv")
    baseline_main = _read_csv(baseline_dir / "table_main_mvtec15.csv")
    main_rows = _method_rows(ours_main, "lite_seer_ad_crv035")
    if crv05_main:
        main_rows.extend(_method_rows(crv05_main, "lite_seer_ad_crv05"))
    for row in baseline_main:
        method = row.get("method", "baseline")
        main_rows.append({**row, "method": method})

    ours_eff = _read_csv(ours_dir / "table_efficiency_mvtec15.csv")
    crv05_eff = _read_csv(crv05_dir / "table_efficiency_mvtec15.csv")
    baseline_eff = _read_csv(baseline_dir / "table_efficiency_mvtec15.csv")
    eff_rows = _eff_method_rows(ours_eff, "lite_seer_ad_crv035")
    if crv05_eff:
        eff_rows.extend(_eff_method_rows(crv05_eff, "lite_seer_ad_crv05"))
    for row in baseline_eff:
        method = row.get("method", "baseline")
        eff_rows.append({**row, "method": method})

    crv_rows, crv_summary = _crv_weight_rows(ours_main, crv05_main)
    _write(out / "table_main_ours_vs_baselines.csv", main_rows, MAIN_FIELDS)
    _write(out / "table_efficiency_ours_vs_baselines.csv", eff_rows, EFF_FIELDS)
    _write(out / "table_crv_weight_ablation.csv", crv_rows, CRV_FIELDS)
    (out / "crv_weight_decision.json").write_text(json.dumps(crv_summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Exported comparison tables to {out}")


if __name__ == "__main__":
    main()

