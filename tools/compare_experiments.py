from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two Lite-SEER-AD experiment table directories and gate expansion.")
    p.add_argument("--base", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: Any) -> float:
    try:
        if value in {"", None}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_float(row.get(key)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _category_passes(evidence_rows: list[dict[str, Any]], module: str) -> int:
    passed = set()
    for row in evidence_rows:
        if row.get("module") == module and str(row.get("passed", "")).lower() == "true":
            passed.add(str(row.get("category", "")))
    return len(passed)


def _summary(table_dir: Path) -> dict[str, Any]:
    main = _read_csv(table_dir / "table_main_mvtec5.csv")
    evidence = _read_csv(table_dir / "evidence_summary.csv")
    return {
        "full_runs": len(main),
        "mean_pixel_auroc": _mean(main, "pixel_auroc"),
        "mean_aupro": _mean(main, "aupro"),
        "mean_pixel_ap": _mean(main, "pixel_ap"),
        "hn_sev_category_passes": _category_passes(evidence, "hn_sev"),
        "crv_category_passes": _category_passes(evidence, "crv"),
        "lc_rds_category_passes": _category_passes(evidence, "lc_rds"),
    }


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base)
    candidate_dir = Path(args.candidate)
    base = _summary(base_dir)
    candidate = _summary(candidate_dir)
    pixel_auroc_delta = candidate["mean_pixel_auroc"] - base["mean_pixel_auroc"]
    aupro_delta = candidate["mean_aupro"] - base["mean_aupro"]
    ready = candidate["hn_sev_category_passes"] >= 4 and candidate["crv_category_passes"] >= 3 and (
        (math.isfinite(pixel_auroc_delta) and pixel_auroc_delta > 0) or (math.isfinite(aupro_delta) and aupro_delta > 0)
    )
    row = {
        "base": str(base_dir),
        "candidate": str(candidate_dir),
        "base_mean_pixel_auroc": base["mean_pixel_auroc"],
        "candidate_mean_pixel_auroc": candidate["mean_pixel_auroc"],
        "pixel_auroc_delta": pixel_auroc_delta,
        "base_mean_aupro": base["mean_aupro"],
        "candidate_mean_aupro": candidate["mean_aupro"],
        "aupro_delta": aupro_delta,
        "candidate_hn_sev_category_passes": candidate["hn_sev_category_passes"],
        "candidate_crv_category_passes": candidate["crv_category_passes"],
        "candidate_lc_rds_category_passes": candidate["lc_rds_category_passes"],
        "ready_for_mvtec15": ready,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    (out.with_suffix(".json")).write_text(json.dumps(_json_safe({"base": base, "candidate": candidate, "comparison": row}), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(_json_safe(row), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
