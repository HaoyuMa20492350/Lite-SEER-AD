from __future__ import annotations

import argparse
import csv
import json
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
]
EFF_FIELDS = ["run", "ablation", "latency_ms_mean", "fps", "nfe_mean", "repaired_area_ratio_mean", "local_region_ratio_mean", "gpu_memory_mb"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export paper-ready CSV tables from run directories.")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="tables")
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


def _run_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        if not (run_dir / "metrics.json").exists() or not (run_dir / "run_args.json").exists() or not (run_dir / "config.yaml").exists():
            continue
        metrics = _load_json(run_dir / "metrics.json")
        efficiency = _load_metric_csv(run_dir / "efficiency.csv")
        args = _load_json(run_dir / "run_args.json")
        cfg = {}
        try:
            import yaml

            cfg_path = run_dir / "config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        except Exception:
            cfg = {}
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


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows = _run_rows(Path(args.runs))
    out = Path(args.out)
    _write(out / "table_main_mvtec.csv", rows, MAIN_FIELDS)
    _write(out / "table_efficiency.csv", rows, EFF_FIELDS)
    _write(out / "table_ablation_hn_sev.csv", rows, MAIN_FIELDS)
    _write(out / "table_ablation_crv.csv", rows, MAIN_FIELDS)
    _write(out / "table_ablation_lc_rds.csv", rows, MAIN_FIELDS)
    print(f"Exported {len(rows)} run rows to {out}")


if __name__ == "__main__":
    main()
