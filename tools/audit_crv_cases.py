from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


RUN_VARIANTS = ["full", "no_crv", "no_sev", "residual_only"]
ARTIFACTS = ["input.png", "reconstruction.png", "residual_heatmap.npz", "repair.png", "mask.png", "roi_log.jsonl", "candidate_roi.png", "verified_roi.png"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit CRV failure cases by aligning full/no_crv/no_sev/residual_only runs.")
    p.add_argument("--runs", default="runs")
    p.add_argument("--prefix", default="mini_mvtec")
    p.add_argument("--categories", default="bottle,capsule")
    p.add_argument("--out-root", default="runs")
    p.add_argument("--limit", type=int, default=16)
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _read_scores(path: Path) -> dict[int, dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return {int(float(row["index"])): row for row in csv.DictReader(f)}


def _load_roi_stats(path: Path, idx: int) -> dict[str, float]:
    if not path.exists():
        return {"sdr_mean": 0.0, "sdr_max": 0.0, "hn_sev_mean": 0.0, "hn_sev_max": 0.0}
    rows = [row for row in json.loads(path.read_text(encoding="utf-8")) if int(row.get("image_index", -1)) == idx]
    if not rows:
        return {"sdr_mean": 0.0, "sdr_max": 0.0, "hn_sev_mean": 0.0, "hn_sev_max": 0.0}
    sdr = [float(row.get("sdr", 0.0)) for row in rows]
    sev = [float(row.get("hn_sev_confidence", 0.0)) for row in rows]
    return {
        "sdr_mean": sum(sdr) / max(1, len(sdr)),
        "sdr_max": max(sdr),
        "hn_sev_mean": sum(sev) / max(1, len(sev)),
        "hn_sev_max": max(sev),
    }


def _copy(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _audit_category(runs_root: Path, prefix: str, category: str, out_root: Path, limit: int) -> None:
    dirs = {variant: runs_root / f"{prefix}_{category}_{variant}" for variant in RUN_VARIANTS}
    missing = [variant for variant, path in dirs.items() if not (path / "scores.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing scores for {category}: {missing}")

    scores = {variant: _read_scores(path / "scores.csv") for variant, path in dirs.items()}
    full_scores = scores["full"]
    no_crv_scores = scores["no_crv"]
    rows: list[dict[str, Any]] = []
    for idx, full_row in full_scores.items():
        if idx not in no_crv_scores:
            continue
        label = int(float(full_row.get("label", 0)))
        full_score = float(full_row.get("image_score", 0.0))
        no_crv_score = float(no_crv_scores[idx].get("image_score", 0.0))
        score_delta = full_score - no_crv_score
        failure_margin = (-score_delta) if label == 1 else score_delta
        if failure_margin <= 0:
            continue
        row = {
            "index": idx,
            "path": full_row.get("path", ""),
            "label": label,
            "full_score": full_score,
            "no_crv_score": no_crv_score,
            "score_delta_full_minus_no_crv": score_delta,
            "failure_margin": failure_margin,
        }
        row.update(_load_roi_stats(dirs["full"] / "roi_budget.json", idx))
        rows.append(row)
    rows.sort(key=lambda row: float(row["failure_margin"]), reverse=True)
    rows = rows[:limit]

    out_dir = out_root / f"audit_crv_{category}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        idx = int(row["index"])
        case_dir = out_dir / f"{idx:05d}"
        row["case_dir"] = str(case_dir)
        for artifact in ARTIFACTS:
            _copy(dirs["full"] / "images" / f"{idx:05d}" / artifact, case_dir / artifact)
        for variant in RUN_VARIANTS:
            _copy(dirs[variant] / "heatmaps" / f"{idx:05d}.png", case_dir / f"{variant}_heatmap.png")
            _copy(dirs[variant] / "scores.csv", out_dir / f"{variant}_scores.csv")
    fields = [
        "index",
        "case_dir",
        "path",
        "label",
        "full_score",
        "no_crv_score",
        "score_delta_full_minus_no_crv",
        "failure_margin",
        "sdr_mean",
        "sdr_max",
        "hn_sev_mean",
        "hn_sev_max",
    ]
    with (out_dir / "case_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} CRV audit cases to {out_dir}")


def main() -> None:
    args = parse_args()
    for category in split_csv(args.categories):
        _audit_category(Path(args.runs), args.prefix, category, Path(args.out_root), args.limit)


if __name__ == "__main__":
    main()
