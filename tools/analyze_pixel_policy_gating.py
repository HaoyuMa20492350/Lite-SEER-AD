from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze train-normal pixel-policy gating features against a reference table.")
    p.add_argument("--reference-table", required=True)
    p.add_argument("--reference-ablation", default="feature_tuned_crv")
    p.add_argument("--policy-table", required=True)
    p.add_argument("--policy-ablation", default="feature_pixel_policy")
    p.add_argument("--policy-run-root", default="runs")
    p.add_argument("--out", required=True)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, Any], key: str) -> float:
    try:
        value = row.get(key)
        if value in {"", None}:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def highpass_energy(img: np.ndarray, sigma: float = 9.0) -> float:
    k = max(3, int(round(sigma * 6)) | 1)
    blur = cv2.GaussianBlur(img.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)
    hp = np.maximum(img.astype(np.float32) - blur, 0.0)
    denom = float(np.mean(np.abs(img))) + 1e-6
    return float(np.mean(hp) / denom)


def stat_features(stats_path: Path) -> dict[str, float]:
    data = np.load(stats_path)
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)
    iqr = np.asarray(data["iqr"], dtype=np.float32)
    median = np.asarray(data["median"], dtype=np.float32)
    eps = 1e-6
    return {
        "normal_mean_avg": float(np.mean(mean)),
        "normal_mean_spatial_cv": float(np.std(mean) / (np.mean(np.abs(mean)) + eps)),
        "normal_mean_range_ratio": float((np.max(mean) - np.min(mean)) / (np.mean(np.abs(mean)) + eps)),
        "normal_mean_highpass9": highpass_energy(mean, 9.0),
        "normal_std_avg": float(np.mean(std)),
        "normal_std_spatial_cv": float(np.std(std) / (np.mean(np.abs(std)) + eps)),
        "normal_iqr_avg": float(np.mean(iqr)),
        "normal_iqr_spatial_cv": float(np.std(iqr) / (np.mean(np.abs(iqr)) + eps)),
        "normal_median_avg": float(np.mean(median)),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_rows = [r for r in read_csv(Path(args.reference_table)) if r.get("ablation") == args.reference_ablation]
    pol_rows = [r for r in read_csv(Path(args.policy_table)) if r.get("ablation") == args.policy_ablation]
    ref_by_cat = {r.get("category", ""): r for r in ref_rows}
    rows: list[dict[str, Any]] = []
    for policy in pol_rows:
        category = str(policy.get("category", ""))
        ref = ref_by_cat.get(category)
        if not ref:
            continue
        run_dir = Path(args.policy_run_root) / str(policy.get("run", ""))
        stats_path = run_dir / "normal_pixel_stats.npz"
        features = stat_features(stats_path) if stats_path.exists() else {}
        row: dict[str, Any] = {
            "category": category,
            "policy_run": policy.get("run", ""),
            "delta_pixel_auroc": to_float(policy, "pixel_auroc") - to_float(ref, "pixel_auroc"),
            "delta_aupro": to_float(policy, "aupro") - to_float(ref, "aupro"),
            "delta_pixel_ap": to_float(policy, "pixel_ap") - to_float(ref, "pixel_ap"),
            "delta_dice": to_float(policy, "dice") - to_float(ref, "dice"),
            "ref_pixel_ap": to_float(ref, "pixel_ap"),
            "policy_pixel_ap": to_float(policy, "pixel_ap"),
            "ref_dice": to_float(ref, "dice"),
            "policy_dice": to_float(policy, "dice"),
        }
        row.update(features)
        rows.append(row)

    fields = sorted({key for row in rows for key in row.keys()})
    with (out / "pixel_policy_gating_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    numeric_fields = [field for field in fields if field.startswith("normal_")]
    summary: dict[str, Any] = {"rows": len(rows), "top_by_delta_pixel_ap": []}
    for row in sorted(rows, key=lambda r: float(r["delta_pixel_ap"]), reverse=True)[:10]:
        summary["top_by_delta_pixel_ap"].append(
            {
                "category": row["category"],
                "delta_pixel_ap": row["delta_pixel_ap"],
                "delta_dice": row["delta_dice"],
                **{field: row.get(field) for field in numeric_fields},
            }
        )
    (out / "pixel_policy_gating_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
