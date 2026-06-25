from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Materialize normal-calibrated heatmap-fusion candidates for every "
            "category where both source candidates are available."
        )
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--gate-root", required=True)
    p.add_argument("--categories", default="all")
    p.add_argument(
        "--fusion",
        action="append",
        required=True,
        help="name=source_a,source_b,weight_a",
    )
    p.add_argument("--out-root", default="runs")
    p.add_argument("--calibration-seed", type=int, default=7)
    p.add_argument("--synthetic-seeds", default="7,13,23")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-file", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_fusions(
    values: list[str],
) -> list[tuple[str, str, str, float]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Fusion must be name=a,b,weight: {value}")
        name, spec = value.split("=", 1)
        parts = split_csv(spec)
        if len(parts) != 3:
            raise ValueError(f"Fusion must be name=a,b,weight: {value}")
        result.append((name.strip(), parts[0], parts[1], float(parts[2])))
    return result


def run(command: list[str], log_handle: TextIO | None) -> None:
    if log_handle is not None:
        log_handle.write(f"\n$ {shlex.join(command)}\n")
        log_handle.flush()
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT if log_handle is not None else None,
    )


def main() -> None:
    args = parse_args()
    gate_root = Path(args.gate_root)
    seed_rows = read_csv(
        gate_root
        / f"seed{args.calibration_seed}"
        / "normal_gate_metrics.csv"
    )
    available_categories = sorted({row["category"] for row in seed_rows})
    categories = (
        available_categories
        if args.categories == "all"
        else split_csv(args.categories)
    )
    fusions = parse_fusions(args.fusion)
    out_root = Path(args.out_root)
    report = {
        "dataset": args.dataset,
        "calibration_seed": args.calibration_seed,
        "synthetic_seeds": split_csv(args.synthetic_seeds),
        "fusions": [],
        "materialized": [],
        "missing_sources": [],
    }
    log_handle: TextIO | None = None
    try:
        if args.log_file:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
        for name, source_a, source_b, weight_a in fusions:
            report["fusions"].append(
                {
                    "name": name,
                    "source_a": source_a,
                    "source_b": source_b,
                    "weight_a": weight_a,
                }
            )
            for category in categories:
                source_runs = {
                    row["candidate"]: Path(row["run"])
                    for row in seed_rows
                    if row["category"] == category
                }
                if source_a not in source_runs or source_b not in source_runs:
                    report["missing_sources"].append(
                        {
                            "category": category,
                            "fusion": name,
                            "source_a_available": source_a in source_runs,
                            "source_b_available": source_b in source_runs,
                        }
                    )
                    continue
                out_dir = (
                    out_root
                    / f"feature_fusion_{name}_{args.dataset}_{category}"
                )
                expected = [
                    out_dir / "predictions.npz",
                    *[
                        out_dir
                        / f"synthetic_validation_seed{seed}_metrics.json"
                        for seed in split_csv(args.synthetic_seeds)
                    ],
                ]
                if not (args.resume and all(path.exists() for path in expected)):
                    run(
                        [
                            sys.executable,
                            "tools/materialize_heatmap_fusion_candidate.py",
                            "--source-a",
                            str(source_runs[source_a]),
                            "--source-b",
                            str(source_runs[source_b]),
                            "--out",
                            str(out_dir),
                            "--weight-a",
                            str(weight_a),
                            "--calibration-seed",
                            str(args.calibration_seed),
                            "--synthetic-seeds",
                            args.synthetic_seeds,
                            "--overwrite",
                        ],
                        log_handle,
                    )
                report["materialized"].append(
                    {
                        "category": category,
                        "fusion": name,
                        "run": str(out_dir),
                    }
                )
    finally:
        if log_handle is not None:
            log_handle.close()
    report_path = gate_root / "fusion_candidate_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "dataset": args.dataset,
        "materialized": len(report["materialized"]),
        "missing_sources": len(report["missing_sources"]),
        "report": str(report_path),
    }
    print(json.dumps(summary if args.quiet else report, indent=2))


if __name__ == "__main__":
    main()
