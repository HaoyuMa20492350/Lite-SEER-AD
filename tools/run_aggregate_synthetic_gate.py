from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rerun held-out selection with one candidate manifest and synthetic "
            "metrics aggregated across seeds."
        )
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--source-gate-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="7,13,23")
    p.add_argument("--fusion-report", default=None)
    p.add_argument("--materialize", action="store_true")
    p.add_argument("--log-file", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "category", "candidate", "run", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)


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
    source_root = Path(args.source_gate_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in split_csv(args.seeds)]
    source_rows = read_csv(
        source_root / f"seed{seeds[0]}" / "normal_gate_metrics.csv"
    )
    manifest: dict[tuple[str, str], dict[str, str]] = {}
    for row in source_rows:
        key = (row["category"], row["candidate"])
        manifest[key] = {
            "dataset": args.dataset,
            "category": row["category"],
            "candidate": row["candidate"],
            "run": row["run"],
            "source": "original_gate",
        }
    fusion_report = read_json(
        Path(args.fusion_report)
        if args.fusion_report
        else source_root / "fusion_candidate_report.json"
    )
    for row in fusion_report.get("materialized", []):
        key = (str(row["category"]), str(row["fusion"]))
        manifest[key] = {
            "dataset": args.dataset,
            "category": str(row["category"]),
            "candidate": str(row["fusion"]),
            "run": str(row["run"]),
            "source": "normal_calibrated_fusion",
        }
    manifest_rows = [
        manifest[key] for key in sorted(manifest)
    ]
    manifest_path = out_root / "candidate_manifest.csv"
    write_csv(manifest_path, manifest_rows)
    categories = sorted({row["category"] for row in manifest_rows})

    log_handle: TextIO | None = None
    completed = []
    try:
        if args.log_file:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
        for seed in seeds:
            seed_out = out_root / f"seed{seed}"
            command = [
                sys.executable,
                "tools/select_pixel_policy_with_normal_gate.py",
                "--dataset",
                args.dataset,
                "--categories",
                ",".join(categories),
                "--candidate-manifest",
                str(manifest_path),
                "--out",
                str(seed_out),
                "--seed",
                str(seed),
                "--gate-metric",
                "synthetic_normal_utility",
                "--aggregate-synthetic-seeds",
                ",".join(str(value) for value in seeds),
            ]
            if args.materialize:
                command.append("--materialize")
            run(command, log_handle)
            completed.append(
                {"seed": seed, "out": str(seed_out)}
            )
    finally:
        if log_handle is not None:
            log_handle.close()
    report = {
        "dataset": args.dataset,
        "source_gate_root": str(source_root),
        "seeds": seeds,
        "aggregate_synthetic_seeds": seeds,
        "selection_protocol": (
            "normal_plus_synthetic_cross_seed_mean_no_real_anomaly_labels"
        ),
        "uses_real_anomaly_labels_for_selection": False,
        "uses_real_anomaly_masks_for_selection": False,
        "candidate_manifest": str(manifest_path),
        "candidate_rows": len(manifest_rows),
        "runs": completed,
    }
    (out_root / "synthetic_gate_run_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "candidate_rows": len(manifest_rows),
                "runs": len(completed),
                "report": str(
                    out_root / "synthetic_gate_run_report.json"
                ),
            }
            if args.quiet
            else report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
