from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Materialize synthetic-normal evidence and run the label-free policy gate."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--categories", required=True)
    p.add_argument("--candidate", action="append", required=True, help="name=run/template/with/{category}")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", default="7,13,23")
    p.add_argument("--max-normal-images", type=int, default=16)
    p.add_argument("--synthetic-variants", type=int, default=2)
    p.add_argument(
        "--aggregate-selection-seeds",
        action="store_true",
        help="Use mean synthetic metrics across all --seeds for every held-out split.",
    )
    p.add_argument("--canonical-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--materialize", action="store_true")
    p.add_argument("--allow-missing-candidates", action="store_true")
    p.add_argument("--log-file", default=None, help="Append child-process output to this file.")
    p.add_argument("--quiet", action="store_true", help="Print only a compact completion summary.")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_candidates(values: list[str]) -> list[tuple[str, str]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Candidate must be name=template: {value}")
        name, template = value.split("=", 1)
        result.append((name.strip(), template.strip()))
    return result


def run(command: list[str], log_handle: TextIO | None = None) -> None:
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


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    categories = split_csv(args.categories)
    seeds = [int(value) for value in split_csv(args.seeds)]
    candidates = parse_candidates(args.candidate)
    out = Path(args.out)
    report_path = out / "synthetic_gate_run_report.json"
    report: dict[str, object] = {
        "dataset": args.dataset,
        "categories": categories,
        "seeds": seeds,
        "selection_protocol": "normal_plus_synthetic_no_real_anomaly_labels",
        "uses_real_anomaly_labels_for_selection": False,
        "uses_real_anomaly_masks_for_selection": False,
        "runs": [],
        "missing_candidates": [],
    }

    log_handle: TextIO | None = None
    try:
        if args.log_file:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")

        write_report(report_path, report)
        for seed in seeds:
            metrics_name = f"synthetic_validation_seed{seed}_metrics.json"
            for category in categories:
                for name, template in candidates:
                    run_dir = Path(template.format(category=category))
                    if not (run_dir / "predictions.npz").exists():
                        report["missing_candidates"].append(
                            {"seed": seed, "category": category, "candidate": name, "run": str(run_dir)}
                        )
                        if not args.allow_missing_candidates:
                            write_report(report_path, report)
                            raise FileNotFoundError(f"Missing candidate run: {run_dir}")
                        continue
                    artifact = run_dir / f"synthetic_validation_seed{seed}.npz"
                    metrics = run_dir / metrics_name
                    if not (args.resume and metrics.exists()):
                        command = [
                            sys.executable,
                            "tools/materialize_synthetic_normal_validation.py",
                            "--candidate-run-dir",
                            str(run_dir),
                            "--out",
                            str(artifact),
                            "--max-normal-images",
                            str(args.max_normal_images),
                            "--synthetic-variants",
                            str(args.synthetic_variants),
                            "--canonical-size",
                            str(args.canonical_size),
                            "--batch-size",
                            str(args.batch_size),
                            "--seed",
                            str(seed),
                        ]
                        if args.device:
                            command.extend(["--device", args.device])
                        run(command, log_handle)

            seed_out = out / f"seed{seed}"
            command = [
                sys.executable,
                "tools/select_pixel_policy_with_normal_gate.py",
                "--dataset",
                args.dataset,
                "--categories",
                args.categories,
                "--out",
                str(seed_out),
                "--seed",
                str(seed),
                "--gate-metric",
                "synthetic_normal_utility",
                "--synthetic-metrics-name",
                metrics_name,
            ]
            for name, template in candidates:
                command.extend(["--candidate", f"{name}={template}"])
            if args.allow_missing_candidates:
                command.append("--allow-missing-synthetic")
            if args.aggregate_selection_seeds:
                command.extend(
                    [
                        "--aggregate-synthetic-seeds",
                        ",".join(str(seed) for seed in seeds),
                    ]
                )
            if args.materialize:
                command.append("--materialize")
            run(command, log_handle)
            report["runs"].append({"seed": seed, "out": str(seed_out)})
            write_report(report_path, report)
    finally:
        if log_handle is not None:
            log_handle.close()

    if args.quiet:
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "seeds_completed": len(report["runs"]),
                    "missing_candidates": len(report["missing_candidates"]),
                    "report": str(report_path),
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
