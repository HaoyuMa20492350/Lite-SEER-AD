from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SPECS = {
    "mvtec": {
        "config": "configs/mvtec.yaml",
        "run_prefix": "fulltest_mvtec15",
        "model_prefix": "feature_mvtec15",
        "source_prefix": "feature_mvtec15",
        "tables": "tables/fulltest_mvtec15",
        "inference_only": False,
        "refresh_hn_sev": True,
    },
    "visa": {
        "config": "configs/visa.yaml",
        "run_prefix": "fulltest_visa",
        "model_prefix": "fulltest_visa",
        "source_prefix": None,
        "tables": "tables/fulltest_visa",
        "inference_only": False,
        "refresh_hn_sev": False,
    },
    "mpdd": {
        "config": "configs/mpdd.yaml",
        "run_prefix": "fulltest_mpdd",
        "model_prefix": "feature_mpdd",
        "source_prefix": "feature_mpdd",
        "tables": "tables/fulltest_mpdd",
        "inference_only": False,
        "refresh_hn_sev": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete 33-category official-test Lite-SEER-AD suite."
    )
    parser.add_argument(
        "--datasets",
        default="mvtec,visa,mpdd",
        help="Comma-separated subset of mvtec, visa, mpdd.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command_for(name: str, args: argparse.Namespace) -> list[str]:
    spec = SPECS[name]
    command = [
        sys.executable,
        "tools/run_feature_first_mvtec5.py",
        "--config",
        spec["config"],
        "--categories",
        "auto",
        "--run-prefix",
        spec["run_prefix"],
        "--model-prefix",
        spec["model_prefix"],
        "--tables-out",
        spec["tables"],
        "--image-size",
        "128",
        "--max-samples",
        "all",
        "--freeze-crv-weight",
        "--skip-figures",
        "--device",
        args.device,
    ]
    if spec["inference_only"]:
        command.append("--inference-only")
    if spec["refresh_hn_sev"]:
        command.append("--reuse-base-checkpoints")
    if args.resume or spec["refresh_hn_sev"]:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    return command


def run(command: list[str], dry_run: bool = False) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    selected = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(selected) - set(SPECS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    run(
        [sys.executable, "tools/audit_dataset_protocol.py"],
        dry_run=args.dry_run,
    )
    for name in selected:
        spec = SPECS[name]
        if spec["refresh_hn_sev"]:
            target_prefix = spec["run_prefix"]
            run(
                [
                    sys.executable,
                    "tools/prepare_full_test_models.py",
                    "--config",
                    spec["config"],
                    "--source-prefix",
                    spec["source_prefix"],
                    "--target-prefix",
                    target_prefix,
                ],
                dry_run=args.dry_run,
            )
            spec["model_prefix"] = target_prefix
        run(command_for(name, args), dry_run=args.dry_run)
    if set(selected) == set(SPECS):
        threshold_command = [
            sys.executable,
            "tools/freeze_full_test_thresholds.py",
            "--datasets",
            ",".join(selected),
            "--device",
            args.device,
        ]
        if args.resume:
            threshold_command.append("--resume")
        run(threshold_command, dry_run=args.dry_run)
        run(
            [sys.executable, "tools/audit_full_test_coverage.py"],
            dry_run=args.dry_run,
        )
        run(
            [sys.executable, "tools/export_full_test_paper_tables.py"],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
