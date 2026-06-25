from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.mvtec_ad2_pipeline import (
    audit_pipeline,
    private_command,
    public_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or execute the frozen MVTec AD 2 public/private pipeline."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("status", "public", "private", "all"),
        default="status",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required for public/private execution. Status is always read-only.",
    )
    parser.add_argument("--config", default="configs/mvtec_ad2.yaml")
    parser.add_argument("--run-prefix", default="mvtec_ad2_feature_first")
    parser.add_argument(
        "--public-out",
        default="tables/mvtec_ad2_feature_first",
    )
    parser.add_argument(
        "--public-log",
        default="tables/mvtec_ad2_feature_first/experiment.log",
    )
    parser.add_argument("--private-seed", type=int, default=7)
    parser.add_argument(
        "--submission-out",
        default="submissions/mvtec_ad2_seed7_model256",
    )
    parser.add_argument(
        "--metadata-out",
        default="submissions/mvtec_ad2_seed7_model256_metadata",
    )
    parser.add_argument(
        "--archive-out",
        default="submissions/mvtec_ad2_seed7_model256.tar.gz",
    )
    parser.add_argument(
        "--submission-resolution",
        choices=("native", "model"),
        default="model",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--report",
        default="tables/mvtec_ad2_feature_first/pipeline_status.json",
    )
    return parser.parse_args()


def status(args: argparse.Namespace) -> dict:
    return audit_pipeline(
        REPO_ROOT,
        run_prefix=args.run_prefix,
        public_table_dir=Path(args.public_out),
        submission_dir=Path(args.submission_out),
        metadata_dir=Path(args.metadata_out),
        archive_path=Path(args.archive_out),
        private_seed=args.private_seed,
    )


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def write_report(path: str, report: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    before = status(args)
    commands = []
    if args.phase in {"public", "all"}:
        commands.append(
            public_command(
                sys.executable,
                config=args.config,
                run_prefix=args.run_prefix,
                out=args.public_out,
                log_file=args.public_log,
                device=args.device,
            )
        )
    if args.phase in {"private", "all"}:
        commands.append(
            private_command(
                sys.executable,
                config=args.config,
                run_prefix=args.run_prefix,
                private_seed=args.private_seed,
                out=args.submission_out,
                metadata_out=args.metadata_out,
                archive_out=args.archive_out,
                device=args.device,
                submission_resolution=args.submission_resolution,
            )
        )
    report = {
        "phase": args.phase,
        "execute": bool(args.execute),
        "before": before,
        "commands": commands,
    }
    if args.phase != "status" and not args.execute:
        report["status"] = "dry_run_requires_execute"
        write_report(args.report, report)
        print(json.dumps(report, indent=2))
        return
    if args.phase != "status":
        if not before["dataset_ready"]:
            report["status"] = "blocked_dataset_not_ready"
            write_report(args.report, report)
            print(json.dumps(report, indent=2))
            raise SystemExit(2)
        if not before["official_checker_ready"]:
            report["status"] = "blocked_official_checker_missing"
            write_report(args.report, report)
            print(json.dumps(report, indent=2))
            raise SystemExit(2)
        if args.phase == "private" and not before["public_complete"]:
            report["status"] = "blocked_public_runs_incomplete"
            write_report(args.report, report)
            print(json.dumps(report, indent=2))
            raise SystemExit(2)
        for index, command in enumerate(commands):
            if (
                args.phase == "all"
                and index == 1
                and not status(args)["public_complete"]
            ):
                raise RuntimeError(
                    "Refusing private export before all 24 public runs complete."
                )
            run_command(command)
        report["status"] = "executed"
    else:
        report["status"] = "complete" if before["pipeline_complete"] else "ready_or_blocked"
    report["after"] = status(args)
    write_report(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
