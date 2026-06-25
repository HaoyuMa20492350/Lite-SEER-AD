from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MVTEC15_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
REQUIRED_FILES = [
    "predictions.npz",
    "metrics.json",
    "metrics.csv",
    "efficiency.csv",
    "scores.csv",
    "roi_budget.json",
    "crv_score_drop.npy",
    "pareto.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an MVTec15 Lite-SEER-AD setting by reusing existing checkpoints.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--source-prefix", default="mvtec15_ours")
    p.add_argument("--run-prefix", default="mvtec15_ours_crv05")
    p.add_argument("--categories", default=",".join(MVTEC15_CATEGORIES))
    p.add_argument("--ablations", default="full,no_crv")
    p.add_argument("--crv-weight", type=float, default=0.5)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--tables-out", default="tables/mvtec15_ours_crv05")
    p.add_argument("--exclude-prefix", default=None, help="Comma-separated additional run prefixes to exclude when summarizing.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def py(script: str, *args: Any) -> list[str]:
    return [sys.executable, script, *[str(a) for a in args]]


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_if_needed(cmd: list[str], output: Path, *, resume: bool, dry_run: bool) -> None:
    if resume and output.exists():
        print(f"SKIP existing {output}", flush=True)
        return
    run_command(cmd, dry_run=dry_run)


def check_schema(run_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    image_count = len(list((run_dir / "images").glob("*"))) if (run_dir / "images").exists() else 0
    return {"run": run_dir.name, "ok": not missing and image_count > 0, "missing": missing, "image_count": image_count}


def write_mvtec15_aliases(tables_dir: Path) -> None:
    aliases = {
        "table_main_mvtec5.csv": "table_main_mvtec15.csv",
        "table_efficiency_mvtec5.csv": "table_efficiency_mvtec15.csv",
    }
    for src_name, dst_name in aliases.items():
        src = tables_dir / src_name
        if src.exists():
            shutil.copyfile(src, tables_dir / dst_name)


def main() -> None:
    args = parse_args()
    report: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    categories = split_csv(args.categories)
    ablations = split_csv(args.ablations)
    for category in categories:
        source_dir = REPO_ROOT / "runs" / f"{args.source_prefix}_{category}_full"
        ckpt = source_dir / "diffusion.pt"
        sev_ckpt = source_dir / "hn_sev.pt"
        scheduler_ckpt = source_dir / "lc_rds.pt"
        try:
            for required in [ckpt, sev_ckpt, scheduler_ckpt]:
                if not required.exists():
                    raise FileNotFoundError(required)
            for ablation in ablations:
                run_name = f"{args.run_prefix}_{category}_{ablation}"
                run_dir = REPO_ROOT / "runs" / run_name
                infer_cmd = py(
                    "infer.py",
                    "--config",
                    args.config,
                    "--category",
                    category,
                    "--ckpt",
                    ckpt,
                    "--sev-checkpoint",
                    sev_ckpt,
                    "--scheduler-checkpoint",
                    scheduler_ckpt,
                    "--image-size",
                    args.image_size,
                    "--seed",
                    args.seed,
                    "--device",
                    args.device,
                    "--crv-weight",
                    args.crv_weight,
                    "--run-name",
                    run_name,
                    "--ablation",
                    ablation,
                )
                run_if_needed(infer_cmd, run_dir / "predictions.npz", resume=args.resume, dry_run=args.dry_run)
                run_command(py("evaluate.py", "--pred_dir", run_dir, "--out", run_dir / "eval_metrics.json"), dry_run=args.dry_run)
                report.append(check_schema(run_dir))
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            failures.append({"category": category, "error": str(exc)})
            if args.fail_fast:
                raise

    report_path = REPO_ROOT / "runs" / f"{args.run_prefix}_report.json"
    if not args.dry_run:
        report_path.write_text(json.dumps({"schema": report, "failures": failures}, indent=2), encoding="utf-8")
    summary_cmd = py("tools/summarize_evidence.py", "--runs", "runs", "--prefix", args.run_prefix, "--out", args.tables_out)
    excluded = [f"{args.run_prefix}_smoke"]
    excluded.extend(split_csv(args.exclude_prefix or ""))
    summary_cmd.extend(["--exclude-prefix", ",".join(excluded)])
    run_command(summary_cmd, dry_run=args.dry_run)
    if not args.dry_run:
        write_mvtec15_aliases(REPO_ROOT / args.tables_out)
    print(f"Finished setting run. Report: {report_path}. Failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
