from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["bottle", "cable", "capsule", "metal_nut", "zipper"]
ALL_MVTEC_CATEGORIES = [
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
FULL_ABLATIONS = [
    "residual_only",
    "no_sev",
    "no_crv",
    "repair_visualization_only",
    "fixed10",
    "fixed25",
    "rule_brds",
    "learned_lc_rds",
]
SMOKE_ABLATIONS = ["no_sev", "no_crv", "fixed10"]
MVTEC15_ABLATIONS = ["residual_only", "no_sev", "no_crv", "rule_brds"]
REQUIRED_RUN_FILES = [
    "config.yaml",
    "run_args.json",
    "environment.txt",
    "git_hash.txt",
    "predictions.npz",
    "metrics.json",
    "metrics.csv",
    "efficiency.csv",
    "scores.csv",
    "roi_budget.json",
    "crv_score_drop.npy",
    "pareto.csv",
]
REQUIRED_IMAGE_FILES = [
    "input.png",
    "reconstruction.png",
    "residual.png",
    "residual_heatmap.npz",
    "candidate_roi.png",
    "verified_roi.png",
    "mask.png",
    "ground_truth.png",
    "final_heatmap.png",
    "final_mask.png",
    "repair.png",
    "roi_log.jsonl",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the MVTec 5-class Lite-SEER-AD mini experiment.")
    p.add_argument("--profile", choices=["mini5", "long5", "mvtec15_ours"], default="mini5")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--categories", default=",".join(CATEGORIES))
    p.add_argument("--run-prefix", default="mini_mvtec")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-samples", default="64")
    p.add_argument("--diffusion-epochs", type=int, default=3)
    p.add_argument("--sev-epochs", type=int, default=2)
    p.add_argument("--scheduler-epochs", type=int, default=10)
    p.add_argument("--scheduler-samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sev-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--crv-weight", type=float, default=0.35)
    p.add_argument("--ablations", default=None, help="Comma-separated ablations. Defaults to the full mini set, or core smoke set with --smoke.")
    p.add_argument("--tables-out", default=None)
    p.add_argument("--qualitative-limit", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="Use bottle, image_size=64, max_samples=8, one-epoch training, and core ablations.")
    p.add_argument("--fallback", action="store_true", help="Use the lighter long5 fallback: max_samples=128, diffusion_epochs=10, sev_epochs=3.")
    p.add_argument("--resume", action="store_true", help="Skip stages whose expected output already exists.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def max_sample_args(value: str | int | None) -> list[str]:
    if value is None:
        return []
    text = str(value).strip().lower()
    if text in {"", "all", "none", "null"}:
        return []
    return ["--max-samples", str(int(float(text)))]


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    try:
        print(" ".join(cmd), flush=True)
    except OSError:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        raise SystemExit(0)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_if_needed(cmd: list[str], output: Path, *, resume: bool, dry_run: bool) -> None:
    if resume and output.exists():
        print(f"SKIP existing {output}", flush=True)
        return
    run_command(cmd, dry_run=dry_run)


def py(script: str, *args: str | int) -> list[str]:
    return [sys.executable, script, *[str(a) for a in args]]


def check_schema(run_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).exists()]
    image_dirs = sorted((run_dir / "images").glob("*"))
    image_missing: list[str] = []
    for image_dir in image_dirs[:1]:
        image_missing.extend([f"{image_dir.name}/{name}" for name in REQUIRED_IMAGE_FILES if not (image_dir / name).exists()])
    return {
        "run": run_dir.name,
        "ok": not missing and bool(image_dirs) and not image_missing,
        "missing": missing,
        "image_count": len(image_dirs),
        "image_missing": image_missing,
    }


def write_mvtec15_table_aliases(tables_dir: Path) -> None:
    aliases = {
        "table_main_mvtec5.csv": "table_main_mvtec15.csv",
        "table_efficiency_mvtec5.csv": "table_efficiency_mvtec15.csv",
    }
    for src_name, dst_name in aliases.items():
        src = tables_dir / src_name
        dst = tables_dir / dst_name
        if src.exists():
            shutil.copyfile(src, dst)


def run_category(category: str, args: argparse.Namespace, ablations: list[str], report: list[dict[str, Any]]) -> None:
    base_run = f"{args.run_prefix}_{category}_full"
    base_dir = REPO_ROOT / "runs" / base_run
    ckpt = base_dir / "diffusion.pt"
    hn_dir = base_dir / "hard_negatives"
    sev_ckpt = base_dir / "hn_sev.pt"
    scheduler_ckpt = base_dir / "lc_rds.pt"

    common = ["--config", args.config, "--category", category, "--image-size", args.image_size, *max_sample_args(args.max_samples), "--seed", args.seed, "--device", args.device]
    run_if_needed(
        py("train_diffusion.py", *common, "--epochs", args.diffusion_epochs, "--batch-size", args.batch_size, "--run-name", base_run),
        ckpt,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    run_if_needed(
        py("mine_hard_negatives.py", *common, "--checkpoint", ckpt, "--run-name", base_run),
        hn_dir,
        resume=args.resume and hn_dir.exists() and any(hn_dir.iterdir()),
        dry_run=args.dry_run,
    )
    run_if_needed(
        py(
            "train_hn_sev.py",
            *common,
            "--checkpoint",
            ckpt,
            "--hard-negative-dir",
            hn_dir,
            "--epochs",
            args.sev_epochs,
            "--batch-size",
            args.sev_batch_size,
            "--run-name",
            base_run,
        ),
        sev_ckpt,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    run_if_needed(
        py("train_lc_rds.py", "--config", args.config, "--samples", args.scheduler_samples, "--epochs", args.scheduler_epochs, "--run-name", base_run, "--seed", args.seed, "--device", args.device),
        scheduler_ckpt,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    infer_common = [
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
        *max_sample_args(args.max_samples),
        "--seed",
        args.seed,
        "--device",
        args.device,
        "--crv-weight",
        args.crv_weight,
    ]
    run_if_needed(
        py("infer.py", *infer_common, "--run-name", base_run, "--ablation", "full"),
        base_dir / "predictions.npz",
        resume=args.resume,
        dry_run=args.dry_run,
    )
    run_command(py("evaluate.py", "--pred_dir", base_dir, "--out", base_dir / "eval_metrics.json"), dry_run=args.dry_run)
    run_command(py("tools/export_figures.py", "--run-dir", base_dir, "--out", base_dir / "qualitative_cases", "--limit", args.qualitative_limit), dry_run=args.dry_run)
    report.append(check_schema(base_dir))

    for ablation in ablations:
        run_name = f"{args.run_prefix}_{category}_{ablation}"
        run_dir = REPO_ROOT / "runs" / run_name
        run_if_needed(
            py("infer.py", *infer_common, "--run-name", run_name, "--ablation", ablation),
            run_dir / "predictions.npz",
            resume=args.resume,
            dry_run=args.dry_run,
        )
        run_command(py("evaluate.py", "--pred_dir", run_dir, "--out", run_dir / "eval_metrics.json"), dry_run=args.dry_run)
        report.append(check_schema(run_dir))


def main() -> None:
    args = parse_args()
    default_tables = {
        "mini5": "tables/mini_mvtec",
        "long5": "tables/mini_mvtec_long",
        "mvtec15_ours": "tables/mvtec15_ours",
    }
    if args.profile == "long5":
        args.categories = ",".join(CATEGORIES)
        if args.run_prefix == "mini_mvtec":
            args.run_prefix = "mini_mvtec_long"
        args.max_samples = "128" if args.fallback else "all"
        args.diffusion_epochs = 10 if args.fallback else 20
        args.sev_epochs = 3 if args.fallback else 5
        args.scheduler_epochs = 20
    elif args.profile == "mvtec15_ours":
        args.categories = ",".join(ALL_MVTEC_CATEGORIES)
        if args.run_prefix == "mini_mvtec":
            args.run_prefix = "mvtec15_ours"
        if str(args.max_samples).strip().lower() == "64":
            args.max_samples = "all"
    if args.tables_out is None:
        args.tables_out = default_tables[args.profile]
    if args.smoke:
        args.categories = "bottle"
        args.run_prefix = f"{args.run_prefix}_smoke"
        args.image_size = 64
        args.max_samples = "8"
        args.diffusion_epochs = 1
        args.sev_epochs = 1
        args.scheduler_epochs = min(args.scheduler_epochs, 3)
        args.scheduler_samples = min(args.scheduler_samples, 128)
    categories = split_csv(args.categories)
    if args.ablations:
        ablations = split_csv(args.ablations)
    elif args.smoke:
        ablations = SMOKE_ABLATIONS
    elif args.profile == "mvtec15_ours":
        ablations = MVTEC15_ABLATIONS
    else:
        ablations = FULL_ABLATIONS

    report: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for category in categories:
        try:
            run_command(py("tools/index_datasets.py", "--config", args.config, "--category", category, "--out", f"runs/index_mvtec_{category}.csv"), dry_run=args.dry_run)
            run_category(category, args, ablations, report)
        except subprocess.CalledProcessError as exc:
            failures.append({"category": category, "error": str(exc)})
            if args.fail_fast:
                raise

    report_path = REPO_ROOT / "runs" / f"{args.run_prefix}_report.json"
    if not args.dry_run:
        report_path.write_text(json.dumps({"schema": report, "failures": failures}, indent=2), encoding="utf-8")
    summary_cmd = py("tools/summarize_evidence.py", "--runs", "runs", "--prefix", args.run_prefix, "--out", args.tables_out)
    if not args.smoke:
        summary_cmd.extend(["--exclude-prefix", f"{args.run_prefix}_smoke"])
    run_command(summary_cmd, dry_run=args.dry_run)
    if args.profile == "mvtec15_ours" and not args.dry_run:
        write_mvtec15_table_aliases(REPO_ROOT / args.tables_out)
    print(f"Finished mini experiment. Report: {report_path}. Failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
