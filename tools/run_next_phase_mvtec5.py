from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ["bottle", "cable", "capsule", "metal_nut", "zipper"]
FULL_ABLATIONS = [
    "residual_only",
    "no_sev",
    "synthetic_only_sev",
    "no_prototype",
    "no_crv",
    "repair_visualization_only",
    "fixed10",
    "fixed25",
    "rule_brds",
    "learned_lc_rds",
]
SMOKE_ABLATIONS = [
    "residual_only",
    "no_sev",
    "synthetic_only_sev",
    "no_prototype",
    "no_crv",
    "fixed10",
    "rule_brds",
]
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
    "roi_budget.jsonl",
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
    "ground_truth.png",
    "final_heatmap.png",
    "final_mask.png",
    "repair.png",
    "roi_log.jsonl",
]

COMMAND_LOG: list[list[str]] = []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the gated Lite-SEER-AD v2 MVTec5 evidence loop.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    p.add_argument("--run-prefix", default="next_mvtec5")
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
    p.add_argument("--reconstruction-steps", default="1,5,10")
    p.add_argument("--crv-weights", default="0.25,0.35,0.5,1.0")
    p.add_argument("--crv-weight", type=float, default=0.35)
    p.add_argument("--tables-out", default="tables/next_mvtec5")
    p.add_argument("--qualitative-limit", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def split_ints(value: str) -> list[int]:
    return [int(float(v.strip())) for v in value.split(",") if v.strip()]


def max_sample_args(value: str | int | None) -> list[str]:
    if value is None:
        return []
    text = str(value).strip().lower()
    if text in {"", "all", "none", "null"}:
        return []
    return ["--max-samples", str(int(float(text)))]


def py(script: str, *args: str | int | float | Path) -> list[str]:
    return [sys.executable, script, *[str(arg) for arg in args]]


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    COMMAND_LOG.append([str(part) for part in cmd])
    print(" ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_if_needed(cmd: list[str], output: Path, *, resume: bool, dry_run: bool) -> None:
    if resume and output.exists():
        print(f"SKIP existing {output}", flush=True)
        return
    run_command(cmd, dry_run=dry_run)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except Exception:
        return default


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _quality(metrics: dict[str, Any]) -> float:
    vals = [_safe_float(metrics.get("pixel_ap")), _safe_float(metrics.get("dice")), _safe_float(metrics.get("aupro"))]
    vals = [v for v in vals if _finite(v)]
    if not vals:
        vals = [_safe_float(metrics.get("aupro_proxy"))]
        vals = [v for v in vals if _finite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(value) for value in obj]
    return obj


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_schema(run_dir: Path, qualitative_limit: int) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).exists()]
    image_dirs = sorted((run_dir / "images").glob("*"))
    image_missing: list[str] = []
    case_limit = min(max(1, qualitative_limit), len(image_dirs))
    for image_dir in image_dirs[:case_limit]:
        for name in REQUIRED_IMAGE_FILES:
            if not (image_dir / name).exists():
                image_missing.append(f"{image_dir.name}/{name}")
    qualitative_dirs = sorted((run_dir / "qualitative_cases").glob("*"))
    return {
        "run": run_dir.name,
        "ok": not missing and not image_missing and len(image_dirs) > 0 and len(qualitative_dirs) >= min(qualitative_limit, len(image_dirs)),
        "missing": missing,
        "image_count": len(image_dirs),
        "qualitative_case_count": len(qualitative_dirs),
        "image_missing": image_missing,
    }


def common_data_args(args: argparse.Namespace, category: str) -> list[str]:
    return [
        "--config",
        args.config,
        "--category",
        category,
        "--image-size",
        args.image_size,
        *max_sample_args(args.max_samples),
        "--seed",
        args.seed,
        "--device",
        args.device,
    ]


def infer_cmd(
    args: argparse.Namespace,
    *,
    category: str,
    checkpoint: Path,
    run_name: str,
    ablation: str,
    reconstruction_steps: int,
    sev_checkpoint: Path | None = None,
    scheduler_checkpoint: Path | None = None,
) -> list[str]:
    cmd = py(
        "infer.py",
        "--config",
        args.config,
        "--category",
        category,
        "--ckpt",
        checkpoint,
        "--image-size",
        args.image_size,
        *max_sample_args(args.max_samples),
        "--run-name",
        run_name,
        "--ablation",
        ablation,
        "--crv-weight",
        args.crv_weight,
        "--reconstruction-steps",
        reconstruction_steps,
        "--seed",
        args.seed,
        "--device",
        args.device,
    )
    if sev_checkpoint is not None:
        cmd.extend(["--sev-checkpoint", str(sev_checkpoint)])
    if scheduler_checkpoint is not None:
        cmd.extend(["--scheduler-checkpoint", str(scheduler_checkpoint)])
    return cmd


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> None:
    run_command(py("evaluate.py", "--pred_dir", run_dir, "--out", run_dir / "eval_metrics.json"), dry_run=args.dry_run)


def export_run(run_dir: Path, args: argparse.Namespace) -> None:
    run_command(py("tools/export_figures.py", "--run-dir", run_dir, "--out", run_dir / "qualitative_cases", "--limit", args.qualitative_limit), dry_run=args.dry_run)


def run_reconstruction_sweep(
    category: str,
    args: argparse.Namespace,
    checkpoint: Path,
    sweep_rows: list[dict[str, Any]],
) -> int:
    best_steps = split_ints(args.reconstruction_steps)[0]
    best_quality = -1.0
    for steps in split_ints(args.reconstruction_steps):
        run_name = f"recon_sweep_{args.run_prefix}_{category}_steps{steps}"
        run_dir = REPO_ROOT / "runs" / run_name
        run_if_needed(
            infer_cmd(args, category=category, checkpoint=checkpoint, run_name=run_name, ablation="residual_only", reconstruction_steps=steps),
            run_dir / "predictions.npz",
            resume=args.resume,
            dry_run=args.dry_run,
        )
        evaluate_run(run_dir, args)
        metrics = {} if args.dry_run else _load_json(run_dir / "metrics.json")
        quality = _quality(metrics)
        row = {
            "category": category,
            "run": run_name,
            "reconstruction_steps": steps,
            "image_auroc": metrics.get("image_auroc", ""),
            "pixel_auroc": metrics.get("pixel_auroc", ""),
            "aupro": metrics.get("aupro", ""),
            "pixel_ap": metrics.get("pixel_ap", ""),
            "dice": metrics.get("dice", ""),
            "quality": quality if _finite(quality) else "",
        }
        sweep_rows.append(row)
        if _finite(quality) and quality > best_quality:
            best_quality = quality
            best_steps = steps
    return best_steps


def train_category_models(category: str, args: argparse.Namespace, reconstruction_steps: int) -> dict[str, Path]:
    base_run = f"{args.run_prefix}_{category}_full"
    base_dir = REPO_ROOT / "runs" / base_run
    checkpoint = base_dir / "diffusion.pt"
    hn_dir = base_dir / "hard_negatives"
    hn_manifest = hn_dir / "manifest.csv"
    sev_checkpoint = base_dir / "hn_sev.pt"
    scheduler_checkpoint = base_dir / "lc_rds.pt"
    synthetic_run = f"next_models_{args.run_prefix}_{category}_synthetic_only_sev"
    no_proto_run = f"next_models_{args.run_prefix}_{category}_no_prototype"
    synthetic_checkpoint = REPO_ROOT / "runs" / synthetic_run / "hn_sev.pt"
    no_proto_checkpoint = REPO_ROOT / "runs" / no_proto_run / "hn_sev.pt"
    common = common_data_args(args, category)

    run_if_needed(
        py("mine_hard_negatives.py", *common, "--checkpoint", checkpoint, "--reconstruction-steps", reconstruction_steps, "--run-name", base_run),
        hn_manifest,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    run_if_needed(
        py(
            "train_hn_sev.py",
            *common,
            "--checkpoint",
            checkpoint,
            "--hard-negative-dir",
            hn_dir,
            "--epochs",
            args.sev_epochs,
            "--batch-size",
            args.sev_batch_size,
            "--run-name",
            base_run,
        ),
        sev_checkpoint,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    run_if_needed(
        py(
            "train_hn_sev.py",
            *common,
            "--checkpoint",
            checkpoint,
            "--synthetic-only",
            "--epochs",
            args.sev_epochs,
            "--batch-size",
            args.sev_batch_size,
            "--run-name",
            synthetic_run,
        ),
        synthetic_checkpoint,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    run_if_needed(
        py(
            "train_hn_sev.py",
            *common,
            "--checkpoint",
            checkpoint,
            "--hard-negative-dir",
            hn_dir,
            "--disable-prototype",
            "--epochs",
            args.sev_epochs,
            "--batch-size",
            args.sev_batch_size,
            "--run-name",
            no_proto_run,
        ),
        no_proto_checkpoint,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    run_if_needed(
        py(
            "train_lc_rds.py",
            "--config",
            args.config,
            "--samples",
            args.scheduler_samples,
            "--epochs",
            args.scheduler_epochs,
            "--run-name",
            base_run,
            "--seed",
            args.seed,
            "--device",
            args.device,
        ),
        scheduler_checkpoint,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    return {
        "base_dir": base_dir,
        "checkpoint": checkpoint,
        "sev_checkpoint": sev_checkpoint,
        "synthetic_checkpoint": synthetic_checkpoint,
        "no_proto_checkpoint": no_proto_checkpoint,
        "scheduler_checkpoint": scheduler_checkpoint,
    }


def run_ablation(
    category: str,
    args: argparse.Namespace,
    paths: dict[str, Path],
    ablation: str,
    reconstruction_steps: int,
) -> Path:
    run_name = f"{args.run_prefix}_{category}_{ablation}"
    run_dir = REPO_ROOT / "runs" / run_name
    sev = paths["sev_checkpoint"]
    if ablation == "synthetic_only_sev":
        sev = paths["synthetic_checkpoint"]
    elif ablation == "no_prototype":
        sev = paths["no_proto_checkpoint"]
    run_if_needed(
        infer_cmd(
            args,
            category=category,
            checkpoint=paths["checkpoint"],
            sev_checkpoint=sev,
            scheduler_checkpoint=paths["scheduler_checkpoint"],
            run_name=run_name,
            ablation=ablation,
            reconstruction_steps=reconstruction_steps,
        ),
        run_dir / "predictions.npz",
        resume=args.resume,
        dry_run=args.dry_run,
    )
    evaluate_run(run_dir, args)
    return run_dir


def run_full_inference(category: str, args: argparse.Namespace, paths: dict[str, Path], reconstruction_steps: int) -> Path:
    run_name = f"{args.run_prefix}_{category}_full"
    run_dir = paths["base_dir"]
    run_if_needed(
        infer_cmd(
            args,
            category=category,
            checkpoint=paths["checkpoint"],
            sev_checkpoint=paths["sev_checkpoint"],
            scheduler_checkpoint=paths["scheduler_checkpoint"],
            run_name=run_name,
            ablation="full",
            reconstruction_steps=reconstruction_steps,
        ),
        run_dir / "predictions.npz",
        resume=args.resume,
        dry_run=args.dry_run,
    )
    evaluate_run(run_dir, args)
    export_run(run_dir, args)
    return run_dir


def run_crv_search(category: str, args: argparse.Namespace, full_dir: Path, crv_rows: list[dict[str, Any]]) -> None:
    run_command(py("tools/search_crv_weight.py", "--run-dir", full_dir, "--weights", args.crv_weights, "--out", full_dir), dry_run=args.dry_run)
    if args.dry_run:
        return
    for row in read_csv(full_dir / "crv_weight_search.csv"):
        row = dict(row)
        row["category"] = category
        crv_rows.append(row)


def crv_decision(crv_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in crv_rows:
        weight = str(row.get("crv_weight", ""))
        quality = _safe_float(row.get("quality"))
        if _finite(quality):
            grouped.setdefault(weight, []).append(quality)
    scored = [
        {"crv_weight": weight, "mean_quality": sum(vals) / len(vals), "categories": len(vals)}
        for weight, vals in grouped.items()
        if vals
    ]
    best = max(scored, key=lambda row: row["mean_quality"], default={})
    return {"recommended": best, "weights": scored}


def run_category(
    category: str,
    args: argparse.Namespace,
    ablations: list[str],
    sweep_rows: list[dict[str, Any]],
    crv_rows: list[dict[str, Any]],
    schema_report: list[dict[str, Any]],
) -> dict[str, Any]:
    run_command(py("tools/index_datasets.py", "--config", args.config, "--category", category, "--out", f"runs/index_mvtec_{category}.csv"), dry_run=args.dry_run)
    base_run = f"{args.run_prefix}_{category}_full"
    base_checkpoint = REPO_ROOT / "runs" / base_run / "diffusion.pt"
    run_if_needed(
        py("train_diffusion.py", *common_data_args(args, category), "--epochs", args.diffusion_epochs, "--batch-size", args.batch_size, "--run-name", base_run),
        base_checkpoint,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    chosen_steps = run_reconstruction_sweep(category, args, base_checkpoint, sweep_rows)
    paths = train_category_models(category, args, chosen_steps)
    full_dir = run_full_inference(category, args, paths, chosen_steps)
    run_crv_search(category, args, full_dir, crv_rows)
    if not args.dry_run:
        schema_report.append(check_schema(full_dir, args.qualitative_limit))
    for ablation in ablations:
        run_dir = run_ablation(category, args, paths, ablation, chosen_steps)
        export_run(run_dir, args)
        if not args.dry_run:
            schema_report.append(check_schema(run_dir, args.qualitative_limit))
    return {"category": category, "reconstruction_steps": chosen_steps}


def write_crv_tables(args: argparse.Namespace, crv_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = REPO_ROOT / args.tables_out
    if not crv_rows:
        return {}
    fields = ["category", "crv_weight", "image_auroc", "pixel_auroc", "pixel_ap", "aupro", "dice", "quality"]
    write_csv(out / "table_crv_weight_search.csv", crv_rows, fields)
    decision = crv_decision(crv_rows)
    (out / "crv_weight_decision.json").write_text(json.dumps(_json_safe(decision), indent=2, allow_nan=False), encoding="utf-8")
    return decision


def write_reconstruction_tables(args: argparse.Namespace, sweep_rows: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    out = REPO_ROOT / args.tables_out
    fields = ["category", "run", "reconstruction_steps", "image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice", "quality"]
    write_csv(out / "table_reconstruction_sweep.csv", sweep_rows, fields)
    (out / "reconstruction_decisions.json").write_text(json.dumps(_json_safe(decisions), indent=2, allow_nan=False), encoding="utf-8")


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.categories = "bottle"
    if not args.run_prefix.endswith("_smoke"):
        args.run_prefix = f"{args.run_prefix}_smoke"
    args.image_size = 64
    args.max_samples = "8"
    args.diffusion_epochs = 1
    args.sev_epochs = 1
    args.scheduler_epochs = min(args.scheduler_epochs, 2)
    args.scheduler_samples = min(args.scheduler_samples, 64)
    if args.reconstruction_steps == "1,5,10":
        args.reconstruction_steps = "1,5"
    if args.crv_weights == "0.25,0.35,0.5,1.0":
        args.crv_weights = "0.25,0.35"
    if args.tables_out == "tables/next_mvtec5":
        args.tables_out = "tables/next_mvtec5_smoke"


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    categories = split_csv(args.categories)
    ablations = SMOKE_ABLATIONS if args.smoke else FULL_ABLATIONS
    sweep_rows: list[dict[str, Any]] = []
    crv_rows: list[dict[str, Any]] = []
    schema_report: list[dict[str, Any]] = []
    reconstruction_decisions: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for category in categories:
        try:
            decision = run_category(category, args, ablations, sweep_rows, crv_rows, schema_report)
            reconstruction_decisions.append(decision)
        except subprocess.CalledProcessError as exc:
            failures.append({"category": category, "error": str(exc)})
            if args.fail_fast:
                raise

    report_path = REPO_ROOT / "runs" / f"{args.run_prefix}_report.json"
    if not args.dry_run:
        write_reconstruction_tables(args, sweep_rows, reconstruction_decisions)
        crv_choice = write_crv_tables(args, crv_rows)
    else:
        crv_choice = {}

    summary_cmd = py("tools/summarize_evidence.py", "--runs", "runs", "--prefix", args.run_prefix, "--out", args.tables_out)
    if not args.smoke:
        summary_cmd.extend(["--exclude-prefix", f"{args.run_prefix}_smoke"])
    run_command(summary_cmd, dry_run=args.dry_run)

    gate = {}
    if not args.dry_run:
        gate = _load_json(REPO_ROOT / args.tables_out / "gate_summary.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "orchestrator": "tools/run_next_phase_mvtec5.py",
            "args": vars(args),
            "commands": COMMAND_LOG,
            "reconstruction_decisions": reconstruction_decisions,
            "crv_weight_decision": crv_choice,
            "schema": schema_report,
            "gate": gate,
            "failures": failures,
        }
        report_path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")

    ready = bool(gate.get("ready_for_mvtec15", False))
    print(f"Finished next-phase MVTec5 loop. Report: {report_path}. ready_for_mvtec15={ready}. Failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
