from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ["bottle", "cable", "capsule", "metal_nut", "zipper"]
FEATURE_ABLATIONS = [
    "residual_only",
    "feature_only",
    "feature_hn_sev",
    "feature_hn_sev_crv",
    "feature_fixed10",
    "feature_fixed25",
    "feature_rule_brds",
    "utility_lc_rds",
]
MATERIALIZED_ABLATIONS = ["feature_tuned_crv"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the feature-first Lite-SEER-AD architecture-flip experiment.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES), help="Comma-separated categories, or 'auto' to discover from the dataset config.")
    p.add_argument("--run-prefix", default="feature_mvtec5")
    p.add_argument(
        "--model-prefix",
        default=None,
        help="Prefix containing <prefix>_<category>_models checkpoints. Defaults to --run-prefix.",
    )
    p.add_argument(
        "--inference-only",
        action="store_true",
        help="Reuse existing model checkpoints and skip training and hard-negative mining.",
    )
    p.add_argument(
        "--reuse-base-checkpoints",
        action="store_true",
        help="Reuse diffusion/feature checkpoints, then remine hard negatives and retrain HN-SEV.",
    )
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-samples", default="64")
    p.add_argument("--diffusion-epochs", type=int, default=3)
    p.add_argument("--sev-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sev-batch-size", type=int, default=8)
    p.add_argument("--feature-method", choices=["patchcore", "padim"], default="patchcore")
    p.add_argument("--feature-backbone", choices=["wide_resnet50_2", "resnet18"], default="wide_resnet50_2")
    p.add_argument("--coreset-ratio", type=float, default=0.1)
    p.add_argument("--max-bank-patches", type=int, default=20000)
    p.add_argument("--crv-weight", type=float, default=0.35)
    p.add_argument(
        "--freeze-crv-weight",
        action="store_true",
        help="Materialize feature_tuned_crv with --crv-weight instead of selecting a weight on test labels.",
    )
    p.add_argument("--crv-weights", default="0,0.05,0.1,0.2,0.35,0.5")
    p.add_argument("--image-score-mode", default="top5")
    p.add_argument("--image-score-source", default="feature_raw_cosine")
    p.add_argument("--pixel-heatmap-source", default="feature_raw")
    p.add_argument("--reconstruction-steps", type=int, default=5)
    p.add_argument("--tables-out", default="tables/feature_mvtec5")
    p.add_argument("--qualitative-limit", type=int, default=5)
    p.add_argument("--selector-only", action="store_true", help="Only materialize the feature_hn_sev_crv and feature_tuned_crv runs needed by held-out policy selection.")
    p.add_argument("--skip-figures", action="store_true", help="Skip qualitative figure export for faster candidate generation.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-random-feature-weights", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8")) or {}


def dataset_name(cfg: dict[str, Any]) -> str:
    return str((cfg.get("dataset", {}) or {}).get("name", "dataset"))


def discover_categories(cfg: dict[str, Any]) -> list[str]:
    dataset = cfg.get("dataset", {}) or {}
    categories = dataset.get("categories")
    if isinstance(categories, list) and categories:
        return [str(item) for item in categories]
    if isinstance(categories, str) and categories not in {"", "all", "auto"}:
        return split_csv(categories)

    root = REPO_ROOT / str(dataset.get("root", ""))
    if root.exists():
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        return [name for name in names if name not in {"split_csv", "archives"}]
    category = dataset.get("category")
    if isinstance(category, str) and category:
        return [category]
    raise FileNotFoundError(f"Cannot discover categories because dataset root does not exist: {root}")


def write_dataset_table_aliases(tables_dir: Path, name: str) -> None:
    aliases = {
        "table_main_mvtec5.csv": f"table_main_{name}.csv",
        "table_efficiency_mvtec5.csv": f"table_efficiency_{name}.csv",
    }
    for src_name, dst_name in aliases.items():
        src = tables_dir / src_name
        if src.exists():
            shutil.copyfile(src, tables_dir / dst_name)


def max_sample_args(value: str | int | None) -> list[str]:
    if value is None:
        return []
    text = str(value).strip().lower()
    if text in {"", "all", "none", "null"}:
        return []
    return ["--max-samples", str(int(float(text)))]


def py(script: str, *args: str | int | float | Path) -> list[str]:
    return [sys.executable, script, *[str(arg) for arg in args]]


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(str(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_if_needed(cmd: list[str], output: Path, resume: bool, dry_run: bool) -> None:
    if resume and output.exists():
        print(f"SKIP existing {output}", flush=True)
        return
    run_command(cmd, dry_run)


def common_args(args: argparse.Namespace, category: str) -> list[str]:
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


def run_category(args: argparse.Namespace, category: str) -> None:
    model_prefix = args.model_prefix or args.run_prefix
    base_run = f"{model_prefix}_{category}_models"
    base_dir = REPO_ROOT / "runs" / base_run
    diffusion_ckpt = base_dir / "diffusion.pt"
    feature_ckpt = base_dir / "feature_prior.pt"
    hn_dir = base_dir / "hard_negatives"
    sev_ckpt = base_dir / "hn_sev.pt"

    common = common_args(args, category)
    if args.inference_only:
        missing = [
            str(path)
            for path in (diffusion_ckpt, feature_ckpt, sev_ckpt)
            if not path.exists()
        ]
        if missing and not args.dry_run:
            raise FileNotFoundError(
                f"Missing checkpoints for inference-only category {category}: {missing}"
            )
    else:
        if args.reuse_base_checkpoints:
            missing = [
                str(path)
                for path in (diffusion_ckpt, feature_ckpt)
                if not path.exists()
            ]
            if missing and not args.dry_run:
                raise FileNotFoundError(
                    f"Missing reusable base checkpoints for {category}: {missing}"
                )
        else:
            run_if_needed(
                py("train_diffusion.py", *common, "--epochs", args.diffusion_epochs, "--batch-size", args.batch_size, "--run-name", base_run),
                diffusion_ckpt,
                args.resume,
                args.dry_run,
            )
            feature_cmd = py(
                "train_feature_prior.py",
                *common,
                "--method",
                args.feature_method,
                "--backbone",
                args.feature_backbone,
                "--coreset-ratio",
                args.coreset_ratio,
                "--max-bank-patches",
                args.max_bank_patches,
                "--batch-size",
                args.batch_size,
                "--run-name",
                base_run,
            )
            if args.allow_random_feature_weights:
                feature_cmd.append("--allow-random-weights")
            run_if_needed(feature_cmd, feature_ckpt, args.resume, args.dry_run)
        run_if_needed(
            py("mine_hard_negatives.py", *common, "--checkpoint", diffusion_ckpt, "--reconstruction-steps", args.reconstruction_steps, "--run-name", base_run),
            hn_dir / "manifest.csv",
            args.resume,
            args.dry_run,
        )
        hn_cmd = py(
            "train_hn_sev.py",
            *common,
            "--checkpoint",
            diffusion_ckpt,
            "--hard-negative-dir",
            hn_dir,
            "--feature-prior-checkpoint",
            feature_ckpt,
            "--epochs",
            args.sev_epochs,
            "--batch-size",
            args.sev_batch_size,
            "--run-name",
            base_run,
        )
        if args.allow_random_feature_weights:
            hn_cmd.append("--allow-random-feature-weights")
        run_if_needed(hn_cmd, sev_ckpt, args.resume, args.dry_run)

    ablations = ["feature_hn_sev_crv"] if args.selector_only else FEATURE_ABLATIONS
    for ablation in ablations:
        run_name = f"{args.run_prefix}_{category}_{ablation}"
        run_dir = REPO_ROOT / "runs" / run_name
        infer_cmd = py(
            "infer.py",
            *common,
            "--ckpt",
            diffusion_ckpt,
            "--sev-checkpoint",
            sev_ckpt,
            "--feature-prior-checkpoint",
            feature_ckpt,
            "--run-name",
            run_name,
            "--ablation",
            ablation,
            "--crv-weight",
            args.crv_weight,
            "--image-score-mode",
            args.image_score_mode,
            "--reconstruction-steps",
            args.reconstruction_steps,
        )
        if args.image_score_source:
            infer_cmd.extend(["--image-score-source", args.image_score_source])
        if args.pixel_heatmap_source:
            infer_cmd.extend(["--pixel-heatmap-source", args.pixel_heatmap_source])
        if args.allow_random_feature_weights:
            infer_cmd.append("--allow-random-feature-weights")
        run_if_needed(infer_cmd, run_dir / "predictions.npz", args.resume, args.dry_run)
        run_command(py("evaluate.py", "--pred_dir", run_dir, "--out", run_dir / "eval_metrics.json"), args.dry_run)
        if not args.skip_figures:
            run_command(py("tools/export_figures.py", "--run-dir", run_dir, "--out", run_dir / "qualitative_cases", "--limit", args.qualitative_limit), args.dry_run)
    recommendation = (
        REPO_ROOT
        / args.tables_out
        / f"crv_weight_{category}"
        / "recommended_crv_weight.json"
    )
    if not args.freeze_crv_weight:
        search_cmd = py(
            "tools/search_crv_weight.py",
            "--run-dir",
            REPO_ROOT / "runs" / f"{args.run_prefix}_{category}_feature_hn_sev_crv",
            "--weights",
            args.crv_weights,
            "--base-key",
            "auto",
            "--image-score-mode",
            args.image_score_mode,
            "--out",
            recommendation.parent,
        )
        if args.image_score_source:
            search_cmd.extend(["--image-score-source", args.image_score_source])
        run_if_needed(
            search_cmd,
            recommendation,
            args.resume,
            args.dry_run,
        )
    tuned_run_dir = REPO_ROOT / "runs" / f"{args.run_prefix}_{category}_feature_tuned_crv"
    tuned_cmd = py(
        "tools/materialize_tuned_crv.py",
        "--source-run-dir",
        REPO_ROOT / "runs" / f"{args.run_prefix}_{category}_feature_hn_sev_crv",
        "--out-run-dir",
        tuned_run_dir,
        "--base-key",
        "auto",
        "--image-score-mode",
        args.image_score_mode,
    )
    if args.freeze_crv_weight:
        tuned_cmd.extend(["--crv-weight", str(args.crv_weight)])
    else:
        tuned_cmd.extend(["--recommendation", recommendation])
    if args.image_score_source:
        tuned_cmd.extend(["--image-score-source", args.image_score_source])
    if args.pixel_heatmap_source:
        tuned_cmd.extend(["--pixel-heatmap-source", args.pixel_heatmap_source])
    run_if_needed(tuned_cmd, tuned_run_dir / "predictions.npz", args.resume, args.dry_run)
    run_command(py("evaluate.py", "--pred_dir", tuned_run_dir, "--out", tuned_run_dir / "eval_metrics.json"), args.dry_run)


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.categories = split_csv(args.categories)[0]
    if not args.run_prefix.endswith("_smoke"):
        args.run_prefix = f"{args.run_prefix}_smoke"
    if not str(args.tables_out).replace("\\", "/").rstrip("/").endswith("_smoke"):
        args.tables_out = f"{args.tables_out}_smoke"
    args.image_size = 64
    args.max_samples = "8"
    args.diffusion_epochs = 1
    args.sev_epochs = 1
    args.batch_size = min(args.batch_size, 2)
    args.sev_batch_size = min(args.sev_batch_size, 2)
    args.max_bank_patches = min(args.max_bank_patches, 512)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.categories.strip().lower() in {"auto", "all"}:
        args.categories = ",".join(discover_categories(cfg))
    apply_smoke_defaults(args)
    failures: list[dict[str, Any]] = []
    for category in split_csv(args.categories):
        try:
            run_category(args, category)
        except subprocess.CalledProcessError as exc:
            failures.append({"category": category, "error": str(exc)})
            raise
    summary_cmd = py(
        "tools/summarize_evidence.py",
        "--runs",
        "runs",
        "--prefix",
        args.run_prefix,
        "--out",
        args.tables_out,
        "--include-ablations",
        ",".join((["feature_hn_sev_crv"] if args.selector_only else FEATURE_ABLATIONS) + MATERIALIZED_ABLATIONS),
    )
    if not args.smoke:
        summary_cmd.extend(["--exclude-prefix", f"{args.run_prefix}_smoke"])
    run_command(summary_cmd, args.dry_run)
    if not args.dry_run:
        write_dataset_table_aliases(REPO_ROOT / args.tables_out, dataset_name(cfg))
    report_path = REPO_ROOT / "runs" / f"{args.run_prefix}_report.json"
    if not args.dry_run:
        report_path.write_text(json.dumps({"args": vars(args), "failures": failures}, indent=2), encoding="utf-8")
    print(f"Finished feature-first MVTec5 loop. Report: {report_path}. Failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
