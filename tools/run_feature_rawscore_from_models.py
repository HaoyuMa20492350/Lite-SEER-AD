from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MVTEC15 = [
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
METRIC_KEYS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice", "fprr"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run raw-feature image scoring from existing feature-first model checkpoints.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--categories", default=",".join(DEFAULT_MVTEC15))
    p.add_argument("--source-prefix", default="feature_mvtec15", help="Prefix containing <prefix>_<category>_models checkpoints.")
    p.add_argument("--run-prefix", default="feature_rawscore_mvtec15")
    p.add_argument("--tables-out", default="tables/feature_rawscore_mvtec15")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-samples", default="64")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--ablation", default="feature_hn_sev_crv")
    p.add_argument("--crv-weight", type=float, default=0.35)
    p.add_argument("--crv-weights", default="0,0.05,0.1,0.2,0.35,0.5")
    p.add_argument("--reconstruction-steps", type=int, default=5)
    p.add_argument("--image-score-source", default="feature_raw_cosine")
    p.add_argument("--image-score-mode", default="top5")
    p.add_argument("--pixel-heatmap-source", default="feature_raw")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(args: argparse.Namespace, categories: list[str]) -> None:
    out = REPO_ROOT / args.tables_out
    rows: list[dict[str, Any]] = []
    crv_rows: list[dict[str, Any]] = []
    image_score_rows: list[dict[str, Any]] = []
    for category in categories:
        tuned_run = f"{args.run_prefix}_{category}_feature_tuned_crv"
        metrics = _load_json(REPO_ROOT / "runs" / tuned_run / "metrics.json")
        if metrics:
            row = {"category": category, "run": tuned_run, "ablation": "feature_tuned_crv"}
            row.update(metrics)
            rows.append(row)
        crv = _load_json(out / f"crv_weight_{category}" / "recommended_crv_weight.json")
        if crv:
            crv_rows.append({"category": category, **crv})
        score = _load_json(out / f"image_score_{category}" / "recommended_image_score_mode.json")
        if score:
            image_score_rows.append({"category": category, **score})

    main_fields = [
        "category",
        "run",
        "ablation",
        "image_auroc",
        "pixel_auroc",
        "aupro",
        "pixel_ap",
        "dice",
        "fprr",
        "crv_weight",
        "image_score_mode",
        "image_score_source",
        "pixel_heatmap_source",
    ]
    _write_rows(out / "table_rawscore_mvtec15.csv", rows, main_fields)
    _write_rows(
        out / "table_crv_weight_search_feature_raw.csv",
        crv_rows,
        [
            "category",
            "crv_weight",
            "image_auroc",
            "pixel_auroc",
            "pixel_ap",
            "aupro",
            "dice",
            "quality",
            "base_key",
            "image_score_mode",
            "image_score_source",
        ],
    )
    _write_rows(
        out / "table_image_score_mode_search_raw.csv",
        image_score_rows,
        ["category", "image_score_mode", "heatmap_key", "image_auroc", "pixel_auroc", "pixel_ap", "aupro", "dice"],
    )
    mean = {
        key: sum(float(row[key]) for row in rows if row.get(key) not in {"", None}) / max(1, sum(1 for row in rows if row.get(key) not in {"", None}))
        for key in METRIC_KEYS
    }
    (out / "rawscore_mvtec15_summary.json").write_text(
        json.dumps({"categories": categories, "completed": len(rows), "mean": mean}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"table": str(out / "table_rawscore_mvtec15.csv"), "completed": len(rows), "mean": mean}, indent=2))


def run_category(args: argparse.Namespace, category: str) -> None:
    model_dir = REPO_ROOT / "runs" / f"{args.source_prefix}_{category}_models"
    diffusion = model_dir / "diffusion.pt"
    feature = model_dir / "feature_prior.pt"
    sev = model_dir / "hn_sev.pt"
    missing = [str(path) for path in [diffusion, feature, sev] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints for {category}: {missing}")

    run_name = f"{args.run_prefix}_{category}_{args.ablation}"
    run_dir = REPO_ROOT / "runs" / run_name
    common = [
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
    run_if_needed(
        py(
            "infer.py",
            *common,
            "--ckpt",
            diffusion,
            "--sev-checkpoint",
            sev,
            "--feature-prior-checkpoint",
            feature,
            "--run-name",
            run_name,
            "--ablation",
            args.ablation,
            "--crv-weight",
            args.crv_weight,
            "--reconstruction-steps",
            args.reconstruction_steps,
            "--image-score-source",
            args.image_score_source,
            "--image-score-mode",
            args.image_score_mode,
            "--pixel-heatmap-source",
            args.pixel_heatmap_source,
        ),
        run_dir / "predictions.npz",
        args.resume,
        args.dry_run,
    )
    run_command(py("evaluate.py", "--pred_dir", run_dir, "--out", run_dir / "eval_metrics.json"), args.dry_run)

    crv_out = REPO_ROOT / args.tables_out / f"crv_weight_{category}"
    run_if_needed(
        py(
            "tools/search_crv_weight.py",
            "--run-dir",
            run_dir,
            "--weights",
            args.crv_weights,
            "--base-key",
            "auto",
            "--image-score-source",
            args.image_score_source,
            "--image-score-mode",
            args.image_score_mode,
            "--out",
            crv_out,
        ),
        crv_out / "recommended_crv_weight.json",
        args.resume,
        args.dry_run,
    )

    tuned_run_dir = REPO_ROOT / "runs" / f"{args.run_prefix}_{category}_feature_tuned_crv"
    run_if_needed(
        py(
            "tools/materialize_tuned_crv.py",
            "--source-run-dir",
            run_dir,
            "--recommendation",
            crv_out / "recommended_crv_weight.json",
            "--out-run-dir",
            tuned_run_dir,
            "--base-key",
            "auto",
            "--image-score-source",
            args.image_score_source,
            "--image-score-mode",
            args.image_score_mode,
            "--pixel-heatmap-source",
            args.pixel_heatmap_source,
        ),
        tuned_run_dir / "predictions.npz",
        args.resume,
        args.dry_run,
    )
    run_command(py("evaluate.py", "--pred_dir", tuned_run_dir, "--out", tuned_run_dir / "eval_metrics.json"), args.dry_run)

    score_out = REPO_ROOT / args.tables_out / f"image_score_{category}"
    run_if_needed(
        py("tools/search_image_score_mode.py", "--run-dir", tuned_run_dir, "--heatmap-key", "score_heatmaps", "--out", score_out),
        score_out / "recommended_image_score_mode.json",
        args.resume,
        args.dry_run,
    )


def main() -> None:
    args = parse_args()
    categories = split_csv(args.categories)
    for category in categories:
        run_category(args, category)
    if not args.dry_run:
        aggregate(args, categories)


if __name__ == "__main__":
    main()
