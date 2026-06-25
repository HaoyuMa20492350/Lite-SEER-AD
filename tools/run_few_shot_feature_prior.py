from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run frozen-policy MVTec few-shot feature-prior experiments."
    )
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument(
        "--policy-table",
        default="tables/feature_first_paper_package/table_selection_stability.csv",
    )
    p.add_argument("--categories", default="all")
    p.add_argument("--shots", default="8,16,32")
    p.add_argument("--seeds", default="7,13,23")
    p.add_argument("--max-test-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--run-prefix", default="fewshot_mvtec")
    p.add_argument("--out", default="tables/fewshot_mvtec")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-random-feature-weights", action="store_true")
    p.add_argument("--log-file", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_categories(config: Path) -> list[str]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    root = REPO_ROOT / str((cfg.get("dataset", {}) or {}).get("root", ""))
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def frozen_policies(path: Path) -> dict[str, str]:
    return {
        row["category"]: row["dominant_candidate"]
        for row in read_csv(path)
        if row.get("dataset") == "mvtec15"
    }


def policy_spec(candidate: str) -> dict[str, Any]:
    if candidate == "padim128r18l123":
        return {
            "method": "padim",
            "backbone": "resnet18",
            "layers": "layer1,layer2,layer3",
            "image_size": 128,
            "image_score_mode": "max",
            "image_score_source": "feature_raw",
            "postprocess": None,
        }
    image_size = 128 if candidate == "pixelraw" else 256
    return {
        "method": "patchcore",
        "backbone": "wide_resnet50_2",
        "layers": "layer2,layer3",
        "image_size": image_size,
        "image_score_mode": "top5",
        "image_score_source": "feature_raw_cosine",
        "postprocess": "gaussian:3" if candidate == "post_highres_gaussian3" else None,
    }


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


def final_run_dir(base_run: str, postprocess: str | None) -> Path:
    base = REPO_ROOT / "runs" / base_run
    if postprocess:
        return base / f"feature_pixelpost_{postprocess.replace(':', '')}"
    return base


def collect_result(
    dataset: str,
    category: str,
    candidate: str,
    shot: int,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    if metrics.get("latency_ms") is None and run_dir.parent != run_dir:
        source_metrics = read_json(run_dir.parent / "metrics.json")
        metrics["latency_ms"] = source_metrics.get("latency_ms")
    predictions = run_dir / "predictions.npz"
    test_images = ""
    if predictions.exists():
        with np.load(predictions) as pred:
            test_images = len(pred["labels"])
    return {
        "dataset": dataset,
        "category": category,
        "candidate": candidate,
        "shot": shot,
        "seed": seed,
        "run": str(run_dir.relative_to(REPO_ROOT)),
        "train_images": shot,
        "test_images": test_images,
        **{
            metric: metrics.get(metric)
            for metric in [
                "image_auroc",
                "pixel_auroc",
                "aupro",
                "pixel_ap",
                "dice",
                "latency_ms",
            ]
        },
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    policies = frozen_policies(Path(args.policy_table))
    categories = (
        discover_categories(config_path)
        if args.categories in {"all", "auto"}
        else split_csv(args.categories)
    )
    missing = sorted(set(categories) - set(policies))
    if missing:
        raise ValueError(f"Frozen policy table is missing categories: {', '.join(missing)}")
    shots = [int(value) for value in split_csv(args.shots)]
    seeds = [int(value) for value in split_csv(args.seeds)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle: TextIO | None = None
    rows: list[dict[str, Any]] = []
    try:
        if args.log_file:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
        for category in categories:
            candidate = policies[category]
            spec = policy_spec(candidate)
            for shot in shots:
                for seed in seeds:
                    stem = (
                        f"{args.run_prefix}_{category}_{candidate}_shot{shot}_seed{seed}"
                    )
                    model_run = f"{stem}_models"
                    checkpoint = REPO_ROOT / "runs" / model_run / "feature_prior.pt"
                    base_run = f"{stem}_candidate"
                    base_dir = REPO_ROOT / "runs" / base_run
                    result_dir = final_run_dir(base_run, spec["postprocess"])

                    if not (args.resume and checkpoint.exists()):
                        command = [
                            sys.executable,
                            "train_feature_prior.py",
                            "--config",
                            args.config,
                            "--category",
                            category,
                            "--method",
                            spec["method"],
                            "--backbone",
                            spec["backbone"],
                            "--layers",
                            spec["layers"],
                            "--image-size",
                            str(spec["image_size"]),
                            "--train-samples",
                            str(shot),
                            "--batch-size",
                            str(args.batch_size),
                            "--retrieval-max-references",
                            "0",
                            "--run-name",
                            model_run,
                            "--seed",
                            str(seed),
                            "--device",
                            args.device,
                        ]
                        if args.allow_random_feature_weights:
                            command.append("--allow-random-weights")
                        run(command, log_handle)

                    if not (args.resume and (base_dir / "predictions.npz").exists()):
                        command = [
                            sys.executable,
                            "tools/materialize_feature_prior_candidate.py",
                            "--config",
                            args.config,
                            "--category",
                            category,
                            "--feature-prior-checkpoint",
                            str(checkpoint.relative_to(REPO_ROOT)),
                            "--image-size",
                            str(spec["image_size"]),
                            "--max-samples",
                            str(args.max_test_samples),
                            "--batch-size",
                            str(args.batch_size),
                            "--run-name",
                            base_run,
                            "--image-score-mode",
                            spec["image_score_mode"],
                            "--image-score-source",
                            spec["image_score_source"],
                            "--pixel-heatmap-source",
                            "feature_raw",
                            "--seed",
                            str(seed),
                            "--device",
                            args.device,
                        ]
                        if args.allow_random_feature_weights:
                            command.append("--allow-random-feature-weights")
                        run(command, log_handle)

                    if spec["postprocess"] and not (
                        args.resume and (result_dir / "predictions.npz").exists()
                    ):
                        run(
                            [
                                sys.executable,
                                "tools/search_pixel_postprocess.py",
                                "--run-dir",
                                str(base_dir.relative_to(REPO_ROOT)),
                                "--modes",
                                spec["postprocess"],
                                "--out",
                                str(base_dir.relative_to(REPO_ROOT)),
                                "--materialize",
                            ],
                            log_handle,
                        )
                    rows.append(
                        collect_result(
                            "mvtec15",
                            category,
                            candidate,
                            shot,
                            seed,
                            result_dir,
                        )
                    )
                    write_csv(
                        out_dir / "table_few_shot_runs.csv",
                        rows,
                        [
                            "dataset",
                            "category",
                            "candidate",
                            "shot",
                            "seed",
                            "run",
                            "train_images",
                            "test_images",
                            "image_auroc",
                            "pixel_auroc",
                            "aupro",
                            "pixel_ap",
                            "dice",
                            "latency_ms",
                        ],
                    )
    finally:
        if log_handle is not None:
            log_handle.close()

    summary_rows = []
    for shot in shots:
        shot_rows = [row for row in rows if row["shot"] == shot]
        for metric in ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]:
            seed_means = []
            for seed in seeds:
                values = [
                    float(row[metric])
                    for row in shot_rows
                    if row["seed"] == seed and row.get(metric) not in {None, ""}
                ]
                if values:
                    seed_means.append(sum(values) / len(values))
            summary_rows.append(
                {
                    "shot": shot,
                    "metric": metric,
                    "seeds": len(seed_means),
                    "mean": sum(seed_means) / len(seed_means) if seed_means else None,
                    "std": (
                        float(np.std(seed_means, ddof=0)) if seed_means else None
                    ),
                }
            )
    write_csv(
        out_dir / "table_few_shot_summary.csv",
        summary_rows,
        ["shot", "metric", "seeds", "mean", "std"],
    )
    protocol = {
        "dataset": "mvtec15",
        "categories": categories,
        "shots": shots,
        "seeds": seeds,
        "test_samples_per_category": args.max_test_samples,
        "policy_source": args.policy_table,
        "policy_frozen_before_few_shot_evaluation": True,
        "uses_real_anomaly_labels_for_policy_selection": False,
        "runs_completed": len(rows),
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    if args.quiet:
        print(json.dumps(protocol, indent=2))
    else:
        print(json.dumps({"protocol": protocol, "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
