from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, TextIO

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the frozen feature-first detector on MVTec AD 2 public test data.")
    p.add_argument("--config", default="configs/mvtec_ad2.yaml")
    p.add_argument("--categories", default="all")
    p.add_argument("--seeds", default="7,13,23")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--run-prefix", default="mvtec_ad2_feature_first")
    p.add_argument("--out", default="tables/mvtec_ad2_feature_first")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-file", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dataset_root(config_path: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return REPO_ROOT / str((cfg.get("dataset", {}) or {}).get("root", ""))


def readiness(root: Path, categories: list[str]) -> dict[str, Any]:
    required_splits = [
        "train/good",
        "validation/good",
        "test_public/good",
        "test_public/bad",
        "test_public/ground_truth/bad",
    ]
    rows = []
    for category in categories:
        missing = [
            split
            for split in required_splits
            if not (root / category / Path(split)).exists()
        ]
        rows.append(
            {
                "category": category,
                "ready": not missing,
                "missing": missing,
            }
        )
    return {
        "dataset_root": str(root),
        "root_exists": root.exists(),
        "categories": rows,
        "ready": root.exists() and all(row["ready"] for row in rows),
        "official_public_split": "test_public",
        "official_private_splits": ["test_private", "test_private_mixed"],
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


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    categories = OFFICIAL_CATEGORIES if args.categories == "all" else split_csv(args.categories)
    seeds = [int(value) for value in split_csv(args.seeds)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = readiness(dataset_root(config_path), categories)
    (out_dir / "readiness.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if not status["ready"]:
        print(json.dumps(status, indent=2))
        raise SystemExit(2)

    rows: list[dict[str, Any]] = []
    log_handle: TextIO | None = None
    try:
        if args.log_file:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8")
        for category in categories:
            for seed in seeds:
                model_run = f"{args.run_prefix}_{category}_seed{seed}_models"
                checkpoint = REPO_ROOT / "runs" / model_run / "feature_prior.pt"
                public_run = f"{args.run_prefix}_{category}_seed{seed}_public"
                public_dir = REPO_ROOT / "runs" / public_run
                if not (args.resume and checkpoint.exists()):
                    run(
                        [
                            sys.executable,
                            "train_feature_prior.py",
                            "--config",
                            args.config,
                            "--category",
                            category,
                            "--method",
                            "patchcore",
                            "--backbone",
                            "wide_resnet50_2",
                            "--layers",
                            "layer2,layer3",
                            "--image-size",
                            str(args.image_size),
                            "--train-splits",
                            "train,validation",
                            "--retrieval-max-references",
                            "0",
                            "--batch-size",
                            str(args.batch_size),
                            "--run-name",
                            model_run,
                            "--seed",
                            str(seed),
                            "--device",
                            args.device,
                        ],
                        log_handle,
                    )
                if not (args.resume and (public_dir / "predictions.npz").exists()):
                    run(
                        [
                            sys.executable,
                            "tools/materialize_feature_prior_candidate.py",
                            "--config",
                            args.config,
                            "--category",
                            category,
                            "--feature-prior-checkpoint",
                            str(checkpoint.relative_to(REPO_ROOT)),
                            "--split",
                            "test_public",
                            "--image-size",
                            str(args.image_size),
                            "--batch-size",
                            str(args.batch_size),
                            "--run-name",
                            public_run,
                            "--image-score-mode",
                            "top5",
                            "--image-score-source",
                            "feature_raw_cosine",
                            "--pixel-heatmap-source",
                            "feature_raw",
                            "--seed",
                            str(seed),
                            "--device",
                            args.device,
                        ],
                        log_handle,
                    )
                metrics = read_json(public_dir / "metrics.json")
                rows.append(
                    {
                        "dataset": "mvtec_ad2",
                        "category": category,
                        "seed": seed,
                        "run": str(public_dir.relative_to(REPO_ROOT)),
                        **{
                            metric: metrics.get(metric)
                            for metric in [
                                "image_auroc",
                                "pixel_auroc",
                                "aupro",
                                "pixel_ap",
                                "dice",
                                "latency_ms",
                                "test_images",
                            ]
                        },
                    }
                )
                write_csv(
                    out_dir / "table_public_runs.csv",
                    rows,
                    [
                        "dataset",
                        "category",
                        "seed",
                        "run",
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
    protocol = {
        "dataset": "mvtec_ad2",
        "categories": categories,
        "seeds": seeds,
        "train_splits": ["train", "validation"],
        "evaluation_split": "test_public",
        "private_submission_required_for_official_leaderboard": True,
        "uses_public_test_labels_for_configuration": False,
        "runs_completed": len(rows),
    }
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol if args.quiet else {"protocol": protocol, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
