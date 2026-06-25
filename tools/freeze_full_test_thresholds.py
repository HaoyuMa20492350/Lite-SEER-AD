from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.pixel_threshold_policy import (
    POLICY_PROTOCOL,
    save_pixel_threshold_policy,
)
from tools.audit_full_test_coverage import ABLATIONS, DATASETS
from tools.audit_full_test_coverage import discover_categories, load_config
from tools.freeze_pixel_threshold_policy import freeze_run_policy


SOURCE_ABLATION = "feature_tuned_crv"
SELECTION_DATA = "official_train_normal_images_plus_synthetic_masks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze one label-free pixel threshold per full-test category, "
            "apply it to all identical-heatmap ablations, and re-evaluate."
        )
    )
    parser.add_argument("--datasets", default="mvtec,visa,mpdd")
    parser.add_argument("--seeds", default="7,13,23")
    parser.add_argument("--max-normal-images", type=int, default=16)
    parser.add_argument("--synthetic-variants", type=int, default=2)
    parser.add_argument("--canonical-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--out",
        default="tables/full_test_threshold_freeze_report.json",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_dir(spec: dict[str, Any], category: str, ablation: str) -> Path:
    return (
        REPO_ROOT
        / "runs"
        / f"{spec['run_prefix']}_{category}_{ablation}"
    )


def verify_shared_heatmaps(
    spec: dict[str, Any],
    category: str,
) -> dict[str, Any]:
    source = run_dir(spec, category, SOURCE_ABLATION)
    with np.load(source / "predictions.npz") as payload:
        source_paths = np.asarray(payload["paths"]).astype(str)
        source_heatmaps = np.asarray(payload["heatmaps"])
    checked = []
    for ablation in ABLATIONS:
        candidate = run_dir(spec, category, ablation)
        with np.load(candidate / "predictions.npz") as payload:
            paths_equal = np.array_equal(
                source_paths,
                np.asarray(payload["paths"]).astype(str),
            )
            heatmaps_equal = np.array_equal(
                source_heatmaps,
                np.asarray(payload["heatmaps"]),
            )
        if not paths_equal or not heatmaps_equal:
            raise ValueError(
                f"Cannot share a pixel threshold because heatmaps differ: {candidate}"
            )
        checked.append(candidate.name)
    return {
        "source_run": source.name,
        "checked_runs": checked,
        "paths_identical": True,
        "heatmaps_identical": True,
    }


def valid_artifact(
    run: Path,
    seed: int,
    synthetic_variants: int,
) -> bool:
    artifact = run / f"synthetic_validation_seed{seed}.npz"
    metrics = read_json(
        run / f"synthetic_validation_seed{seed}_metrics.json"
    )
    return (
        artifact.exists()
        and metrics.get("selection_data") == SELECTION_DATA
        and metrics.get("normal_source_split") == "train"
        and metrics.get("synthetic_variants") == synthetic_variants
        and metrics.get("synthetic_strength_profile") == "standard"
        and set(metrics.get("synthetic_mask_modes", []))
        == {
            ("blob", "scratch", "spot", "patch")[
                index % 4
            ]
            for index in range(synthetic_variants)
        }
        and metrics.get("synthetic_texture_source")
        == "deterministic_random"
        and metrics.get("synthetic_texture_images") == 0
        and metrics.get("uses_real_anomaly_labels_for_selection") is False
        and metrics.get("uses_real_anomaly_masks_for_selection") is False
    )


def materialize_seed(
    source: Path,
    seed: int,
    args: argparse.Namespace,
) -> None:
    if args.resume and valid_artifact(
        source,
        seed,
        int(args.synthetic_variants),
    ):
        print(f"SKIP valid threshold artifact {source.name} seed={seed}")
        return
    command = [
        sys.executable,
        "tools/materialize_synthetic_normal_validation.py",
        "--candidate-run-dir",
        str(source),
        "--out",
        str(source / f"synthetic_validation_seed{seed}.npz"),
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
    run(command)


def evaluate_with_policy(candidate: Path) -> None:
    run(
        [
            sys.executable,
            "evaluate.py",
            "--pred_dir",
            str(candidate),
            "--pixel-threshold-policy",
            str(candidate / "pixel_threshold_policy.json"),
            "--require-fixed-threshold",
        ]
    )


def summarize_dataset(spec: dict[str, Any]) -> None:
    run(
        [
            sys.executable,
            "tools/summarize_evidence.py",
            "--runs",
            "runs",
            "--prefix",
            str(spec["run_prefix"]),
            "--out",
            str(spec["tables"]),
            "--include-ablations",
            ",".join(ABLATIONS),
            "--exclude-prefix",
            f"{spec['run_prefix']}_smoke",
        ]
    )
    table_root = REPO_ROOT / str(spec["tables"])
    cfg = load_config(REPO_ROOT / spec["config"])
    dataset_name = str((cfg.get("dataset", {}) or {}).get("name", "dataset"))
    aliases = {
        "table_main_mvtec5.csv": f"table_main_{dataset_name}.csv",
        "table_efficiency_mvtec5.csv": f"table_efficiency_{dataset_name}.csv",
    }
    for source_name, target_name in aliases.items():
        source = table_root / source_name
        if source.exists():
            shutil.copyfile(source, table_root / target_name)


def main() -> None:
    args = parse_args()
    selected = split_csv(args.datasets)
    seeds = [int(value) for value in split_csv(args.seeds)]
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")

    report: dict[str, Any] = {
        "protocol": POLICY_PROTOCOL,
        "selection_data": SELECTION_DATA,
        "synthetic_seeds": seeds,
        "uses_real_anomaly_labels": False,
        "uses_real_anomaly_masks": False,
        "categories": [],
    }
    for dataset_name in selected:
        spec = DATASETS[dataset_name]
        cfg = load_config(REPO_ROOT / spec["config"])
        for category in discover_categories(cfg):
            shared = verify_shared_heatmaps(spec, category)
            source = run_dir(spec, category, SOURCE_ABLATION)
            for seed in seeds:
                materialize_seed(source, seed, args)
            policy = freeze_run_policy(source, seeds)
            evidence_metrics = [
                read_json(
                    source
                    / f"synthetic_validation_seed{seed}_metrics.json"
                )
                for seed in seeds
            ]
            evidence_fields = {
                "synthetic_variants",
                "synthetic_strength_profile",
                "synthetic_mask_modes",
                "synthetic_texture_source",
                "synthetic_texture_root",
                "synthetic_texture_images",
            }
            for field in evidence_fields:
                values = [metrics.get(field) for metrics in evidence_metrics]
                if any(value != values[0] for value in values[1:]):
                    raise ValueError(
                        f"Inconsistent threshold evidence field {field}: "
                        f"{source}"
                    )
            policy.update(
                {
                    "selection_data": SELECTION_DATA,
                    "normal_source_split": "train",
                    "dataset": dataset_name,
                    "category": category,
                    "shared_across_ablation_runs": list(ABLATIONS),
                    "source_heatmaps_identical_across_runs": True,
                    **{
                        field: evidence_metrics[0].get(field)
                        for field in evidence_fields
                    },
                }
            )
            for ablation in ABLATIONS:
                candidate = run_dir(spec, category, ablation)
                save_pixel_threshold_policy(
                    policy,
                    candidate / "pixel_threshold_policy.json",
                )
                evaluate_with_policy(candidate)
            report["categories"].append(
                {
                    "dataset": dataset_name,
                    "category": category,
                    **shared,
                    "threshold": policy["threshold"],
                    "observed_normal_pixel_fpr": policy[
                        "observed_normal_pixel_fpr"
                    ],
                    "synthetic_dice": policy["synthetic_dice"],
                    "runs_evaluated": len(ABLATIONS),
                }
            )
        summarize_dataset(spec)

    report["category_count"] = len(report["categories"])
    report["run_count"] = len(report["categories"]) * len(ABLATIONS)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), **report}, indent=2))


if __name__ == "__main__":
    main()
