from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.evaluation.pixel_threshold_policy import POLICY_PROTOCOL


ABLATIONS = [
    "residual_only",
    "feature_only",
    "feature_hn_sev",
    "feature_hn_sev_crv",
    "feature_tuned_crv",
    "feature_fixed10",
    "feature_fixed25",
    "feature_rule_brds",
    "utility_lc_rds",
]
DATASETS = {
    "mvtec": {
        "config": Path("configs/mvtec.yaml"),
        "run_prefix": "fulltest_mvtec15",
        "model_prefix": "fulltest_mvtec15",
        "tables": "tables/fulltest_mvtec15",
        "expected_categories": 15,
        "expected_images": 1725,
        "require_full_base_training": False,
    },
    "visa": {
        "config": Path("configs/visa.yaml"),
        "run_prefix": "fulltest_visa",
        "model_prefix": "fulltest_visa",
        "tables": "tables/fulltest_visa",
        "expected_categories": 12,
        "expected_images": 2162,
        "require_full_base_training": True,
    },
    "mpdd": {
        "config": Path("configs/mpdd.yaml"),
        "run_prefix": "fulltest_mpdd",
        "model_prefix": "fulltest_mpdd",
        "tables": "tables/fulltest_mpdd",
        "expected_categories": 6,
        "expected_images": 458,
        "require_full_base_training": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that every full-test run covers every official test image."
    )
    parser.add_argument(
        "--out",
        default="tables/full_test_coverage_audit.json",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def discover_categories(cfg: dict[str, Any]) -> list[str]:
    dataset = cfg.get("dataset", {}) or {}
    root = REPO_ROOT / str(dataset["root"])
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in {"archives", "split_csv"}
    )


def norm_path(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve()).replace("\\", "/").lower()


def expected_paths(cfg: dict[str, Any], category: str) -> set[str]:
    dataset = cfg.get("dataset", {}) or {}
    root = REPO_ROOT / str(dataset["root"])
    test = build_dataset(
        str(dataset["name"]),
        root,
        category,
        "test",
        int(dataset.get("image_size", 256)),
    )
    return {norm_path(row.image_path) for row in test.records}


def train_count(cfg: dict[str, Any], category: str) -> int:
    dataset = cfg.get("dataset", {}) or {}
    root = REPO_ROOT / str(dataset["root"])
    train = build_dataset(
        str(dataset["name"]),
        root,
        category,
        "train",
        int(dataset.get("image_size", 256)),
    )
    return len(train.records)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def audit_models(
    spec: dict[str, Any],
    category: str,
    expected_train: int,
) -> dict[str, Any]:
    model_dir = (
        REPO_ROOT / "runs" / f"{spec['model_prefix']}_{category}_models"
    )
    checkpoints = {
        name: (model_dir / name).exists()
        for name in ("diffusion.pt", "feature_prior.pt", "hn_sev.pt")
    }
    diffusion = load_json(model_dir / "metrics.json")
    feature = load_json(model_dir / "feature_prior_metrics.json")
    hn_sev = load_json(model_dir / "hn_sev_metrics.json")
    base_training_complete = True
    if spec["require_full_base_training"]:
        base_training_complete = (
            diffusion.get("num_train") == expected_train
            and feature.get("train_images") == expected_train
        )
    hn_sev_complete = (
        hn_sev.get("normal_train_images") == expected_train
        and hn_sev.get("texture_images") == 5640
    )
    return {
        "model_dir": str(model_dir),
        "checkpoints": checkpoints,
        "expected_train_images": expected_train,
        "diffusion_train_images": diffusion.get("num_train"),
        "feature_prior_train_images": feature.get("train_images"),
        "hn_sev_train_images": hn_sev.get("normal_train_images"),
        "hn_sev_texture_images": hn_sev.get("texture_images"),
        "base_training_complete": base_training_complete,
        "hn_sev_complete": hn_sev_complete,
        "complete": (
            all(checkpoints.values())
            and base_training_complete
            and hn_sev_complete
        ),
    }


def audit_run(run_dir: Path, expected: set[str]) -> dict[str, Any]:
    predictions = run_dir / "predictions.npz"
    if not predictions.exists():
        return {
            "run": run_dir.name,
            "status": "missing",
            "expected_images": len(expected),
            "actual_images": 0,
            "missing_images": len(expected),
            "extra_images": 0,
            "duplicate_images": 0,
            "complete": False,
        }
    with np.load(predictions) as data:
        paths = [norm_path(str(path)) for path in data["paths"]]
        labels = len(data["labels"])
        masks = len(data["masks"])
        heatmaps = len(data["heatmaps"])
        image_scores = len(data["image_scores"])
    actual = set(paths)
    missing = expected - actual
    extra = actual - expected
    duplicates = len(paths) - len(actual)
    policy = load_json(run_dir / "pixel_threshold_policy.json")
    metrics = load_json(run_dir / "metrics.json")
    threshold_complete = (
        policy.get("protocol") == POLICY_PROTOCOL
        and policy.get("selection_data")
        == "official_train_normal_images_plus_synthetic_masks"
        and policy.get("normal_source_split") == "train"
        and policy.get("synthetic_variants") == 2
        and policy.get("synthetic_strength_profile") == "standard"
        and set(policy.get("synthetic_mask_modes", []))
        == {"blob", "scratch"}
        and policy.get("synthetic_texture_source")
        == "deterministic_random"
        and policy.get("synthetic_texture_images") == 0
        and policy.get("uses_real_anomaly_labels") is False
        and policy.get("uses_real_anomaly_masks") is False
        and metrics.get("threshold_protocol") == POLICY_PROTOCOL
        and metrics.get("threshold_uses_real_anomaly_labels") is False
        and metrics.get("threshold_uses_real_anomaly_masks") is False
    )
    complete = (
        not missing
        and not extra
        and duplicates == 0
        and len(paths) == len(expected)
        and labels == masks == heatmaps == image_scores == len(expected)
        and threshold_complete
    )
    return {
        "run": run_dir.name,
        "status": "complete" if complete else "incomplete",
        "expected_images": len(expected),
        "actual_images": len(paths),
        "missing_images": len(missing),
        "extra_images": len(extra),
        "duplicate_images": duplicates,
        "array_lengths": {
            "labels": labels,
            "masks": masks,
            "heatmaps": heatmaps,
            "image_scores": image_scores,
        },
        "threshold_protocol": metrics.get("threshold_protocol"),
        "threshold_selection_data": policy.get("selection_data"),
        "threshold_complete": threshold_complete,
        "complete": complete,
    }


def audit_dataset(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config(REPO_ROOT / spec["config"])
    run_rows: list[dict[str, Any]] = []
    image_total = 0
    model_rows: list[dict[str, Any]] = []
    categories = discover_categories(cfg)
    for category in categories:
        expected = expected_paths(cfg, category)
        expected_train = train_count(cfg, category)
        image_total += len(expected)
        model_row = audit_models(spec, category, expected_train)
        model_row["category"] = category
        model_rows.append(model_row)
        for ablation in ABLATIONS:
            run_dir = (
                REPO_ROOT
                / "runs"
                / f"{spec['run_prefix']}_{category}_{ablation}"
            )
            row = audit_run(run_dir, expected)
            row.update(
                {
                    "dataset": name,
                    "category": category,
                    "ablation": ablation,
                }
            )
            run_rows.append(row)
    complete_runs = sum(row["complete"] for row in run_rows)
    expected_runs = len(categories) * len(ABLATIONS)
    return {
        "dataset": name,
        "categories": len(categories),
        "expected_categories": spec["expected_categories"],
        "test_images": image_total,
        "expected_test_images": spec["expected_images"],
        "expected_runs": expected_runs,
        "complete_runs": complete_runs,
        "expected_image_records": image_total * len(ABLATIONS),
        "complete_models": sum(row["complete"] for row in model_rows),
        "models": model_rows,
        "complete": (
            len(categories) == spec["expected_categories"]
            and image_total == spec["expected_images"]
            and complete_runs == expected_runs
            and all(row["complete"] for row in model_rows)
        ),
        "runs": run_rows,
    }


def main() -> None:
    args = parse_args()
    rows = [audit_dataset(name, spec) for name, spec in DATASETS.items()]
    payload = {
        "ablations": ABLATIONS,
        "datasets": rows,
        "total_categories": sum(row["categories"] for row in rows),
        "total_test_images": sum(row["test_images"] for row in rows),
        "total_expected_runs": sum(row["expected_runs"] for row in rows),
        "total_complete_runs": sum(row["complete_runs"] for row in rows),
        "total_expected_image_records": sum(
            row["expected_image_records"] for row in rows
        ),
    }
    payload["complete"] = (
        all(row["complete"] for row in rows)
        and payload["total_categories"] == 33
        and payload["total_test_images"] == 4345
        and payload["total_expected_runs"] == 297
        and payload["total_expected_image_records"] == 39105
    )
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
