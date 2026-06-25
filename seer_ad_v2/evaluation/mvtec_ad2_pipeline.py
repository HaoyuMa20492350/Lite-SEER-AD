from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from seer_ad_v2.data.mvtec_ad2_discovery import inspect_dataset_directory


OFFICIAL_CATEGORIES = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)
DEFAULT_SEEDS = (7, 13, 23)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def expected_public_runs(
    run_prefix: str = "mvtec_ad2_feature_first",
    categories: tuple[str, ...] = OFFICIAL_CATEGORIES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[dict[str, Any]]:
    return [
        {
            "category": category,
            "seed": seed,
            "model_run": f"{run_prefix}_{category}_seed{seed}_models",
            "public_run": f"{run_prefix}_{category}_seed{seed}_public",
        }
        for category in categories
        for seed in seeds
    ]


def checkpoint_template(
    run_prefix: str = "mvtec_ad2_feature_first",
    private_seed: int = 7,
) -> str:
    return (
        f"runs/{run_prefix}_{{category}}_seed{private_seed}_models/"
        "feature_prior.pt"
    )


def audit_pipeline(
    workspace: Path,
    *,
    run_prefix: str = "mvtec_ad2_feature_first",
    public_table_dir: Path = Path("tables/mvtec_ad2_feature_first"),
    submission_dir: Path = Path("submissions/mvtec_ad2_seed7_model256"),
    metadata_dir: Path = Path(
        "submissions/mvtec_ad2_seed7_model256_metadata"
    ),
    archive_path: Path = Path(
        "submissions/mvtec_ad2_seed7_model256.tar.gz"
    ),
    private_seed: int = 7,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    runs = expected_public_runs(run_prefix)
    checkpoints = [
        workspace / "runs" / row["model_run"] / "feature_prior.pt"
        for row in runs
    ]
    predictions = [
        workspace / "runs" / row["public_run"] / "predictions.npz"
        for row in runs
    ]
    metrics = [
        workspace / "runs" / row["public_run"] / "metrics.json"
        for row in runs
    ]
    table_root = workspace / public_table_dir
    table_rows = read_csv(table_root / "table_public_runs.csv")
    protocol = read_json(table_root / "protocol.json")
    metadata_root = workspace / metadata_dir
    submission_protocol = read_json(metadata_root / "submission_protocol.json")
    submission_root = workspace / submission_dir
    archive = workspace / archive_path
    checker = (
        workspace
        / "official_mvtec_ad2_utils"
        / "MVTecAD2_public_code_utils"
        / "check_and_prepare_data_for_upload.py"
    )
    dataset_root = workspace / "SEER-AD-dataset" / "MVTec-AD2"
    dataset_audit = inspect_dataset_directory(dataset_root)
    dataset_ready = bool(dataset_root.is_dir() and dataset_audit["ready"])
    private_checkpoints = [
        workspace
        / checkpoint_template(run_prefix, private_seed).format(
            category=category
        )
        for category in OFFICIAL_CATEGORIES
    ]
    expected_pairs = {
        (str(row["category"]), int(row["seed"])) for row in runs
    }
    observed_pairs = set()
    for row in table_rows:
        try:
            observed_pairs.add((str(row["category"]), int(row["seed"])))
        except (KeyError, TypeError, ValueError):
            continue
    public_complete = (
        len(table_rows) == len(runs)
        and observed_pairs == expected_pairs
        and protocol.get("runs_completed") == len(runs)
        and protocol.get("categories") == list(OFFICIAL_CATEGORIES)
        and protocol.get("seeds") == list(DEFAULT_SEEDS)
        and all(path.is_file() for path in checkpoints)
        and all(path.is_file() for path in predictions)
        and all(path.is_file() for path in metrics)
    )
    checker_passed = (
        (submission_protocol.get("official_checker") or {}).get("status")
        == "passed"
    )
    private_complete = (
        submission_protocol.get("files") == 4090
        and submission_protocol.get("full_official_submission") is True
        and checker_passed
        and archive.is_file()
    )
    return {
        "dataset_ready": dataset_ready,
        "dataset_matched_paths": dataset_audit["matched_paths"],
        "dataset_required_paths": dataset_audit["required_paths"],
        "official_checker_ready": checker.is_file(),
        "expected_public_runs": len(runs),
        "model_checkpoints": sum(path.is_file() for path in checkpoints),
        "public_predictions": sum(path.is_file() for path in predictions),
        "public_metrics": sum(path.is_file() for path in metrics),
        "public_table_rows": len(table_rows),
        "public_category_seed_pairs": len(observed_pairs),
        "public_protocol_runs": protocol.get("runs_completed", 0),
        "public_complete": public_complete,
        "private_seed": private_seed,
        "private_checkpoints": sum(
            path.is_file() for path in private_checkpoints
        ),
        "submission_root_exists": submission_root.is_dir(),
        "submission_files": submission_protocol.get("files", 0),
        "official_checker_status": (
            submission_protocol.get("official_checker") or {}
        ).get("status"),
        "archive_exists": archive.is_file(),
        "private_complete": private_complete,
        "pipeline_complete": public_complete and private_complete,
    }


def public_command(
    python: str,
    *,
    config: str,
    run_prefix: str,
    out: str,
    log_file: str,
    device: str,
) -> list[str]:
    return [
        python,
        "tools/run_mvtec_ad2_public.py",
        "--config",
        config,
        "--categories",
        "all",
        "--seeds",
        "7,13,23",
        "--run-prefix",
        run_prefix,
        "--out",
        out,
        "--device",
        device,
        "--resume",
        "--log-file",
        log_file,
        "--quiet",
    ]


def private_command(
    python: str,
    *,
    config: str,
    run_prefix: str,
    private_seed: int,
    out: str,
    metadata_out: str,
    archive_out: str,
    device: str,
    submission_resolution: str = "model",
) -> list[str]:
    return [
        python,
        "tools/export_mvtec_ad2_submission.py",
        "--config",
        config,
        "--checkpoint",
        checkpoint_template(run_prefix, private_seed),
        "--categories",
        "all",
        "--splits",
        "test_private,test_private_mixed",
        "--submission-resolution",
        submission_resolution,
        "--out",
        out,
        "--metadata-out",
        metadata_out,
        "--archive-out",
        archive_out,
        "--device",
        device,
    ]
