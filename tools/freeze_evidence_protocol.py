from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze the paper-facing Lite-SEER-AD protocol and workspace evidence.")
    p.add_argument("--out", required=True)
    p.add_argument("--config", action="append", default=[])
    p.add_argument("--candidate", action="append", default=[])
    p.add_argument("--evidence", action="append", default=[])
    p.add_argument("--seeds", default="7,13,23")
    return p.parse_args()


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).strip()


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(value: str) -> dict[str, Any]:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256(path),
    }


def untracked_records() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    records = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="replace")
        records.append(file_record(relative))
    return records


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = command_output(["git", "status", "--short"])
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    ).stdout
    configs = args.config or [
        "configs/mvtec.yaml",
        "configs/visa.yaml",
        "configs/mpdd.yaml",
        "configs/mvtec_ad2.yaml",
    ]
    candidates = args.candidate or [
        "pixelraw",
        "highres256",
        "post_highres_gaussian3",
        "fixed_train_normal_calibration",
        "resnet18_multilayer_padim",
        "normal_calibrated_highres_pixelraw_fusion",
        "normal_calibrated_padim_highres_fusion",
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_protocol": (
            "normal_plus_synthetic_cross_seed_mean_"
            "no_real_anomaly_labels"
        ),
        "oracle_protocol": "validation_mask_selector_upper_bound_only",
        "feature_first_is_main": True,
        "diffusion_first_role": "ablation_only",
        "uses_real_anomaly_labels_for_main_selection": False,
        "uses_real_anomaly_masks_for_main_selection": False,
        "seeds": [int(value.strip()) for value in args.seeds.split(",") if value.strip()],
        "git": {
            "commit": command_output(["git", "rev-parse", "HEAD"]),
            "branch": command_output(["git", "branch", "--show-current"]),
            "dirty": bool(status),
            "status": status.splitlines(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_files": untracked_records(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]).splitlines(),
            "gpu": command_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
            ),
        },
        "files": {
            "requirements": file_record("requirements.txt"),
            "configs": [file_record(value) for value in configs],
            "evidence": [file_record(value) for value in args.evidence],
        },
        "candidate_templates": candidates,
        "dataset_split_contract": {
            "selection": "held-out normal plus deterministic synthetic defects",
            "evaluation": "independent held-out split per seed",
            "selection_evidence_seeds": [7, 13, 23],
            "datasets": ["mvtec15", "visa", "mpdd"],
            "mvtec_ad2": {
                "train": ["train", "validation"],
                "public_evaluation": "test_public",
                "private_submission": [
                    "test_private",
                    "test_private_mixed",
                ],
            },
        },
        "prediction_schema": {
            "canonical_heatmap_keys": [
                "detection_heatmaps",
                "verification_heatmaps",
                "image_score_heatmaps",
            ],
            "legacy_aliases_retained": [
                "heatmaps",
                "final_heatmaps",
                "score_heatmaps",
            ],
        },
        "selection_contract": {
            "artifact": "synthetic_validation_seed{seed}.npz",
            "metrics": "synthetic_validation_seed{seed}_metrics.json",
            "selection_inputs": [
                "held-out normal images",
                "deterministic synthetic defect masks",
                "normal false-positive statistics",
                "augmentation stability",
                "measured candidate latency",
            ],
            "reporting_only_inputs": ["real held-out anomaly labels and masks"],
        },
        "mvtec_ad2_submission_contract": {
            "official_checker_required": True,
            "private_samples": 4090,
            "output_files": 8180,
            "submission_root_entries": [
                "anomaly_images",
                "anomaly_images_thresholded",
            ],
            "metadata_must_be_outside_submission_root": True,
        },
        "mvtec_ad2_acquisition_contract": {
            "official_form": (
                "https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2"
            ),
            "required_user_inputs": [
                "first_name",
                "last_name",
                "email",
                "explicit_non_commercial_acceptance",
            ],
            "newsletter_default": False,
            "no_automatic_submission_without_explicit_flag": True,
            "reports_redact_email_and_do_not_store_download_tokens": True,
            "installer_rejects": [
                "path_traversal",
                "non_regular_tar_members",
                "merge_into_nonempty_target",
            ],
            "orchestration": {
                "expected_public_runs": 24,
                "dataset_required_paths": 56,
                "private_seed": 7,
                "private_requires_public_complete": True,
                "execution_requires_explicit_flag": True,
            },
            "official_result_import": {
                "accepted_formats": ["json", "csv"],
                "required_source_host": "benchmark.mvtec.com",
                "required_metrics": [
                    "image_auroc",
                    "pixel_auroc",
                    "aupro",
                ],
                "requires_checker_passed_local_archive": True,
                "requires_exact_public_category_seed_pairs": 24,
                "automatic_sota_claim": False,
                "strict_sota_rule": (
                    "submission row exactly matches official result and is "
                    "strictly best on image_auroc, pixel_auroc, and aupro in "
                    "a dated official leaderboard snapshot"
                ),
            },
        },
    }
    (out / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "git_status.txt").write_text(status + "\n", encoding="utf-8")
    (out / "environment_lock.txt").write_text(
        "\n".join(manifest["environment"]["pip_freeze"]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), "git_dirty": bool(status), "configs": configs}, indent=2))


if __name__ == "__main__":
    main()
