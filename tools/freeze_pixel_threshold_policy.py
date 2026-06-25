from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.pixel_threshold_policy import (
    save_pixel_threshold_policy,
    select_synthetic_normal_threshold,
)
from seer_ad_v2.evaluation.heatmap_fusion import fuse_heatmaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a pixel threshold from normal and deterministic synthetic "
            "validation artifacts, without using real anomaly labels or masks."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--synthetic-seeds", default="7,13,23")
    parser.add_argument(
        "--artifact-template",
        default="synthetic_validation_seed{seed}.npz",
    )
    parser.add_argument("--max-normal-fpr", type=float, default=0.005)
    parser.add_argument("--max-candidates", type=int, default=2048)
    parser.add_argument("--out", default=None)
    parser.add_argument("--evaluate-predictions", action="store_true")
    return parser.parse_args()


def _seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one synthetic seed is required")
    return seeds


def _load_artifact(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    required = {"clean_heatmaps", "synthetic_heatmaps", "synthetic_masks"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
    return {key: np.asarray(data[key]) for key in required}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_threshold_artifact(
    run_dir: Path,
    seed: int,
    artifact_template: str,
) -> dict[str, np.ndarray]:
    artifact_path = run_dir / artifact_template.format(seed=seed)
    if artifact_path.exists():
        return _load_artifact(artifact_path)

    payload = _read_json(run_dir / "run_args.json")
    run_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    if not isinstance(run_args, dict):
        run_args = {}
    source_a_value = run_args.get("fusion_source_a")
    source_b_value = run_args.get("fusion_source_b")
    if not source_a_value or not source_b_value:
        raise FileNotFoundError(f"Missing synthetic threshold evidence: {artifact_path}")

    source_a = _resolve_repo_path(str(source_a_value))
    source_b = _resolve_repo_path(str(source_b_value))
    artifact_a = _load_threshold_artifact(source_a, seed, artifact_template)
    artifact_b = _load_threshold_artifact(source_b, seed, artifact_template)
    target_shape = tuple(artifact_b["clean_heatmaps"].shape[1:])
    scale_a = tuple(float(value) for value in run_args["normal_scale_a"])
    scale_b = tuple(float(value) for value in run_args["normal_scale_b"])
    weight_a = float(run_args["fusion_weight_a"])
    return {
        "clean_heatmaps": fuse_heatmaps(
            artifact_a["clean_heatmaps"],
            artifact_b["clean_heatmaps"],
            weight_a=weight_a,
            scale_a=scale_a,
            scale_b=scale_b,
            target_shape=target_shape,
        ),
        "synthetic_heatmaps": fuse_heatmaps(
            artifact_a["synthetic_heatmaps"],
            artifact_b["synthetic_heatmaps"],
            weight_a=weight_a,
            scale_a=scale_a,
            scale_b=scale_b,
            target_shape=target_shape,
        ),
        "synthetic_masks": artifact_b["synthetic_masks"],
    }


def freeze_run_policy(
    run_dir: Path,
    seeds: list[int],
    *,
    artifact_template: str = "synthetic_validation_seed{seed}.npz",
    max_normal_fpr: float = 0.005,
    max_candidates: int = 2048,
) -> dict[str, Any]:
    artifact_paths = [
        run_dir / artifact_template.format(seed=seed) for seed in seeds
    ]
    payloads = [
        _load_threshold_artifact(run_dir, seed, artifact_template)
        for seed in seeds
    ]
    policy: dict[str, Any] = select_synthetic_normal_threshold(
        np.concatenate([item["clean_heatmaps"] for item in payloads], axis=0),
        np.concatenate(
            [item["synthetic_heatmaps"] for item in payloads],
            axis=0,
        ),
        np.concatenate([item["synthetic_masks"] for item in payloads], axis=0),
        max_normal_fpr=max_normal_fpr,
        max_candidates=max_candidates,
    )
    policy.update(
        {
            "source_run": str(run_dir),
            "synthetic_seeds": seeds,
            "source_artifacts": [str(path) for path in artifact_paths],
            "source_artifacts_reconstructed": any(
                not path.exists() for path in artifact_paths
            ),
        }
    )
    return policy


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    seeds = _seeds(args.synthetic_seeds)
    policy = freeze_run_policy(
        run_dir,
        seeds,
        artifact_template=args.artifact_template,
        max_normal_fpr=args.max_normal_fpr,
        max_candidates=args.max_candidates,
    )
    out = Path(args.out) if args.out else run_dir / "pixel_threshold_policy.json"
    save_pixel_threshold_policy(policy, out)

    if args.evaluate_predictions:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "evaluate.py"),
                "--pred_dir",
                str(run_dir),
                "--pixel-threshold-policy",
                str(out),
                "--require-fixed-threshold",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    print(json.dumps({"out": str(out), "policy": policy}, indent=2))


if __name__ == "__main__":
    main()
