from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.score_aggregation import (
    IMAGE_SCORE_MODES,
    image_scores_from_heatmaps,
)
from seer_ad_v2.evaluation.heatmap_fusion import fuse_heatmaps
from seer_ad_v2.evaluation.synthetic_validation import (
    evaluate_synthetic_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one image-score aggregation mode per category using only "
            "normal and deterministic synthetic evidence, then audit the "
            "frozen mode on held-out predictions."
        )
    )
    parser.add_argument(
        "--selection-root",
        default="tables/synthetic_gate_fusion_aggregate_mvtec15",
    )
    parser.add_argument(
        "--out",
        default="tables/image_score_aggregation_mvtec15",
    )
    parser.add_argument("--modes", default=",".join(IMAGE_SCORE_MODES))
    parser.add_argument("--materialize-evidence", action="store_true")
    parser.add_argument("--refresh-evidence", action="store_true")
    parser.add_argument("--max-normal-images", type=int, default=16)
    parser.add_argument("--synthetic-variants", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def split_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(modes) - set(IMAGE_SCORE_MODES))
    if unknown:
        raise ValueError(f"Unknown image-score modes: {', '.join(unknown)}")
    if not modes:
        raise ValueError("At least one image-score mode is required")
    return modes


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_run_args(run_dir: Path) -> dict[str, Any]:
    payload = json.loads(
        (run_dir / "run_args.json").read_text(encoding="utf-8")
    )
    run_args = payload.get("args", {})
    if not isinstance(run_args, dict):
        raise ValueError(f"Invalid run_args.json in {run_dir}")
    return run_args


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part) for part in value.replace(",", " ").split()]
    if not seeds:
        raise ValueError("Selection row has no synthetic evidence seeds")
    return seeds


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def synthetic_mode_metrics(
    clean_score_heatmaps: np.ndarray,
    synthetic_score_heatmaps: np.ndarray,
    mode: str,
) -> dict[str, float]:
    clean = image_scores_from_heatmaps(clean_score_heatmaps, mode=mode)
    synthetic = image_scores_from_heatmaps(
        synthetic_score_heatmaps,
        mode=mode,
    )
    labels = np.concatenate(
        [
            np.zeros(len(clean), dtype=np.uint8),
            np.ones(len(synthetic), dtype=np.uint8),
        ]
    )
    scores = np.concatenate([clean, synthetic])
    clean_median = float(np.median(clean))
    clean_mad = float(np.median(np.abs(clean - clean_median)))
    margin = float(
        (np.median(synthetic) - np.percentile(clean, 95))
        / max(clean_mad, 1e-8)
    )
    return {
        "synthetic_image_auroc": safe_auc(labels, scores),
        "clean_score_mean": float(np.mean(clean)),
        "clean_score_p95": float(np.percentile(clean, 95)),
        "synthetic_score_mean": float(np.mean(synthetic)),
        "robust_margin": margin,
    }


def choose_mode(
    evidence_rows: list[dict[str, Any]],
    modes: list[str],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence_rows:
        grouped[str(row["mode"])].append(row)
    candidates = []
    for mode in modes:
        rows = grouped[mode]
        aucs = np.asarray(
            [float(row["synthetic_image_auroc"]) for row in rows],
            dtype=np.float64,
        )
        margins = np.asarray(
            [float(row["robust_margin"]) for row in rows],
            dtype=np.float64,
        )
        candidates.append(
            {
                "mode": mode,
                "mean_synthetic_image_auroc": float(np.mean(aucs)),
                "min_seed_synthetic_image_auroc": float(np.min(aucs)),
                "mean_robust_margin": float(np.mean(margins)),
                "evidence_seeds": len(rows),
            }
        )
    order = {mode: index for index, mode in enumerate(modes)}
    return max(
        candidates,
        key=lambda row: (
            row["mean_synthetic_image_auroc"],
            row["min_seed_synthetic_image_auroc"],
            row["mean_robust_margin"],
            -order[str(row["mode"])],
        ),
    )


def evidence_needs_refresh(path: Path, *, force: bool) -> bool:
    if force or not path.exists():
        return True
    with np.load(path) as payload:
        return not {
            "clean_score_heatmaps",
            "synthetic_score_heatmaps",
        }.issubset(payload.files)


def aligned_array(
    source_a: dict[str, np.ndarray],
    source_b: dict[str, np.ndarray],
    key: str,
) -> np.ndarray:
    if key not in source_a or key not in source_b:
        raise KeyError(f"Fusion evidence is missing alignment key: {key}")
    if not np.array_equal(source_a[key], source_b[key]):
        raise ValueError(f"Fusion evidence differs for alignment key: {key}")
    return source_b[key]


def fuse_evidence_payloads(
    source_a: dict[str, np.ndarray],
    source_b: dict[str, np.ndarray],
    *,
    weight_a: float,
    scale_a: tuple[float, float],
    scale_b: tuple[float, float],
    image_score_mode: str,
    seed: int,
) -> dict[str, np.ndarray]:
    for key in ("paths", "variant_ids", "mask_modes"):
        aligned_array(source_a, source_b, key)
    fused_arrays = {}
    target_shape = tuple(source_b["clean_heatmaps"].shape[1:])
    for key in (
        "clean_heatmaps",
        "synthetic_heatmaps",
        "flipped_synthetic_heatmaps",
        "photometric_synthetic_heatmaps",
    ):
        if key not in source_a or key not in source_b:
            raise KeyError(f"Fusion evidence is missing heatmap array: {key}")
        fused_arrays[key] = fuse_heatmaps(
            source_a[key],
            source_b[key],
            weight_a=weight_a,
            scale_a=scale_a,
            scale_b=scale_b,
            target_shape=target_shape,
        )
    clean_scores = image_scores_from_heatmaps(
        fused_arrays["clean_heatmaps"],
        mode=image_score_mode,
    )
    synthetic_scores = image_scores_from_heatmaps(
        fused_arrays["synthetic_heatmaps"],
        mode=image_score_mode,
    )
    return {
        **fused_arrays,
        "clean_score_heatmaps": fused_arrays["clean_heatmaps"],
        "synthetic_score_heatmaps": fused_arrays["synthetic_heatmaps"],
        "synthetic_masks": source_b["synthetic_masks"],
        "clean_image_scores": clean_scores.astype(np.float32),
        "synthetic_image_scores": synthetic_scores.astype(np.float32),
        "paths": source_b["paths"],
        "variant_ids": source_b["variant_ids"],
        "mask_modes": source_b["mask_modes"],
        "seed": np.asarray(seed, dtype=np.int64),
    }


def save_fusion_evidence(
    source_run: Path,
    source_a: Path,
    source_b: Path,
    seed: int,
    run_args: dict[str, Any],
) -> None:
    with np.load(
        source_a / f"synthetic_validation_seed{seed}.npz"
    ) as payload_a, np.load(
        source_b / f"synthetic_validation_seed{seed}.npz"
    ) as payload_b:
        arrays = fuse_evidence_payloads(
            {key: payload_a[key] for key in payload_a.files},
            {key: payload_b[key] for key in payload_b.files},
            weight_a=float(run_args["fusion_weight_a"]),
            scale_a=tuple(float(value) for value in run_args["normal_scale_a"]),
            scale_b=tuple(float(value) for value in run_args["normal_scale_b"]),
            image_score_mode=str(
                run_args.get("image_score_mode") or "top5"
            ),
            seed=seed,
        )
    out = source_run / f"synthetic_validation_seed{seed}.npz"
    np.savez_compressed(out, **arrays)
    metrics = evaluate_synthetic_validation(
        arrays["clean_heatmaps"],
        arrays["synthetic_heatmaps"],
        arrays["synthetic_masks"],
        arrays["clean_image_scores"],
        arrays["synthetic_image_scores"],
        [
            arrays["flipped_synthetic_heatmaps"],
            arrays["photometric_synthetic_heatmaps"],
        ],
    )
    metrics.update(
        {
            "candidate_run": str(source_run),
            "seed": seed,
            "selection_data": "normal_images_plus_synthetic_masks",
            "uses_real_anomaly_labels_for_selection": False,
            "uses_real_anomaly_masks_for_selection": False,
            "fusion_source_a": str(source_a),
            "fusion_source_b": str(source_b),
            "fusion_weight_a": float(run_args["fusion_weight_a"]),
            "normal_scale_a": run_args["normal_scale_a"],
            "normal_scale_b": run_args["normal_scale_b"],
            "pixel_heatmap_source": "normal_calibrated_fusion",
            "image_score_source": "normal_calibrated_fusion",
            "image_score_mode": str(
                run_args.get("image_score_mode") or "top5"
            ),
        }
    )
    (source_run / f"synthetic_validation_seed{seed}_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def materialize_evidence(
    source_run: Path,
    seeds: list[int],
    args: argparse.Namespace,
    completed: set[tuple[Path, int]] | None = None,
) -> None:
    completed = completed if completed is not None else set()
    run_args = read_run_args(source_run)
    is_fusion = (
        run_args.get("image_score_source") == "normal_calibrated_fusion"
        or run_args.get("pixel_heatmap_source") == "normal_calibrated_fusion"
    )
    if is_fusion:
        required = {
            "fusion_source_a",
            "fusion_source_b",
            "fusion_weight_a",
            "normal_scale_a",
            "normal_scale_b",
        }
        missing = required - set(run_args)
        if missing:
            raise KeyError(
                f"{source_run} is missing fusion settings: {sorted(missing)}"
            )
        source_a = repo_path(str(run_args["fusion_source_a"]))
        source_b = repo_path(str(run_args["fusion_source_b"]))
        materialize_evidence(source_a, seeds, args, completed)
        materialize_evidence(source_b, seeds, args, completed)

    for seed in seeds:
        cache_key = (source_run.resolve(), seed)
        if cache_key in completed:
            continue
        out = source_run / f"synthetic_validation_seed{seed}.npz"
        if not evidence_needs_refresh(
            out,
            force=bool(args.refresh_evidence),
        ):
            completed.add(cache_key)
            continue
        if is_fusion:
            save_fusion_evidence(
                source_run,
                source_a,
                source_b,
                seed,
                run_args,
            )
            completed.add(cache_key)
            continue
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools/materialize_synthetic_normal_validation.py"),
                "--candidate-run-dir",
                str(source_run),
                "--out",
                str(out),
                "--seed",
                str(seed),
                "--max-normal-images",
                str(args.max_normal_images),
                "--synthetic-variants",
                str(args.synthetic_variants),
                "--batch-size",
                str(args.batch_size),
                "--device",
                str(args.device),
            ],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        completed.add(cache_key)


def main() -> None:
    args = parse_args()
    modes = split_modes(args.modes)
    selection_root = Path(args.selection_root)
    selection_paths = sorted(selection_root.glob("seed*/selection.csv"))
    if not selection_paths:
        raise FileNotFoundError(
            f"No seed selection files found under {selection_root}"
        )
    selections = {
        path.parent.name: {row["category"]: row for row in read_csv(path)}
        for path in selection_paths
    }
    categories = sorted(next(iter(selections.values())))
    for seed_name, rows in selections.items():
        if set(rows) != set(categories):
            raise ValueError(f"Category mismatch in {seed_name}")

    evidence_table = []
    category_table = []
    heldout_table = []
    materialized: set[tuple[Path, int]] = set()
    for category in categories:
        category_rows = [rows[category] for rows in selections.values()]
        selected_candidates = {row["selected_candidate"] for row in category_rows}
        source_runs = {row["selected_run"] for row in category_rows}
        if len(selected_candidates) != 1 or len(source_runs) != 1:
            raise ValueError(
                f"{category} selection is not stable across held-out seeds"
            )
        source_run = repo_path(next(iter(source_runs)))
        seeds = parse_seeds(category_rows[0]["selection_evidence_seeds"])
        if args.materialize_evidence:
            materialize_evidence(source_run, seeds, args, materialized)

        category_evidence = []
        for seed in seeds:
            artifact = source_run / f"synthetic_validation_seed{seed}.npz"
            if not artifact.exists():
                raise FileNotFoundError(
                    f"Missing synthetic evidence for {category}: {artifact}"
                )
            with np.load(artifact) as payload:
                required = {
                    "clean_score_heatmaps",
                    "synthetic_score_heatmaps",
                }
                missing = required - set(payload.files)
                if missing:
                    raise KeyError(
                        f"{artifact} is missing arrays: {sorted(missing)}"
                    )
                for mode in modes:
                    metrics = synthetic_mode_metrics(
                        payload["clean_score_heatmaps"],
                        payload["synthetic_score_heatmaps"],
                        mode,
                    )
                    row = {
                        "category": category,
                        "selected_candidate": next(iter(selected_candidates)),
                        "source_run": str(source_run),
                        "evidence_seed": seed,
                        "mode": mode,
                        **metrics,
                    }
                    evidence_table.append(row)
                    category_evidence.append(row)

        chosen = choose_mode(category_evidence, modes)
        current_mode = read_run_args(source_run).get("image_score_mode", "")
        category_results = []
        for split_seed, rows in selections.items():
            selection = rows[category]
            heldout = (
                selection_root
                / split_seed
                / "selected_runs"
                / (
                    f"{category}_{selection['selected_candidate']}_heldout"
                )
            )
            with np.load(heldout / "predictions.npz") as prediction:
                score_key = (
                    "image_score_heatmaps"
                    if "image_score_heatmaps" in prediction.files
                    else "score_heatmaps"
                )
                heatmaps = prediction[score_key]
                labels = prediction["labels"]
                selected_scores = image_scores_from_heatmaps(
                    heatmaps,
                    mode=str(chosen["mode"]),
                )
                current_scores = prediction["image_scores"]
                result = {
                    "category": category,
                    "split_seed": split_seed,
                    "selected_candidate": selection["selected_candidate"],
                    "selected_mode": chosen["mode"],
                    "current_mode": current_mode,
                    "score_heatmap_key": score_key,
                    "selected_image_auroc": safe_auc(
                        labels,
                        selected_scores,
                    ),
                    "current_image_auroc": safe_auc(
                        labels,
                        current_scores,
                    ),
                    "uses_real_anomaly_labels_for_selection": False,
                    "uses_real_anomaly_masks_for_selection": False,
                }
                result["delta_selected_minus_current"] = (
                    result["selected_image_auroc"]
                    - result["current_image_auroc"]
                )
                heldout_table.append(result)
                category_results.append(result)
        category_table.append(
            {
                "category": category,
                "selected_candidate": next(iter(selected_candidates)),
                "selected_mode": chosen["mode"],
                "current_mode": current_mode,
                "mean_synthetic_image_auroc": chosen[
                    "mean_synthetic_image_auroc"
                ],
                "min_seed_synthetic_image_auroc": chosen[
                    "min_seed_synthetic_image_auroc"
                ],
                "mean_robust_margin": chosen["mean_robust_margin"],
                "heldout_current_image_auroc": float(
                    np.mean(
                        [
                            row["current_image_auroc"]
                            for row in category_results
                        ]
                    )
                ),
                "heldout_selected_image_auroc": float(
                    np.mean(
                        [
                            row["selected_image_auroc"]
                            for row in category_results
                        ]
                    )
                ),
                "heldout_delta": float(
                    np.mean(
                        [
                            row["delta_selected_minus_current"]
                            for row in category_results
                        ]
                    )
                ),
                "uses_real_anomaly_labels_for_selection": False,
                "uses_real_anomaly_masks_for_selection": False,
            }
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_synthetic_mode_evidence.csv", evidence_table)
    write_csv(out_dir / "table_heldout_mode_results.csv", heldout_table)
    write_csv(out_dir / "table_category_summary.csv", category_table)
    current_mean = float(
        np.mean([row["current_image_auroc"] for row in heldout_table])
    )
    selected_mean = float(
        np.mean([row["selected_image_auroc"] for row in heldout_table])
    )
    ranked_changes = sorted(
        category_table,
        key=lambda row: float(row["heldout_delta"]),
        reverse=True,
    )
    adoption_decision = (
        "adopt_label_free_selected_modes"
        if selected_mean > current_mean
        else "retain_current_top5"
    )
    summary = {
        "selection_protocol": (
            "normal_synthetic_image_score_aggregation_v1"
        ),
        "adoption_decision": adoption_decision,
        "decision_rule": (
            "Adopt only if the predeclared label-free selector improves the "
            "mean held-out Image AUROC; otherwise retain the frozen current "
            "mode. Held-out labels are audit-only and are not used to tune a "
            "post-hoc switching threshold."
        ),
        "categories": len(category_table),
        "heldout_runs": len(heldout_table),
        "modes": modes,
        "current_mean_image_auroc": current_mean,
        "selected_mean_image_auroc": selected_mean,
        "mean_delta": selected_mean - current_mean,
        "improved_categories": int(
            sum(row["heldout_delta"] > 0 for row in category_table)
        ),
        "degraded_categories": int(
            sum(row["heldout_delta"] < 0 for row in category_table)
        ),
        "unchanged_categories": int(
            sum(row["heldout_delta"] == 0 for row in category_table)
        ),
        "uses_real_anomaly_labels_for_selection": False,
        "uses_real_anomaly_masks_for_selection": False,
        "category_results": category_table,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report = [
        "# Label-Free Image-Score Aggregation Audit",
        "",
        "## Protocol",
        "",
        "- Candidate modes: " + ", ".join(f"`{mode}`" for mode in modes) + ".",
        "- Selection evidence: retained normal images plus deterministic "
        "synthetic defects at seeds `7/13/23`.",
        "- Real anomaly labels and masks are excluded from mode selection.",
        "- Frozen modes are audited on 45 held-out runs after selection.",
        "",
        "## Result",
        "",
        f"- Current frozen `top5`: `{current_mean:.6f}` mean Image AUROC.",
        f"- Label-free selected modes: `{selected_mean:.6f}`.",
        f"- Delta: `{selected_mean - current_mean:+.6f}`.",
        f"- Category outcomes: `{summary['improved_categories']}` improved, "
        f"`{summary['degraded_categories']}` degraded, "
        f"`{summary['unchanged_categories']}` unchanged.",
        f"- Adoption decision: `{adoption_decision}`.",
        "",
        "The synthetic-domain gain is not a reliable proxy for real-defect "
        "gain. A switching threshold must not be tuned after inspecting these "
        "held-out labels; that would introduce meta-level test leakage.",
        "",
        "## Largest Category Changes",
        "",
        "| Category | Selected mode | Held-out delta |",
        "|---|---|---:|",
    ]
    for row in ranked_changes[:4] + ranked_changes[-4:]:
        report.append(
            f"| {row['category']} | `{row['selected_mode']}` | "
            f"{float(row['heldout_delta']):+.6f} |"
        )
    report.extend(
        [
            "",
            "## Evidence",
            "",
            "- `table_synthetic_mode_evidence.csv`",
            "- `table_heldout_mode_results.csv`",
            "- `table_category_summary.csv`",
            "- `summary.json`",
            "",
        ]
    )
    (out_dir / "analysis.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
