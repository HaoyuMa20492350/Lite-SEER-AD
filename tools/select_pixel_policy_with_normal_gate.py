from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.prediction_schema import (
    prediction_heatmap_payload,
    resolve_prediction_heatmaps,
)
from seer_ad_v2.evaluation.synthetic_validation import synthetic_normal_utility


METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select a pixel policy using only normal validation images or a "
            "label-free candidate prior, then evaluate on a held-out split."
        )
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--categories", required=True, help="Comma-separated categories.")
    p.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate as name=run/template/with/{category}.",
    )
    p.add_argument(
        "--candidate-manifest",
        default=None,
        help="CSV with category,candidate,run rows for category-specific candidates.",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--val-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--normal-label", type=int, default=0)
    p.add_argument(
        "--gate-metric",
        choices=[
            "highres_if_available",
            "normal_score_p95",
            "normal_heatmap_p99",
            "normal_heatmap_cv",
            "normal_rank_sum",
            "normal_guarded_highres",
            "synthetic_normal_utility",
        ],
        default="normal_guarded_highres",
    )
    p.add_argument("--max-score-inflation", type=float, default=1.5)
    p.add_argument("--max-cv-inflation", type=float, default=1.4)
    p.add_argument("--synthetic-metrics-name", default="synthetic_validation_metrics.json")
    p.add_argument(
        "--aggregate-synthetic-seeds",
        default="",
        help=(
            "Comma-separated synthetic evidence seeds. When set, candidate "
            "utilities are computed from the mean metrics across these seeds."
        ),
    )
    p.add_argument("--allow-missing-synthetic", action="store_true")
    p.add_argument("--utility-pixel-ap-weight", type=float, default=0.30)
    p.add_argument("--utility-aupro-weight", type=float, default=0.25)
    p.add_argument("--utility-dice-weight", type=float, default=0.15)
    p.add_argument("--utility-pixel-auroc-weight", type=float, default=0.10)
    p.add_argument("--utility-image-auroc-weight", type=float, default=0.05)
    p.add_argument("--utility-stability-weight", type=float, default=0.15)
    p.add_argument("--utility-normal-fpr-penalty", type=float, default=0.20)
    p.add_argument("--utility-latency-penalty", type=float, default=0.02)
    p.add_argument("--utility-max-latency-ms", type=float, default=100.0)
    p.add_argument("--materialize", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_candidates(values: list[str]) -> list[tuple[str, str]]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Candidate must be name=template: {value}")
        name, template = value.split("=", 1)
        out.append((name.strip(), template.strip()))
    return out


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mean_numeric_metrics(
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = sorted(set().union(*(payload.keys() for payload in payloads)))
    result: dict[str, Any] = {}
    for key in keys:
        values = []
        for payload in payloads:
            raw = payload.get(key)
            if isinstance(raw, bool):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if values:
            result[key] = float(np.mean(values))
    result["aggregate_evidence_count"] = len(payloads)
    return result


def selection_protocol_name(
    gate_metric: str,
    aggregate_synthetic_seeds: list[int],
) -> str:
    if gate_metric != "synthetic_normal_utility":
        return "normal_only_no_real_anomaly_labels"
    if aggregate_synthetic_seeds:
        return (
            "normal_plus_synthetic_cross_seed_mean_"
            "no_real_anomaly_labels"
        )
    return "normal_plus_synthetic_no_real_anomaly_labels"


def stable_seed(seed: int, category: str, label: int) -> int:
    digest = hashlib.sha256(f"{seed}:{category}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def stratified_split(labels: np.ndarray, category: str, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    val_ratio = min(max(float(val_ratio), 0.1), 0.9)
    labels = labels.astype(np.uint8).reshape(-1)
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(set(int(v) for v in labels.tolist())):
        idx = np.where(labels == label)[0]
        rng = np.random.default_rng(stable_seed(seed, category, label))
        idx = idx.copy()
        rng.shuffle(idx)
        if len(idx) <= 1:
            val_count = len(idx)
        else:
            val_count = int(round(len(idx) * val_ratio))
            val_count = min(max(1, val_count), len(idx) - 1)
        val_parts.append(idx[:val_count])
        test_parts.append(idx[val_count:])
    val_idx = np.concatenate(val_parts) if val_parts else np.asarray([], dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.asarray([], dtype=np.int64)
    return np.sort(val_idx), np.sort(test_idx)


def assert_compatible(ref: np.lib.npyio.NpzFile, cur: np.lib.npyio.NpzFile, path: Path) -> None:
    if len(ref["labels"]) != len(cur["labels"]):
        raise ValueError(f"{path} has different sample count")
    if not np.array_equal(ref["labels"], cur["labels"]):
        raise ValueError(f"{path} labels do not match the first candidate")
    if "paths" in ref.files and "paths" in cur.files and not np.array_equal(ref["paths"], cur["paths"]):
        raise ValueError(f"{path} paths do not match the first candidate")


def metrics(pred: np.lib.npyio.NpzFile, indices: np.ndarray) -> dict[str, float]:
    return detection_metrics(
        pred["labels"][indices],
        pred["image_scores"][indices],
        pred["masks"][indices],
        pred["heatmaps"][indices],
    )


def row(prefix: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    out = dict(prefix)
    for metric in METRICS:
        out[metric] = values.get(metric)
    out["f1"] = values.get("f1")
    out["iou"] = values.get("iou")
    out["threshold"] = values.get("threshold")
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def materialize_selected(
    out_dir: Path,
    category: str,
    selected_name: str,
    selected_path: Path,
    pred: np.lib.npyio.NpzFile,
    test_idx: np.ndarray,
    selection_evidence: dict[str, Any],
) -> None:
    run_dir = out_dir / "selected_runs" / f"{category}_{selected_name}_heldout"
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_metrics = metrics(pred, test_idx)
    detection, verification, image_score_heatmaps = resolve_prediction_heatmaps(
        pred
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        labels=pred["labels"][test_idx],
        image_scores=pred["image_scores"][test_idx],
        masks=pred["masks"][test_idx],
        **prediction_heatmap_payload(
            detection[test_idx],
            verification[test_idx],
            image_score_heatmaps[test_idx],
        ),
        paths=pred["paths"][test_idx] if "paths" in pred.files else test_idx.astype(str),
        ablation=np.asarray("heldout_selected_normal_gate"),
    )
    source_args = read_json(selected_path / "run_args.json")
    payload_args = source_args.get("args", {}) if isinstance(source_args, dict) else {}
    if isinstance(payload_args, dict):
        payload_args = dict(payload_args)
        payload_args["run_name"] = run_dir.name
        payload_args["ablation"] = "heldout_selected_normal_gate"
        payload_args["selected_policy"] = selected_name
        payload_args["selected_source_run"] = str(selected_path)
        payload_args["selection_protocol"] = selection_evidence.get("selection_protocol")
        payload_args["uses_real_anomaly_labels_for_selection"] = False
        payload_args["uses_real_anomaly_masks_for_selection"] = False
    (run_dir / "run_args.json").write_text(
        json.dumps({"command": "select_pixel_policy_with_normal_gate", "args": payload_args}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "selection_evidence.json").write_text(
        json.dumps(selection_evidence, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    safe = {
        key: value.item() if isinstance(value, np.generic) else None if isinstance(value, float) and not np.isfinite(value) else value
        for key, value in selected_metrics.items()
    }
    (run_dir / "metrics.json").write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")


def normal_stats(pred: np.lib.npyio.NpzFile, indices: np.ndarray, normal_label: int) -> dict[str, float]:
    labels = pred["labels"][indices].astype(np.int64)
    normal_idx = indices[labels == normal_label]
    if len(normal_idx) == 0:
        normal_idx = indices
    heatmaps = pred["heatmaps"][normal_idx].astype(np.float32)
    scores = pred["image_scores"][normal_idx].astype(np.float32)
    mean_abs = float(np.mean(np.abs(heatmaps))) + 1e-6
    return {
        "normal_count": float(len(normal_idx)),
        "normal_score_mean": float(np.mean(scores)),
        "normal_score_p95": float(np.quantile(scores, 0.95)),
        "normal_heatmap_mean": float(np.mean(heatmaps)),
        "normal_heatmap_p95": float(np.quantile(heatmaps, 0.95)),
        "normal_heatmap_p99": float(np.quantile(heatmaps, 0.99)),
        "normal_heatmap_cv": float(np.std(heatmaps) / mean_abs),
    }


def rank(values: dict[str, float], reverse: bool = False) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=reverse)
    return {name: idx + 1 for idx, (name, _) in enumerate(ordered)}


def gate_scores(
    stats_by_name: dict[str, dict[str, float]],
    gate_metric: str,
    max_score_inflation: float,
    max_cv_inflation: float,
    synthetic_by_name: dict[str, dict[str, Any]] | None = None,
    utility_kwargs: dict[str, float] | None = None,
) -> dict[str, float]:
    if gate_metric == "highres_if_available":
        priority = {"highres256": 3.0, "pixelraw": 2.0, "fixed": 1.0}
        return {name: priority.get(name, 0.0) for name in stats_by_name}

    if gate_metric == "normal_score_p95":
        return {name: -stats["normal_score_p95"] for name, stats in stats_by_name.items()}
    if gate_metric == "normal_heatmap_p99":
        return {name: -stats["normal_heatmap_p99"] for name, stats in stats_by_name.items()}
    if gate_metric == "normal_heatmap_cv":
        return {name: -stats["normal_heatmap_cv"] for name, stats in stats_by_name.items()}
    if gate_metric == "normal_rank_sum":
        score_rank = rank({name: stats["normal_score_p95"] for name, stats in stats_by_name.items()})
        cv_rank = rank({name: stats["normal_heatmap_cv"] for name, stats in stats_by_name.items()})
        p99_rank = rank({name: stats["normal_heatmap_p99"] for name, stats in stats_by_name.items()})
        return {name: -(score_rank[name] + cv_rank[name] + p99_rank[name]) for name in stats_by_name}

    if gate_metric == "normal_guarded_highres":
        names = set(stats_by_name)
        if "highres256" in names:
            high = stats_by_name["highres256"]
            base = stats_by_name.get("pixelraw")
            if base is None:
                return {name: 1.0 if name == "highres256" else 0.0 for name in stats_by_name}
            score_ok = high["normal_score_p95"] <= base["normal_score_p95"] * max_score_inflation
            cv_ok = high["normal_heatmap_cv"] <= base["normal_heatmap_cv"] * max_cv_inflation
            if score_ok or cv_ok:
                return {name: 1.0 if name == "highres256" else 0.0 for name in stats_by_name}
        if "pixelraw" in names:
            return {name: 1.0 if name == "pixelraw" else 0.0 for name in stats_by_name}
        return {name: 1.0 if name == sorted(names)[0] else 0.0 for name in stats_by_name}

    if gate_metric == "synthetic_normal_utility":
        synthetic_by_name = synthetic_by_name or {}
        utility_kwargs = utility_kwargs or {}
        return {
            name: synthetic_normal_utility(synthetic_by_name[name], **utility_kwargs)
            if name in synthetic_by_name
            else float("-inf")
            for name in stats_by_name
        }

    raise ValueError(f"Unknown gate metric: {gate_metric}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = parse_candidates(args.candidate)
    manifest_rows = (
        read_csv(Path(args.candidate_manifest))
        if args.candidate_manifest
        else []
    )
    if not candidates and not manifest_rows:
        raise ValueError("At least one --candidate or --candidate-manifest row is required")
    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    normal_gate_rows: list[dict[str, Any]] = []
    selected_test_rows: list[dict[str, Any]] = []
    missing_synthetic: list[dict[str, str]] = []
    utility_kwargs = {
        "pixel_ap_weight": args.utility_pixel_ap_weight,
        "aupro_weight": args.utility_aupro_weight,
        "dice_weight": args.utility_dice_weight,
        "pixel_auroc_weight": args.utility_pixel_auroc_weight,
        "image_auroc_weight": args.utility_image_auroc_weight,
        "stability_weight": args.utility_stability_weight,
        "normal_fpr_penalty": args.utility_normal_fpr_penalty,
        "latency_penalty": args.utility_latency_penalty,
        "max_latency_ms": args.utility_max_latency_ms,
    }
    aggregate_synthetic_seeds = [
        int(value)
        for value in split_csv(args.aggregate_synthetic_seeds)
    ]

    for category in split_csv(args.categories):
        category_candidates = list(candidates)
        category_candidates.extend(
            (row["candidate"], row["run"])
            for row in manifest_rows
            if row.get("category") == category
        )
        deduplicated: dict[str, str] = {}
        for name, template in category_candidates:
            deduplicated[name] = template
        category_candidates = list(deduplicated.items())
        loaded: list[tuple[str, Path, np.lib.npyio.NpzFile, dict[str, Any]]] = []
        ref_pred: np.lib.npyio.NpzFile | None = None
        for name, template in category_candidates:
            run_dir = Path(template.format(category=category))
            pred_path = run_dir / "predictions.npz"
            if not pred_path.exists():
                continue
            pred = np.load(pred_path)
            if ref_pred is None:
                ref_pred = pred
            else:
                assert_compatible(ref_pred, pred, pred_path)
            loaded.append((name, run_dir, pred, read_json(run_dir / "metrics.json")))
        if not loaded or ref_pred is None:
            continue

        val_idx, test_idx = stratified_split(ref_pred["labels"], category, args.val_ratio, args.seed)
        if len(test_idx) == 0:
            continue

        stats_by_name: dict[str, dict[str, float]] = {}
        test_by_name: dict[str, dict[str, float]] = {}
        pred_by_name: dict[str, np.lib.npyio.NpzFile] = {}
        run_by_name: dict[str, Path] = {}
        meta_by_name: dict[str, dict[str, Any]] = {}
        synthetic_by_name: dict[str, dict[str, Any]] = {}
        for name, run_dir, pred, meta in loaded:
            val_metrics = metrics(pred, val_idx)
            test_metrics = metrics(pred, test_idx)
            stats = normal_stats(pred, val_idx, args.normal_label)
            stats_by_name[name] = stats
            test_by_name[name] = test_metrics
            pred_by_name[name] = pred
            run_by_name[name] = run_dir
            meta_by_name[name] = meta
            if aggregate_synthetic_seeds:
                synthetic_paths = [
                    run_dir
                    / f"synthetic_validation_seed{seed}_metrics.json"
                    for seed in aggregate_synthetic_seeds
                ]
            else:
                synthetic_paths = [
                    run_dir / args.synthetic_metrics_name
                ]
            available_payloads = [
                read_json(path) for path in synthetic_paths if path.exists()
            ]
            if len(available_payloads) == len(synthetic_paths):
                synthetic_by_name[name] = (
                    mean_numeric_metrics(available_payloads)
                    if aggregate_synthetic_seeds
                    else available_payloads[0]
                )
            elif args.gate_metric == "synthetic_normal_utility":
                missing_synthetic.append(
                    {
                        "dataset": args.dataset,
                        "category": category,
                        "candidate": name,
                        "run": str(run_dir),
                        "expected_artifact": ";".join(
                            str(path)
                            for path in synthetic_paths
                            if not path.exists()
                        ),
                    }
                )
            candidate_rows.append(
                row(
                    {
                        "dataset": args.dataset,
                        "category": category,
                        "candidate": name,
                        "run": str(run_dir),
                        "split": "val",
                        "selection_score": None,
                        "pixel_heatmap_source": meta.get("pixel_heatmap_source", ""),
                    },
                    val_metrics,
                )
            )
            candidate_rows.append(
                row(
                    {
                        "dataset": args.dataset,
                        "category": category,
                        "candidate": name,
                        "run": str(run_dir),
                        "split": "heldout_test",
                        "selection_score": None,
                        "pixel_heatmap_source": meta.get("pixel_heatmap_source", ""),
                    },
                    test_metrics,
                )
            )

        if (
            args.gate_metric == "synthetic_normal_utility"
            and not args.allow_missing_synthetic
            and len(synthetic_by_name) != len(stats_by_name)
        ):
            missing = sorted(set(stats_by_name) - set(synthetic_by_name))
            raise FileNotFoundError(
                f"{category} is missing {args.synthetic_metrics_name} for candidates: {', '.join(missing)}"
            )
        scores = gate_scores(
            stats_by_name,
            args.gate_metric,
            args.max_score_inflation,
            args.max_cv_inflation,
            synthetic_by_name=synthetic_by_name,
            utility_kwargs=utility_kwargs,
        )
        if not any(np.isfinite(value) for value in scores.values()):
            raise ValueError(f"{category} has no candidate with finite selection evidence")
        selected_name = max(scores, key=scores.get)
        selected_run = run_by_name[selected_name]
        selected_meta = meta_by_name[selected_name]
        selected_pred = pred_by_name[selected_name]
        selected_test = test_by_name[selected_name]
        selection_protocol = selection_protocol_name(
            args.gate_metric,
            aggregate_synthetic_seeds,
        )
        for name, stats in stats_by_name.items():
            synthetic = synthetic_by_name.get(name, {})
            normal_gate_rows.append(
                {
                    "dataset": args.dataset,
                    "category": category,
                    "candidate": name,
                    "run": str(run_by_name[name]),
                    "gate_metric": args.gate_metric,
                    "gate_score": scores[name],
                    "selected": name == selected_name,
                    **stats,
                    "synthetic_image_auroc": synthetic.get("image_auroc"),
                    "synthetic_pixel_auroc": synthetic.get("pixel_auroc"),
                    "synthetic_aupro": synthetic.get("aupro"),
                    "synthetic_pixel_ap": synthetic.get("pixel_ap"),
                    "synthetic_dice": synthetic.get("dice"),
                    "synthetic_normal_pixel_fpr": synthetic.get("normal_pixel_fpr"),
                    "synthetic_augmentation_stability": synthetic.get("augmentation_stability"),
                    "candidate_latency_ms": synthetic.get("latency_ms"),
                    "selection_protocol": selection_protocol,
                    "uses_real_anomaly_labels_for_selection": False,
                    "uses_real_anomaly_masks_for_selection": False,
                }
            )
        selection_rows.append(
            {
                "dataset": args.dataset,
                "category": category,
                "selected_candidate": selected_name,
                "selected_run": str(selected_run),
                "selection_score": scores[selected_name],
                "val_count": int(len(val_idx)),
                "test_count": int(len(test_idx)),
                "pixel_heatmap_source": selected_meta.get("pixel_heatmap_source", ""),
                "selection_protocol": selection_protocol,
                "uses_real_anomaly_labels_for_selection": False,
                "uses_real_anomaly_masks_for_selection": False,
                "selection_evidence_path": str(selected_run / args.synthetic_metrics_name)
                if (
                    args.gate_metric == "synthetic_normal_utility"
                    and not aggregate_synthetic_seeds
                )
                else ";".join(
                    str(
                        selected_run
                        / f"synthetic_validation_seed{seed}_metrics.json"
                    )
                    for seed in aggregate_synthetic_seeds
                ),
                "selection_evidence_seeds": " ".join(
                    str(seed) for seed in aggregate_synthetic_seeds
                ),
                "val_indices": " ".join(str(int(i)) for i in val_idx),
                "test_indices": " ".join(str(int(i)) for i in test_idx),
            }
        )
        selected_test_rows.append(
            row(
                {
                    "dataset": args.dataset,
                    "category": category,
                    "candidate": selected_name,
                    "run": str(selected_run),
                    "split": "heldout_test",
                    "selection_score": scores[selected_name],
                    "pixel_heatmap_source": selected_meta.get("pixel_heatmap_source", ""),
                },
                selected_test,
            )
        )
        if args.materialize:
            materialize_selected(
                out_dir,
                category,
                selected_name,
                selected_run,
                selected_pred,
                test_idx,
                {
                    "dataset": args.dataset,
                    "category": category,
                    "candidate": selected_name,
                    "selection_score": scores[selected_name],
                    "selection_protocol": selection_protocol,
                    "gate_metric": args.gate_metric,
                    "uses_real_anomaly_labels_for_selection": False,
                    "uses_real_anomaly_masks_for_selection": False,
                    "normal_statistics": stats_by_name[selected_name],
                    "synthetic_metrics": synthetic_by_name.get(selected_name, {}),
                    "utility_weights": utility_kwargs,
                    "aggregate_synthetic_seeds": aggregate_synthetic_seeds,
                },
            )

    metric_fields = [
        "dataset",
        "category",
        "candidate",
        "run",
        "split",
        "selection_score",
        "pixel_heatmap_source",
        *METRICS,
        "f1",
        "iou",
        "threshold",
    ]
    write_csv(out_dir / "candidate_split_metrics.csv", candidate_rows, metric_fields)
    write_csv(out_dir / "selected_heldout_metrics.csv", selected_test_rows, metric_fields)
    write_csv(
        out_dir / "selection.csv",
        selection_rows,
        [
            "dataset",
            "category",
            "selected_candidate",
            "selected_run",
            "selection_score",
            "val_count",
            "test_count",
            "pixel_heatmap_source",
            "selection_protocol",
            "uses_real_anomaly_labels_for_selection",
            "uses_real_anomaly_masks_for_selection",
            "selection_evidence_path",
            "selection_evidence_seeds",
            "val_indices",
            "test_indices",
        ],
    )
    write_csv(
        out_dir / "normal_gate_metrics.csv",
        normal_gate_rows,
        [
            "dataset",
            "category",
            "candidate",
            "run",
            "gate_metric",
            "gate_score",
            "selected",
            "normal_count",
            "normal_score_mean",
            "normal_score_p95",
            "normal_heatmap_mean",
            "normal_heatmap_p95",
            "normal_heatmap_p99",
            "normal_heatmap_cv",
            "synthetic_image_auroc",
            "synthetic_pixel_auroc",
            "synthetic_aupro",
            "synthetic_pixel_ap",
            "synthetic_dice",
            "synthetic_normal_pixel_fpr",
            "synthetic_augmentation_stability",
            "candidate_latency_ms",
            "selection_protocol",
            "uses_real_anomaly_labels_for_selection",
            "uses_real_anomaly_masks_for_selection",
        ],
    )
    write_csv(
        out_dir / "missing_synthetic_artifacts.csv",
        missing_synthetic,
        ["dataset", "category", "candidate", "run", "expected_artifact"],
    )
    means: dict[str, Any] = {"dataset": args.dataset, "categories": len(selected_test_rows)}
    for metric in METRICS:
        vals = [float(item[metric]) for item in selected_test_rows if item.get(metric) not in {None, ""} and np.isfinite(float(item[metric]))]
        means[metric] = float(np.mean(vals)) if vals else None
    summary = {
        "dataset": args.dataset,
        "gate_metric": args.gate_metric,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "selection_protocol": selection_protocol_name(
            args.gate_metric,
            aggregate_synthetic_seeds,
        ),
        "uses_real_anomaly_labels_for_selection": False,
        "uses_real_anomaly_masks_for_selection": False,
        "utility_weights": utility_kwargs if args.gate_metric == "synthetic_normal_utility" else None,
        "aggregate_synthetic_seeds": aggregate_synthetic_seeds,
        "missing_synthetic_artifacts": len(missing_synthetic),
        "means": means,
        "selected_counts": {
            name: sum(1 for item in selection_rows if item["selected_candidate"] == name)
            for name in sorted(
                {item["candidate"] for item in normal_gate_rows}
            )
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
