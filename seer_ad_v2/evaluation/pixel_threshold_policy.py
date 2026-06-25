from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


POLICY_PROTOCOL = "synthetic_normal_fixed_threshold_v1"


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = (np.asarray(y_true).reshape(-1) > 0).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Pixel labels and scores must align: {y_true.shape} vs {y_score.shape}"
        )
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("Pixel threshold must be finite")

    pred = y_score >= threshold
    positive = y_true == 1
    tp = float(np.count_nonzero(pred & positive))
    fp = float(np.count_nonzero(pred & ~positive))
    fn = float(np.count_nonzero(~pred & positive))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    return {
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
        "threshold": threshold,
    }


def _finite_flat(values: np.ndarray, name: str) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError(f"{name} contains no finite values")
    return flat


def _candidate_thresholds(
    normal_scores: np.ndarray,
    synthetic_scores: np.ndarray,
    max_candidates: int,
) -> np.ndarray:
    values = np.concatenate([normal_scores, synthetic_scores])
    unique = np.unique(values)
    if unique.size <= max_candidates:
        return unique
    quantiles = np.linspace(0.0, 1.0, max_candidates, dtype=np.float64)
    return np.unique(np.quantile(values, quantiles))


def _normal_quantile_threshold(
    normal_scores: np.ndarray,
    max_normal_fpr: float,
) -> tuple[float, float]:
    quantile = float(np.quantile(normal_scores, 1.0 - max_normal_fpr))
    threshold = quantile
    observed = float(np.mean(normal_scores >= threshold))
    if observed > max_normal_fpr:
        threshold = float(np.nextafter(threshold, np.inf))
        observed = float(np.mean(normal_scores >= threshold))
    return threshold, quantile


def _metrics_from_counts(
    tp: float,
    fp: float,
    fn: float,
    threshold: float,
) -> dict[str, float]:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    return {
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
        "threshold": float(threshold),
    }


def select_synthetic_normal_threshold(
    clean_heatmaps: np.ndarray,
    synthetic_heatmaps: np.ndarray,
    synthetic_masks: np.ndarray,
    *,
    max_normal_fpr: float = 0.005,
    max_candidates: int = 2048,
) -> dict[str, Any]:
    if not 0.0 < float(max_normal_fpr) < 1.0:
        raise ValueError("max_normal_fpr must be between 0 and 1")
    if max_candidates < 2:
        raise ValueError("max_candidates must be at least 2")

    clean = np.asarray(clean_heatmaps)
    synthetic = np.asarray(synthetic_heatmaps)
    masks = np.asarray(synthetic_masks)
    if clean.shape != synthetic.shape or synthetic.shape != masks.shape:
        raise ValueError(
            "Clean heatmaps, synthetic heatmaps, and synthetic masks must align: "
            f"{clean.shape}, {synthetic.shape}, {masks.shape}"
        )
    if clean.size == 0:
        raise ValueError("Synthetic-normal threshold evidence must be non-empty")

    normal_scores = _finite_flat(clean, "clean_heatmaps")
    synthetic_scores = _finite_flat(synthetic, "synthetic_heatmaps")
    synthetic_true = (masks.reshape(-1) > 0).astype(np.uint8)
    finite_synthetic = np.isfinite(synthetic.reshape(-1))
    synthetic_true = synthetic_true[finite_synthetic]
    if synthetic_true.size != synthetic_scores.size:
        raise ValueError("Synthetic masks and finite synthetic scores do not align")
    if synthetic_true.sum() == 0:
        raise ValueError("Synthetic masks contain no anomalous pixels")

    candidates = _candidate_thresholds(
        normal_scores,
        synthetic_scores,
        max_candidates,
    )
    normal_sorted = np.sort(normal_scores)
    synthetic_order = np.argsort(synthetic_scores)
    synthetic_sorted = synthetic_scores[synthetic_order]
    synthetic_true_sorted = synthetic_true[synthetic_order]
    positive_prefix = np.concatenate(
        [[0], np.cumsum(synthetic_true_sorted, dtype=np.int64)]
    )
    total_positive = int(positive_prefix[-1])
    feasible: list[tuple[float, float, float, dict[str, float]]] = []
    for threshold in candidates:
        normal_index = int(np.searchsorted(normal_sorted, threshold, side="left"))
        normal_fpr = float(
            (normal_sorted.size - normal_index) / normal_sorted.size
        )
        if normal_fpr > max_normal_fpr:
            continue
        synthetic_index = int(
            np.searchsorted(synthetic_sorted, threshold, side="left")
        )
        predicted_positive = synthetic_sorted.size - synthetic_index
        tp = total_positive - int(positive_prefix[synthetic_index])
        fp = predicted_positive - tp
        fn = total_positive - tp
        metrics = _metrics_from_counts(
            float(tp),
            float(fp),
            float(fn),
            float(threshold),
        )
        feasible.append(
            (
                metrics["dice"],
                -normal_fpr,
                float(threshold),
                metrics,
            )
        )

    fallback_used = not feasible
    fallback_quantile = None
    if feasible:
        _, neg_normal_fpr, threshold, selected_metrics = max(
            feasible,
            key=lambda item: (item[0], item[1], item[2]),
        )
        observed_normal_fpr = -neg_normal_fpr
    else:
        threshold, fallback_quantile = _normal_quantile_threshold(
            normal_scores,
            max_normal_fpr,
        )
        observed_normal_fpr = float(np.mean(normal_scores >= threshold))
        selected_metrics = binary_metrics_at_threshold(
            synthetic_true,
            synthetic_scores,
            threshold,
        )

    return {
        "protocol": POLICY_PROTOCOL,
        "threshold": float(threshold),
        "max_normal_pixel_fpr": float(max_normal_fpr),
        "observed_normal_pixel_fpr": float(observed_normal_fpr),
        "synthetic_f1": selected_metrics["f1"],
        "synthetic_iou": selected_metrics["iou"],
        "synthetic_dice": selected_metrics["dice"],
        "normal_pixel_count": int(normal_scores.size),
        "synthetic_pixel_count": int(synthetic_scores.size),
        "synthetic_positive_pixel_count": int(synthetic_true.sum()),
        "candidate_threshold_count": int(candidates.size),
        "feasible_threshold_count": int(len(feasible)),
        "fallback_used": fallback_used,
        "fallback_normal_quantile": fallback_quantile,
        "uses_real_anomaly_labels": False,
        "uses_real_anomaly_masks": False,
    }


def save_pixel_threshold_policy(policy: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2, allow_nan=False), encoding="utf-8")
    return out


def load_pixel_threshold_policy(path: str | Path) -> dict[str, Any]:
    policy_path = Path(path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    threshold = float(policy["threshold"])
    if not np.isfinite(threshold):
        raise ValueError(f"{policy_path} contains a non-finite threshold")
    policy["threshold"] = threshold
    return policy
