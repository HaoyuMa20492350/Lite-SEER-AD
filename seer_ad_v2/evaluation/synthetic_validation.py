from __future__ import annotations

from typing import Any

import numpy as np

from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.pixel_threshold_policy import (
    select_synthetic_normal_threshold,
)


SYNTHETIC_METRIC_KEYS = (
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "pixel_ap",
    "dice",
)


def normalized_map_similarity(reference: np.ndarray, transformed: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float32)
    cur = np.asarray(transformed, dtype=np.float32)
    if ref.shape != cur.shape:
        raise ValueError(f"Map shapes do not match: {ref.shape} vs {cur.shape}")

    def normalize(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(values.shape[0], -1)
        lo = np.min(flat, axis=1).reshape(-1, 1, 1)
        hi = np.max(flat, axis=1).reshape(-1, 1, 1)
        return (values - lo) / (hi - lo + 1e-8)

    delta = np.mean(np.abs(normalize(ref) - normalize(cur)), axis=(1, 2))
    return float(np.mean(np.clip(1.0 - delta, 0.0, 1.0)))


def evaluate_synthetic_validation(
    clean_heatmaps: np.ndarray,
    synthetic_heatmaps: np.ndarray,
    synthetic_masks: np.ndarray,
    clean_image_scores: np.ndarray,
    synthetic_image_scores: np.ndarray,
    augmented_synthetic_heatmaps: list[np.ndarray],
    *,
    max_normal_pixel_fpr: float = 0.005,
) -> dict[str, Any]:
    clean_heatmaps = np.asarray(clean_heatmaps, dtype=np.float32)
    synthetic_heatmaps = np.asarray(synthetic_heatmaps, dtype=np.float32)
    synthetic_masks = (np.asarray(synthetic_masks) > 0).astype(np.uint8)
    count = len(clean_heatmaps)
    if count == 0 or len(synthetic_heatmaps) != count or len(synthetic_masks) != count:
        raise ValueError("Synthetic validation arrays must be non-empty and aligned")

    labels = np.concatenate(
        [np.zeros(count, dtype=np.uint8), np.ones(count, dtype=np.uint8)],
        axis=0,
    )
    masks = np.concatenate([np.zeros_like(synthetic_masks), synthetic_masks], axis=0)
    heatmaps = np.concatenate([clean_heatmaps, synthetic_heatmaps], axis=0)
    image_scores = np.concatenate(
        [
            np.asarray(clean_image_scores, dtype=np.float32),
            np.asarray(synthetic_image_scores, dtype=np.float32),
        ],
        axis=0,
    )
    threshold_policy = select_synthetic_normal_threshold(
        clean_heatmaps,
        synthetic_heatmaps,
        synthetic_masks,
        max_normal_fpr=max_normal_pixel_fpr,
    )
    metrics = detection_metrics(
        labels,
        image_scores,
        masks,
        heatmaps,
        pixel_threshold=float(threshold_policy["threshold"]),
        threshold_protocol=str(threshold_policy["protocol"]),
    )
    stability_values = [
        normalized_map_similarity(synthetic_heatmaps, maps)
        for maps in augmented_synthetic_heatmaps
    ]
    metrics["normal_pixel_fpr"] = threshold_policy["observed_normal_pixel_fpr"]
    metrics["max_normal_pixel_fpr"] = threshold_policy["max_normal_pixel_fpr"]
    metrics["threshold_fallback_used"] = threshold_policy["fallback_used"]
    metrics["uses_real_anomaly_labels_for_threshold"] = False
    metrics["uses_real_anomaly_masks_for_threshold"] = False
    metrics["augmentation_stability"] = float(np.mean(stability_values)) if stability_values else 1.0
    metrics["synthetic_samples"] = int(count)
    return metrics


def synthetic_normal_utility(
    metrics: dict[str, Any],
    *,
    pixel_ap_weight: float = 0.30,
    aupro_weight: float = 0.25,
    dice_weight: float = 0.15,
    pixel_auroc_weight: float = 0.10,
    image_auroc_weight: float = 0.05,
    stability_weight: float = 0.15,
    normal_fpr_penalty: float = 0.20,
    latency_penalty: float = 0.02,
    max_latency_ms: float = 100.0,
) -> float:
    def value(name: str, default: float = 0.0) -> float:
        raw = metrics.get(name, default)
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            return default
        return parsed if np.isfinite(parsed) else default

    latency = max(0.0, value("latency_ms"))
    latency_scale = min(1.0, latency / max(1e-6, float(max_latency_ms)))
    return float(
        pixel_ap_weight * value("pixel_ap")
        + aupro_weight * value("aupro")
        + dice_weight * value("dice")
        + pixel_auroc_weight * value("pixel_auroc")
        + image_auroc_weight * value("image_auroc")
        + stability_weight * value("augmentation_stability")
        - normal_fpr_penalty * value("normal_pixel_fpr", 1.0)
        - latency_penalty * latency_scale
    )
