from __future__ import annotations

import cv2
import numpy as np


def resize_heatmaps(heatmaps: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(heatmaps, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected N x H x W heatmaps, got {values.shape}")
    if values.shape[1:] == shape:
        return values
    height, width = shape
    return np.stack(
        [
            cv2.resize(item, (width, height), interpolation=cv2.INTER_LINEAR)
            for item in values
        ]
    ).astype(np.float32)


def normal_scale(
    clean_heatmaps: np.ndarray,
    *,
    center_quantile: float = 0.5,
    upper_quantile: float = 0.995,
) -> tuple[float, float]:
    values = np.asarray(clean_heatmaps, dtype=np.float32)
    if values.size == 0:
        raise ValueError("Normal calibration heatmaps are empty")
    if not 0.0 <= center_quantile < upper_quantile <= 1.0:
        raise ValueError("Expected 0 <= center_quantile < upper_quantile <= 1")
    center = float(np.quantile(values, center_quantile))
    upper = float(np.quantile(values, upper_quantile))
    return center, max(upper - center, 1e-8)


def apply_normal_scale(
    heatmaps: np.ndarray,
    center: float,
    scale: float,
) -> np.ndarray:
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Normal calibration center and scale must be finite")
    return ((np.asarray(heatmaps, dtype=np.float32) - center) / scale).astype(
        np.float32
    )


def fuse_heatmaps(
    source_a: np.ndarray,
    source_b: np.ndarray,
    *,
    weight_a: float,
    scale_a: tuple[float, float],
    scale_b: tuple[float, float],
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight_a must be between 0 and 1")
    shape = target_shape or tuple(np.asarray(source_b).shape[1:])
    calibrated_a = apply_normal_scale(
        resize_heatmaps(source_a, shape), *scale_a
    )
    calibrated_b = apply_normal_scale(
        resize_heatmaps(source_b, shape), *scale_b
    )
    return (
        weight_a * calibrated_a + (1.0 - weight_a) * calibrated_b
    ).astype(np.float32)
