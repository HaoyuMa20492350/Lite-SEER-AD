from __future__ import annotations

import math

import numpy as np


IMAGE_SCORE_MODES = [
    "max_mean",
    "max",
    "mean",
    "p99",
    "p99_mean",
    "p95",
    "p95_mean",
    "top1",
    "top1_mean",
    "top5",
    "top5_mean",
]


def _top_fraction_mean(values: np.ndarray, fraction: float) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    k = max(1, int(math.ceil(flat.size * fraction)))
    k = min(k, flat.size)
    top = np.partition(flat, flat.size - k)[-k:]
    return float(np.mean(top))


def image_score_from_heatmap(heatmap: np.ndarray, mode: str = "max_mean") -> float:
    h = np.asarray(heatmap, dtype=np.float32)
    if h.size == 0:
        return 0.0
    h_max = float(np.nanmax(h))
    h_mean = float(np.nanmean(h))
    if mode == "max_mean":
        return h_max + h_mean
    if mode == "max":
        return h_max
    if mode == "mean":
        return h_mean
    if mode == "p99":
        return float(np.nanpercentile(h, 99.0))
    if mode == "p99_mean":
        return float(np.nanpercentile(h, 99.0)) + h_mean
    if mode == "p95":
        return float(np.nanpercentile(h, 95.0))
    if mode == "p95_mean":
        return float(np.nanpercentile(h, 95.0)) + h_mean
    if mode == "top1":
        return _top_fraction_mean(h, 0.01)
    if mode == "top1_mean":
        return _top_fraction_mean(h, 0.01) + h_mean
    if mode == "top5":
        return _top_fraction_mean(h, 0.05)
    if mode == "top5_mean":
        return _top_fraction_mean(h, 0.05) + h_mean
    raise ValueError(f"Unknown image score mode: {mode}")


def image_scores_from_heatmaps(heatmaps: np.ndarray, mode: str = "max_mean") -> np.ndarray:
    return np.asarray([image_score_from_heatmap(heatmap, mode=mode) for heatmap in heatmaps], dtype=np.float32)
