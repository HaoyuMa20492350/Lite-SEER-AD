from __future__ import annotations

import numpy as np

from seer_ad_v2.data.hard_negative_mining import ROI
from seer_ad_v2.utils.image import normalize01


def roi_score(heatmap: np.ndarray, roi: ROI) -> float:
    patch = heatmap[roi.y1 : roi.y2, roi.x1 : roi.x2]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch) + np.max(patch))


def score_drop(before: np.ndarray, after: np.ndarray, roi: ROI) -> float:
    return max(0.0, roi_score(before, roi) - roi_score(after, roi))


def apply_crv_to_heatmap(base: np.ndarray, rois: list[ROI], drops: list[float], weight: float = 0.35) -> np.ndarray:
    out = base.astype(np.float32).copy()
    if not rois:
        return normalize01(out)
    positive = [float(drop) for drop in drops if float(drop) > 0.0]
    if not positive:
        return normalize01(out)
    max_drop = max(positive) + 1e-8
    for roi, drop in zip(rois, drops):
        if float(drop) <= 0.0:
            continue
        out[roi.y1 : roi.y2, roi.x1 : roi.x2] += weight * float(drop / max_drop)
    return normalize01(out)


def apply_verifier_to_heatmap(base: np.ndarray, rois: list[ROI], scores: list[float], weight: float = 0.5) -> np.ndarray:
    out = base.astype(np.float32).copy()
    if not rois or weight <= 0.0:
        return normalize01(out)
    for roi, score in zip(rois, scores):
        evidence = float(np.clip(score, 0.0, 1.0)) - 0.5
        out[roi.y1 : roi.y2, roi.x1 : roi.x2] += float(weight) * evidence
    return normalize01(out)
