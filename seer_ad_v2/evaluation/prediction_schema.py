from __future__ import annotations

from typing import Mapping

import numpy as np


CANONICAL_HEATMAP_KEYS = (
    "detection_heatmaps",
    "verification_heatmaps",
    "image_score_heatmaps",
)


def resolve_prediction_heatmaps(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detection = arrays.get("detection_heatmaps")
    if detection is None:
        detection = arrays.get("heatmaps")
    if detection is None:
        raise KeyError("Prediction artifact has no detection heatmap.")

    verification = arrays.get("verification_heatmaps")
    if verification is None:
        verification = arrays.get("final_heatmaps")
    if verification is None:
        verification = detection

    image_score = arrays.get("image_score_heatmaps")
    if image_score is None:
        image_score = arrays.get("score_heatmaps")
    if image_score is None:
        image_score = detection
    return detection, verification, image_score


def prediction_heatmap_payload(
    detection: np.ndarray,
    verification: np.ndarray | None = None,
    image_score: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    detection = np.asarray(detection, dtype=np.float32)
    verification = (
        detection
        if verification is None
        else np.asarray(verification, dtype=np.float32)
    )
    image_score = (
        detection
        if image_score is None
        else np.asarray(image_score, dtype=np.float32)
    )
    if detection.shape != verification.shape or detection.shape != image_score.shape:
        raise ValueError(
            "Detection, verification, and image-score heatmaps must have "
            f"matching shapes, got {detection.shape}, {verification.shape}, "
            f"{image_score.shape}."
        )
    return {
        "heatmaps": detection,
        "final_heatmaps": verification,
        "score_heatmaps": image_score,
        "detection_heatmaps": detection,
        "verification_heatmaps": verification,
        "image_score_heatmaps": image_score,
    }
