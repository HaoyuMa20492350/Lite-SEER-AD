from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


SDR_KEYS = ("sdr", "pixel_sdr", "feature_sdr", "prototype_sdr")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def safe_pearson(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2:
        return float("nan")
    return safe_pearson(_average_ranks(x), _average_ranks(y))


def roi_ground_truth_records(
    masks: np.ndarray,
    labels: np.ndarray,
    roi_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    masks = np.asarray(masks)
    labels = np.asarray(labels).reshape(-1)
    records = []
    for row in roi_rows:
        image_index = int(row.get("image_index", -1))
        if image_index < 0 or image_index >= len(masks):
            continue
        x1, y1, x2, y2 = [
            int(value) for value in row.get("bbox", [0, 0, 0, 0])
        ]
        height, width = masks[image_index].shape[-2:]
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        patch = np.asarray(masks[image_index, y1:y2, x1:x2]) > 0
        record = {
            "image_index": image_index,
            "label": int(labels[image_index]),
            "gt_fraction": float(np.mean(patch)),
            "gt_hit": int(np.any(patch)),
        }
        for key in SDR_KEYS:
            record[key] = float(row.get(key, 0.0))
        records.append(record)
    return records


def sdr_gt_summary(records: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    rows = list(records)
    anomaly_rows = [row for row in rows if int(row["label"]) == 1]
    summary: dict[str, float | int] = {
        "roi_count": len(rows),
        "anomaly_roi_count": len(anomaly_rows),
        "gt_hit_rate": (
            float(np.mean([row["gt_hit"] for row in anomaly_rows]))
            if anomaly_rows
            else float("nan")
        ),
        "positive_sdr_gt_hit_rate": (
            float(
                np.mean(
                    [
                        row["gt_hit"]
                        for row in anomaly_rows
                        if float(row["sdr"]) > 0
                    ]
                )
            )
            if any(float(row["sdr"]) > 0 for row in anomaly_rows)
            else float("nan")
        ),
    }
    for key in SDR_KEYS:
        values = [float(row[key]) for row in anomaly_rows]
        fractions = [float(row["gt_fraction"]) for row in anomaly_rows]
        hits = [float(row["gt_hit"]) for row in anomaly_rows]
        summary[f"{key}_gt_fraction_pearson"] = safe_pearson(values, fractions)
        summary[f"{key}_gt_fraction_spearman"] = safe_spearman(values, fractions)
        summary[f"{key}_gt_hit_pearson"] = safe_pearson(values, hits)
    return summary


def image_repair_quality(
    original: np.ndarray,
    repaired: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    original = np.asarray(original, dtype=np.float32)
    repaired = np.asarray(repaired, dtype=np.float32)
    if original.max(initial=0.0) > 1.5:
        original = original / 255.0
    if repaired.max(initial=0.0) > 1.5:
        repaired = repaired / 255.0
    if original.shape != repaired.shape:
        raise ValueError(
            f"Original and repaired images differ: {original.shape} != {repaired.shape}"
        )
    binary = (np.asarray(mask).squeeze() > 0).astype(np.uint8)
    if binary.shape != original.shape[:2]:
        binary = cv2.resize(
            binary,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    absolute = np.mean(np.abs(original - repaired), axis=-1)
    background = binary == 0
    foreground = binary > 0
    psnr = peak_signal_noise_ratio(original, repaired, data_range=1.0)
    ssim = structural_similarity(
        original,
        repaired,
        channel_axis=-1,
        data_range=1.0,
    )
    background_mse = (
        float(np.mean((original[background] - repaired[background]) ** 2))
        if np.any(background)
        else float("nan")
    )
    background_psnr = (
        float("inf")
        if background_mse == 0
        else (
            float(10.0 * np.log10(1.0 / background_mse))
            if np.isfinite(background_mse)
            else float("nan")
        )
    )
    if np.any(foreground):
        kernel = np.ones((3, 3), dtype=np.uint8)
        boundary = (
            cv2.dilate(binary, kernel, iterations=1)
            - cv2.erode(binary, kernel, iterations=1)
        ) > 0
        boundary_consistency = (
            float(1.0 - np.mean(absolute[boundary]))
            if np.any(boundary)
            else float("nan")
        )
        foreground_mae = float(np.mean(absolute[foreground]))
    else:
        boundary_consistency = float("nan")
        foreground_mae = float("nan")
    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "background_psnr": background_psnr,
        "background_mae": (
            float(np.mean(absolute[background]))
            if np.any(background)
            else float("nan")
        ),
        "foreground_mae": foreground_mae,
        "boundary_consistency": boundary_consistency,
        "identity": float(np.max(absolute) <= (0.5 / 255.0)),
    }
