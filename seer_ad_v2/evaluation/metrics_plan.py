from __future__ import annotations

from typing import Any

import numpy as np


def _roi_is_positive(row: dict[str, Any]) -> bool:
    if "hn_sev_positive" in row:
        return int(row.get("hn_sev_positive", 0)) == 1
    threshold = float(row.get("hn_sev_threshold", 0.5))
    return float(row.get("hn_sev_confidence", 0.0)) >= threshold


def false_positive_region_rate(roi_rows: list[dict[str, Any]], labels: np.ndarray | None = None) -> float:
    if not roi_rows:
        return 0.0
    rows = roi_rows
    if labels is not None and len(labels):
        normal_ids = {int(i) for i, y in enumerate(labels.reshape(-1).tolist()) if int(y) == 0}
        rows = [r for r in roi_rows if int(r.get("image_index", -1)) in normal_ids]
    if not rows:
        return 0.0
    positives = [r for r in rows if _roi_is_positive(r)]
    return float(len(positives) / max(1, len(rows)))


def sdr_summary(drops: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(drops, dtype=np.float32)
    if arr.size == 0:
        return {"sdr_mean": 0.0, "sdr_median": 0.0, "sdr_positive_rate": 0.0}
    return {
        "sdr_mean": float(arr.mean()),
        "sdr_median": float(np.median(arr)),
        "sdr_positive_rate": float((arr > 0).mean()),
    }


def repair_detection_consistency(masks: np.ndarray, roi_rows: list[dict[str, Any]]) -> float:
    if masks.size == 0 or not roi_rows:
        return 0.0
    hits = 0
    considered = 0
    for row in roi_rows:
        if float(row.get("sdr", 0.0)) <= 0:
            continue
        image_index = int(row.get("image_index", -1))
        if image_index < 0 or image_index >= len(masks):
            continue
        x1, y1, x2, y2 = [int(v) for v in row.get("bbox", [0, 0, 0, 0])]
        patch = masks[image_index, y1:y2, x1:x2]
        considered += 1
        hits += int(patch.sum() > 0)
    return float(hits / max(1, considered))


def pareto_area(rows: list[dict[str, Any]], score_key: str = "image_score") -> float:
    if not rows:
        return 0.0
    points = []
    for row in rows:
        latency = float(row.get("latency_ms", 0.0))
        score = float(row.get(score_key, 0.0))
        points.append((latency, score))
    points.sort(key=lambda x: x[0])
    xs = np.asarray([p[0] for p in points], dtype=np.float32)
    ys = np.asarray([p[1] for p in points], dtype=np.float32)
    if len(xs) < 2:
        return 0.0
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-8)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-8)
    return float(np.trapz(ys, xs))


def plan_metric_summary(
    masks: np.ndarray,
    heatmaps: np.ndarray,
    roi_rows: list[dict[str, Any]],
    score_drops: list[float] | np.ndarray,
    pareto_rows: list[dict[str, Any]],
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    summary = {
        "fprr": false_positive_region_rate(roi_rows, labels),
        "rdc": repair_detection_consistency(masks, roi_rows),
        "pareto_area": pareto_area(pareto_rows),
    }
    summary.update(sdr_summary(score_drops))
    return summary


def efficiency_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "latency_ms_mean": 0.0,
            "fps": 0.0,
            "nfe_mean": 0.0,
            "repaired_area_ratio_mean": 0.0,
            "local_region_ratio_mean": 0.0,
            "gpu_memory_mb": 0.0,
        }
    latency = np.asarray([float(r.get("latency_ms", 0.0)) for r in rows], dtype=np.float32)
    nfe = np.asarray([float(r.get("nfe", 0.0)) for r in rows], dtype=np.float32)
    repaired = np.asarray([float(r.get("repaired_area_ratio", 0.0)) for r in rows], dtype=np.float32)
    local = np.asarray([float(r.get("local_region_ratio", 0.0)) for r in rows], dtype=np.float32)
    gpu_mem = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            gpu_mem = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    except Exception:
        gpu_mem = 0.0
    latency_mean = float(latency.mean()) if latency.size else 0.0
    return {
        "latency_ms_mean": latency_mean,
        "fps": float(1000.0 / latency_mean) if latency_mean > 0 else 0.0,
        "nfe_mean": float(nfe.mean()) if nfe.size else 0.0,
        "repaired_area_ratio_mean": float(repaired.mean()) if repaired.size else 0.0,
        "local_region_ratio_mean": float(local.mean()) if local.size else 0.0,
        "gpu_memory_mb": gpu_mem,
    }
