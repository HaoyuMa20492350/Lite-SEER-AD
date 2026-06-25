from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from seer_ad_v2.evaluation.pixel_threshold_policy import (
    binary_metrics_at_threshold,
)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.uint8)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.uint8)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def best_f1_iou_dice(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.uint8).reshape(-1)
    y_score = y_score.reshape(-1)
    if len(np.unique(y_true)) < 2:
        return {"f1": float("nan"), "iou": float("nan"), "dice": float("nan"), "threshold": float("nan")}
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = int(np.nanargmax(f1))
    thr = thresholds[min(idx, len(thresholds) - 1)] if len(thresholds) else 0.5
    pred = (y_score >= thr).astype(np.uint8)
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return {"f1": float(f1[idx]), "iou": float(iou), "dice": float(dice), "threshold": float(thr)}


def aupro_score(masks: np.ndarray, heatmaps: np.ndarray, max_fpr: float = 0.3, num_thresholds: int = 200) -> float:
    """Compute PRO AUC up to a maximum false positive rate.

    This follows the common MVTec-style approximation: threshold the anomaly
    map, average per-component overlap on anomalous regions, and integrate PRO
    over image-background FPR in [0, max_fpr].
    """
    masks = (masks > 0).astype(np.uint8)
    heatmaps = heatmaps.astype(np.float32)
    if masks.size == 0 or heatmaps.size == 0 or masks.sum() == 0:
        return float("nan")
    lo = float(np.nanmin(heatmaps))
    hi = float(np.nanmax(heatmaps))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0

    finite_scores = heatmaps[np.isfinite(heatmaps)]
    quantiles = np.linspace(0.0, 1.0, max(2, int(num_thresholds)))
    thresholds = np.unique(np.quantile(finite_scores, quantiles))[::-1]
    thresholds = np.concatenate(
        [[np.nextafter(hi, np.inf)], thresholds.astype(np.float64)]
    )
    bg = masks == 0
    bg_total = int(bg.sum())
    if bg_total == 0:
        return float("nan")

    try:
        from scipy import ndimage
    except Exception:
        ndimage = None

    components: list[tuple[int, np.ndarray, int]] = []
    for idx, mask in enumerate(masks):
        if ndimage is None:
            labels = mask.astype(np.int32)
            num = 1 if mask.sum() > 0 else 0
        else:
            labels, num = ndimage.label(mask)
        for comp_id in range(1, int(num) + 1):
            comp = labels == comp_id
            area = int(comp.sum())
            if area > 0:
                components.append((idx, comp, area))
    if not components:
        return float("nan")

    points: list[tuple[float, float]] = []
    for thr in thresholds:
        pred = heatmaps >= float(thr)
        fpr = float((pred & bg).sum() / max(1, bg_total))
        overlaps = [float(pred[idx][comp].sum() / area) for idx, comp, area in components]
        points.append((fpr, float(np.mean(overlaps))))
    if not points:
        return 0.0

    best_pro_by_fpr: dict[float, float] = {}
    for fpr, pro in points:
        best_pro_by_fpr[fpr] = max(pro, best_pro_by_fpr.get(fpr, 0.0))
    ordered = sorted(best_pro_by_fpr.items())
    xs_all = np.asarray([item[0] for item in ordered], dtype=np.float64)
    ys_all = np.asarray([item[1] for item in ordered], dtype=np.float64)
    if xs_all[0] > 0.0:
        xs_all = np.concatenate([[0.0], xs_all])
        ys_all = np.concatenate([[0.0], ys_all])

    keep = xs_all < max_fpr
    xs = xs_all[keep]
    ys = ys_all[keep]
    boundary_y = float(np.interp(max_fpr, xs_all, ys_all))
    xs = np.concatenate([xs, [max_fpr]])
    ys = np.concatenate([ys, [boundary_y]])
    return float(np.trapz(ys, xs) / max_fpr)


def detection_metrics(
    labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    heatmaps: np.ndarray,
    *,
    pixel_threshold: float | None = None,
    threshold_protocol: str | None = None,
) -> dict[str, float | str]:
    labels = labels.astype(np.uint8).reshape(-1)
    image_scores = image_scores.reshape(-1)
    pixel_true = masks.reshape(-1).astype(np.uint8)
    pixel_score = heatmaps.reshape(-1)
    out: dict[str, float | str] = {
        "image_auroc": _safe_auc(labels, image_scores),
        "pixel_auroc": _safe_auc(pixel_true, pixel_score),
        "pixel_ap": _safe_ap(pixel_true, pixel_score),
        "aupro": aupro_score(masks, heatmaps),
        "aupro_protocol": "component_pro_auc_fpr_0.3_v2",
    }
    oracle = best_f1_iou_dice(pixel_true, pixel_score)
    out.update({f"oracle_{key}": value for key, value in oracle.items()})
    if pixel_threshold is None:
        out.update(oracle)
        out["threshold_protocol"] = threshold_protocol or "oracle_test_gt"
    else:
        out.update(binary_metrics_at_threshold(pixel_true, pixel_score, pixel_threshold))
        out["threshold_protocol"] = threshold_protocol or "fixed_external"
    out["aupro_proxy"] = out["pixel_auroc"]
    return out
