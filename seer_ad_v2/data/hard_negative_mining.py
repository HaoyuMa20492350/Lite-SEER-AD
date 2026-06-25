from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int
    area: int
    peak: float

    def as_list(self) -> list[int | float]:
        return [self.x1, self.y1, self.x2, self.y2, self.area, self.peak]


def heatmap_to_rois(
    heatmap: np.ndarray,
    threshold_quantile: float = 0.985,
    min_area: int = 16,
    max_rois: int = 5,
    pad: int = 8,
) -> list[ROI]:
    h = np.squeeze(heatmap).astype(np.float32)
    if h.size == 0:
        return []
    thr = float(np.quantile(h, threshold_quantile))
    binary = (h >= thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rois: list[ROI] = []
    height, width = h.shape
    for idx in range(1, n):
        x, y, w, hh, area = stats[idx]
        if int(area) < min_area:
            continue
        x1 = max(0, int(x) - pad)
        y1 = max(0, int(y) - pad)
        x2 = min(width, int(x + w) + pad)
        y2 = min(height, int(y + hh) + pad)
        peak = float(h[y1:y2, x1:x2].max()) if y2 > y1 and x2 > x1 else 0.0
        rois.append(ROI(x1, y1, x2, y2, int(area), peak))
    rois.sort(key=lambda r: (r.peak, r.area), reverse=True)
    if not rois:
        y, x = np.unravel_index(int(np.argmax(h)), h.shape)
        half = max(4, int(np.sqrt(max(1, min_area))) + pad)
        x1 = max(0, int(x) - half)
        y1 = max(0, int(y) - half)
        x2 = min(width, int(x) + half)
        y2 = min(height, int(y) + half)
        rois.append(ROI(x1, y1, x2, y2, int((x2 - x1) * (y2 - y1)), float(h[y, x])))
    return rois[:max_rois]


def crop_resize(tensor: torch.Tensor, roi: ROI, size: int) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    crop = tensor[..., roi.y1 : roi.y2, roi.x1 : roi.x2]
    if crop.shape[-1] == 0 or crop.shape[-2] == 0:
        crop = tensor
    return F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)[0]


def save_roi_npz(
    path: str | Path,
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    heatmap: np.ndarray,
    roi: ROI,
    source_path: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        original=original.detach().cpu().numpy(),
        reconstruction=reconstruction.detach().cpu().numpy(),
        residual=heatmap.astype(np.float32),
        roi=np.asarray(roi.as_list(), dtype=np.float32),
        source_path=np.asarray(source_path),
    )
