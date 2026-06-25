from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from seer_ad_v2.data.hard_negative_mining import ROI


def roi_mask(shape: tuple[int, int], roi: ROI, feather: int = 7) -> torch.Tensor:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    mask[roi.y1 : roi.y2, roi.x1 : roi.x2] = 1.0
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather)
        mask = np.clip(mask, 0.0, 1.0)
    return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def local_repair(
    image: torch.Tensor,
    model: torch.nn.Module,
    diffusion,
    roi: ROI,
    steps: int,
    native_size: int | None = None,
    reference_patch: torch.Tensor | None = None,
    reference_weight: float = 0.0,
    reference_mode: str = "raw",
) -> torch.Tensor:
    """Repair a single ROI, optionally initialized toward a retrieved normal patch."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if steps <= 0:
        return image
    crop = image[..., roi.y1 : roi.y2, roi.x1 : roi.x2]
    if crop.numel() == 0:
        return image
    target_h, target_w = crop.shape[-2:]
    if native_size is not None:
        work = F.interpolate(crop, size=(native_size, native_size), mode="bilinear", align_corners=False)
    else:
        work = crop
    if reference_patch is not None and reference_weight > 0.0:
        if reference_patch.ndim == 3:
            reference_patch = reference_patch.unsqueeze(0)
        reference_patch = F.interpolate(reference_patch.to(work), size=work.shape[-2:], mode="bilinear", align_corners=False)
        if reference_mode == "texture":
            kernel = max(3, min(work.shape[-2:]) // 4)
            if kernel % 2 == 0:
                kernel += 1
            reference_low = F.avg_pool2d(reference_patch, kernel_size=kernel, stride=1, padding=kernel // 2)
            work_low = F.avg_pool2d(work, kernel_size=kernel, stride=1, padding=kernel // 2)
            reference_patch = (work_low + reference_patch - reference_low).clamp(-1, 1)
        elif reference_mode != "raw":
            raise ValueError(f"Unknown retrieval reference mode: {reference_mode}")
        weight = float(np.clip(reference_weight, 0.0, 1.0))
        work = (1.0 - weight) * work + weight * reference_patch
    repaired = diffusion.reconstruct(model, work, steps=steps)
    repaired = F.interpolate(repaired, size=(target_h, target_w), mode="bilinear", align_corners=False)
    out = image.clone()
    out[..., roi.y1 : roi.y2, roi.x1 : roi.x2] = repaired
    mask = roi_mask(image.shape[-2:], roi).to(image.device)
    return image * (1.0 - mask) + out * mask
