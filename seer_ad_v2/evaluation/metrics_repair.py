from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def repair_metrics(original: np.ndarray, repaired: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    original = original.astype(np.float32)
    repaired = repaired.astype(np.float32)
    if original.max() > 1.5:
        original = original / 255.0
    if repaired.max() > 1.5:
        repaired = repaired / 255.0
    m = (mask.squeeze() > 0).astype(np.float32)
    bg = 1.0 - m
    psnr = peak_signal_noise_ratio(original, repaired, data_range=1.0)
    ssim = structural_similarity(original, repaired, channel_axis=-1, data_range=1.0)
    if bg.sum() > 0:
        diff = ((original - repaired) ** 2 * bg[..., None]).sum() / (bg.sum() * original.shape[-1] + 1e-8)
        bg_psnr = 10.0 * np.log10(1.0 / (float(diff) + 1e-8))
    else:
        bg_psnr = float("nan")
    return {"psnr": float(psnr), "ssim": float(ssim), "background_psnr": float(bg_psnr)}
