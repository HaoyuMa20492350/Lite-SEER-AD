from __future__ import annotations

import numpy as np
import torch

from seer_ad_v2.utils.image import normalize01, sobel_magnitude


@torch.no_grad()
def reconstruct_batch(model: torch.nn.Module, diffusion, images: torch.Tensor, steps: int) -> torch.Tensor:
    return diffusion.reconstruct(model, images, steps=steps)


def residual_heatmap(original: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    pixel = (original - reconstruction).abs().mean(dim=1, keepdim=True)
    return pixel


def fused_residual_heatmap(original: torch.Tensor, reconstruction: torch.Tensor) -> np.ndarray:
    pixel = residual_heatmap(original, reconstruction)[0, 0].detach().cpu().numpy()
    orig_gray = original[0].detach().cpu().mean(dim=0).numpy()
    rec_gray = reconstruction[0].detach().cpu().mean(dim=0).numpy()
    grad = np.abs(sobel_magnitude(orig_gray) - sobel_magnitude(rec_gray))
    return normalize01(0.75 * normalize01(pixel) + 0.25 * normalize01(grad))
