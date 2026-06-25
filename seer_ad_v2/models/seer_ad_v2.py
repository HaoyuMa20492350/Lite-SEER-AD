from __future__ import annotations

import torch

from seer_ad_v2.models.diffusion.ddpm import GaussianDiffusion
from seer_ad_v2.models.diffusion.unet import TinyUNet
from seer_ad_v2.models.region_verifier.hn_sev import HNSEV
from seer_ad_v2.models.scheduler.lc_rds import LCRDS


def build_diffusion_components(base_channels: int, timesteps: int, device: str) -> tuple[TinyUNet, GaussianDiffusion]:
    model = TinyUNet(base_channels=base_channels).to(device)
    diffusion = GaussianDiffusion(timesteps=timesteps, device=device)
    return model, diffusion


def build_hn_sev(base_channels: int, device: str) -> HNSEV:
    return HNSEV(base_channels=base_channels).to(device)


def build_lc_rds(device: str) -> LCRDS:
    return LCRDS().to(device)
