from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HNSEV(nn.Module):
    def __init__(self, in_channels: int = 8, base_channels: int = 24) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, 3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 4, 3, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 4, c * 2), nn.SiLU(), nn.Linear(c * 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x)).squeeze(1)


def build_sev_input(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    residual: torch.Tensor,
    prototype_distance: torch.Tensor | float | None = None,
    feature_anomaly: torch.Tensor | None = None,
    feature_gap: torch.Tensor | float | None = None,
) -> torch.Tensor:
    if residual.ndim == 3:
        residual = residual.unsqueeze(1)
    if residual.shape[1] != 1:
        residual = residual.mean(dim=1, keepdim=True)
    b, _, h, w = original.shape
    if prototype_distance is None:
        proto = torch.zeros(b, 1, h, w, device=original.device, dtype=original.dtype)
    elif isinstance(prototype_distance, torch.Tensor):
        proto = prototype_distance.to(original.device, original.dtype).view(b, 1, 1, 1).expand(b, 1, h, w)
    else:
        proto = torch.full((b, 1, h, w), float(prototype_distance), device=original.device, dtype=original.dtype)
    parts = [original, reconstruction, residual, proto]
    if feature_anomaly is not None:
        if feature_anomaly.ndim == 3:
            feature_anomaly = feature_anomaly.unsqueeze(1)
        if feature_anomaly.shape[1] != 1:
            feature_anomaly = feature_anomaly.mean(dim=1, keepdim=True)
        if feature_anomaly.shape[-2:] != (h, w):
            feature_anomaly = F.interpolate(feature_anomaly, size=(h, w), mode="bilinear", align_corners=False)
        parts.append(feature_anomaly.to(original.device, original.dtype))
    if feature_gap is not None:
        if isinstance(feature_gap, torch.Tensor):
            gap = feature_gap.to(original.device, original.dtype).view(b, 1, 1, 1).expand(b, 1, h, w)
        else:
            gap = torch.full((b, 1, h, w), float(feature_gap), device=original.device, dtype=original.dtype)
        parts.append(gap)
    return torch.cat(parts, dim=1)


def sev_probability(model: HNSEV, x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(model(x))


def binary_focal_bce(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 1.5) -> torch.Tensor:
    labels = labels.float()
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * labels + (1 - p) * (1 - labels)
    return ((1 - pt).pow(gamma) * bce).mean()
