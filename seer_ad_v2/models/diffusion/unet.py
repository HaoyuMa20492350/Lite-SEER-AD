from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(1, half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, time_dim: int = 128) -> None:
        super().__init__()
        c = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.time_dim = time_dim
        self.in_conv = nn.Conv2d(in_channels, c, 3, padding=1)
        self.down1 = ResBlock(c, c, time_dim)
        self.down2 = ResBlock(c, c * 2, time_dim)
        self.down3 = ResBlock(c * 2, c * 4, time_dim)
        self.mid = ResBlock(c * 4, c * 4, time_dim)
        self.up2 = ResBlock(c * 4 + c * 2, c * 2, time_dim)
        self.up1 = ResBlock(c * 2 + c, c, time_dim)
        self.out_norm = nn.GroupNorm(min(8, c), c)
        self.out = nn.Conv2d(c, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        x0 = self.in_conv(x)
        d1 = self.down1(x0, temb)
        d2 = self.down2(F.avg_pool2d(d1, 2), temb)
        d3 = self.down3(F.avg_pool2d(d2, 2), temb)
        m = self.mid(d3, temb)
        u2 = F.interpolate(m, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, d2], dim=1), temb)
        u1 = F.interpolate(u2, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, d1], dim=1), temb)
        return self.out(F.silu(self.out_norm(u1)))
