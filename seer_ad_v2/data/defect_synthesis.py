from __future__ import annotations

import cv2
import numpy as np
import torch

from seer_ad_v2.data.datasets import DTDTextureDataset


def random_blob_mask(
    h: int,
    w: int,
    min_blobs: int = 1,
    max_blobs: int = 5,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    rng = rng or np.random
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.randint(min_blobs, max_blobs + 1)):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        rx = rng.randint(max(2, w // 16), max(3, w // 4))
        ry = rng.randint(max(2, h // 16), max(3, h // 4))
        angle = rng.uniform(0, 180)
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)
    if rng.rand() < 0.35:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, min(h, w) / 64.0))
    return np.clip(mask, 0.0, 1.0)


def random_scratch_mask(h: int, w: int, rng: np.random.RandomState | None = None) -> np.ndarray:
    rng = rng or np.random
    mask = np.zeros((h, w), dtype=np.float32)
    count = int(rng.randint(1, 4))
    for _ in range(count):
        points = []
        x, y = int(rng.randint(0, w)), int(rng.randint(0, h))
        points.append((x, y))
        for _ in range(int(rng.randint(1, 4))):
            x = int(np.clip(x + rng.randint(-max(2, w // 3), max(3, w // 3)), 0, w - 1))
            y = int(np.clip(y + rng.randint(-max(2, h // 3), max(3, h // 3)), 0, h - 1))
            points.append((x, y))
        width = int(rng.randint(max(1, min(h, w) // 128), max(2, min(h, w) // 32)))
        cv2.polylines(mask, [np.asarray(points, dtype=np.int32)], False, 1.0, width, cv2.LINE_AA)
    sigma = max(0.5, min(h, w) / 256.0)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma), 0.0, 1.0)


def random_spot_mask(h: int, w: int, rng: np.random.RandomState | None = None) -> np.ndarray:
    rng = rng or np.random
    mask = np.zeros((h, w), dtype=np.float32)
    count = int(rng.randint(1, 8))
    min_radius = max(1, min(h, w) // 128)
    max_radius = max(min_radius + 1, min(h, w) // 24)
    for _ in range(count):
        center = (int(rng.randint(0, w)), int(rng.randint(0, h)))
        radius = int(rng.randint(min_radius, max_radius))
        cv2.circle(mask, center, radius, 1.0, -1, cv2.LINE_AA)
    sigma = max(0.5, min(h, w) / 384.0)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma), 0.0, 1.0)


def random_patch_mask(h: int, w: int, rng: np.random.RandomState | None = None) -> np.ndarray:
    rng = rng or np.random
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = int(rng.randint(0, w)), int(rng.randint(0, h))
    rw = int(rng.randint(max(2, w // 16), max(3, w // 3)))
    rh = int(rng.randint(max(2, h // 16), max(3, h // 3)))
    points = np.asarray(
        [
            [cx - rw // 2 + rng.randint(-max(1, rw // 4), max(2, rw // 4)), cy - rh // 2],
            [cx + rw // 2, cy - rh // 2 + rng.randint(-max(1, rh // 4), max(2, rh // 4))],
            [cx + rw // 2 + rng.randint(-max(1, rw // 4), max(2, rw // 4)), cy + rh // 2],
            [cx - rw // 2, cy + rh // 2 + rng.randint(-max(1, rh // 4), max(2, rh // 4))],
        ],
        dtype=np.int32,
    )
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [points], 1.0, cv2.LINE_AA)
    sigma = max(0.75, min(h, w) / 192.0)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma), 0.0, 1.0)


def anomaly_mask(
    h: int,
    w: int,
    mode: str,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    if mode == "blob":
        return random_blob_mask(h, w, rng=rng)
    if mode == "scratch":
        return random_scratch_mask(h, w, rng=rng)
    if mode == "spot":
        return random_spot_mask(h, w, rng=rng)
    if mode == "patch":
        return random_patch_mask(h, w, rng=rng)
    raise ValueError(f"Unknown anomaly mask mode: {mode}")


def synthesize_anomaly(
    image: torch.Tensor,
    texture_bank: DTDTextureDataset | None = None,
    strength: float | None = None,
    rng: np.random.RandomState | None = None,
    mask_mode: str = "blob",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic anomaly on a normalized [-1, 1] image tensor."""
    rng = rng or np.random
    c, h, w = image.shape
    mask_np = anomaly_mask(h, w, mask_mode, rng=rng)
    if strength is None:
        strength_ranges = {
            "blob": (0.35, 0.85),
            "scratch": (0.25, 0.70),
            "spot": (0.25, 0.75),
            "patch": (0.20, 0.65),
        }
        lo, hi = strength_ranges.get(mask_mode, (0.35, 0.85))
        strength = float(rng.uniform(lo, hi))

    if texture_bank is not None:
        tex = texture_bank.sample((w, h), rng=rng)
    else:
        tex = rng.rand(h, w, 3).astype(np.float32)
        tex = cv2.GaussianBlur(tex, (0, 0), sigmaX=rng.uniform(0.5, 2.5))

    tex_t = torch.from_numpy(tex).permute(2, 0, 1).float() * 2.0 - 1.0
    mask = torch.from_numpy(mask_np).float().unsqueeze(0)
    out = image * (1.0 - mask * strength) + tex_t * (mask * strength)
    return out.clamp(-1, 1), (mask > 0.15).float()
