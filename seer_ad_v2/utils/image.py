from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".JPG"}


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in {e.lower() for e in IMAGE_EXTENSIONS}])


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str | Path, size: int | tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("L")
    if isinstance(size, int):
        size = (size, size)
    return img.resize(size, Image.NEAREST)


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    x = (x.clamp(-1, 1) + 1.0) * 127.5
    arr = x.permute(1, 2, 0).numpy().astype(np.uint8)
    return arr


def heatmap_to_uint8(h: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(h, torch.Tensor):
        h = h.detach().float().cpu().numpy()
    h = np.squeeze(h).astype(np.float32)
    h -= float(h.min())
    denom = float(h.max()) + 1e-8
    return np.clip(h / denom * 255.0, 0, 255).astype(np.uint8)


def save_image(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray(arr).save(path)
    else:
        Image.fromarray(arr[..., :3]).save(path)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(np.nanmin(x))
    return x / (float(np.nanmax(x)) + 1e-8)


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)
