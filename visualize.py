from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from seer_ad_v2.utils.image import heatmap_to_uint8, save_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create heatmap overlay visualizations.")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--heatmap-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--alpha", type=float, default=0.45)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted([p for p in Path(args.image_dir).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    heatmap_paths = sorted([p for p in Path(args.heatmap_dir).glob("*.png")])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path, hm_path in zip(image_paths, heatmap_paths):
        img = np.asarray(Image.open(img_path).convert("RGB").resize(Image.open(hm_path).size))
        hm = np.asarray(Image.open(hm_path).convert("L"))
        color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)[..., ::-1]
        overlay = np.clip((1.0 - args.alpha) * img + args.alpha * color, 0, 255).astype(np.uint8)
        save_image(out_dir / f"{img_path.stem}_overlay.png", overlay)


if __name__ == "__main__":
    main()
