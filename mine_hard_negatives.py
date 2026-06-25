from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seer_ad_v2.config import (
    cfg_device,
    cfg_first,
    cfg_float,
    cfg_int,
    cfg_seed,
    dataset_category,
    diffusion_base_channels,
    diffusion_timesteps,
    image_size as cfg_image_size,
    load_config,
    make_run_dir,
    max_regions,
    reconstruction_steps,
    resolve_device,
)
from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.data.hard_negative_mining import heatmap_to_rois, save_roi_npz
from seer_ad_v2.models.diffusion.reconstruction import fused_residual_heatmap
from seer_ad_v2.models.seer_ad_v2 import build_diffusion_components
from seer_ad_v2.utils.io import load_checkpoint
from seer_ad_v2.utils.run import save_run_metadata
from seer_ad_v2.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine normal high-residual hard negatives.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--category", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--reconstruction-steps", type=int, default=None)
    p.add_argument("--run-name", default="default")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ckpt = load_checkpoint(args.checkpoint)
    seed_everything(cfg_seed(cfg, args.seed))
    category = args.category or ckpt.get("category") or dataset_category(cfg)
    image_size = cfg_image_size(cfg, args.image_size, ckpt.get("image_size"))
    device = resolve_device(cfg_device(cfg, args.device))
    run_dir = make_run_dir(cfg, args.run_name)
    save_run_metadata(run_dir, cfg, args, device, "mine_hard_negatives")
    out_dir = run_dir / "hard_negatives"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        cfg_first(cfg, ("dataset.name",), "mvtec"),
        cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD"),
        category,
        "train",
        image_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model, diffusion = build_diffusion_components(
        diffusion_base_channels(cfg, ckpt.get("base_channels")),
        diffusion_timesteps(cfg, ckpt.get("timesteps")),
        device,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    recon_steps = int(args.reconstruction_steps or reconstruction_steps(cfg))

    rows: list[dict[str, str | int | float]] = []
    roi_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="mine hard negatives", leave=False):
            x = batch["image"].to(device)
            recon = diffusion.reconstruct(model, x, steps=recon_steps)
            heatmap = fused_residual_heatmap(x, recon)
            rois = heatmap_to_rois(
                heatmap,
                threshold_quantile=cfg_float(cfg, ("roi.threshold_quantile",), 0.985),
                min_area=cfg_int(cfg, ("roi.min_area",), 16),
                max_rois=max_regions(cfg),
                pad=cfg_int(cfg, ("roi.pad",), 8),
            )
            for roi in rois:
                out_path = out_dir / f"hn_{roi_idx:06d}.npz"
                save_roi_npz(out_path, x[0], recon[0], heatmap, roi, batch["path"][0])
                rows.append(
                    {
                        "path": str(out_path),
                        "source_path": batch["path"][0],
                        "x1": roi.x1,
                        "y1": roi.y1,
                        "x2": roi.x2,
                        "y2": roi.y2,
                        "area": roi.area,
                        "peak": roi.peak,
                    }
                )
                roi_idx += 1

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source_path", "x1", "y1", "x2", "y2", "area", "peak"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} hard negatives to {out_dir}")


if __name__ == "__main__":
    main()
