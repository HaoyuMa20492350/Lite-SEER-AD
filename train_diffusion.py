from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seer_ad_v2.config import (
    cfg_device,
    cfg_first,
    cfg_int,
    cfg_seed,
    dataset_category,
    diffusion_base_channels,
    diffusion_timesteps,
    image_size as cfg_image_size,
    load_config,
    make_run_dir,
    resolve_device,
)
from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.models.seer_ad_v2 import build_diffusion_components
from seer_ad_v2.utils.io import save_checkpoint, save_json
from seer_ad_v2.utils.run import save_run_metadata
from seer_ad_v2.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train lightweight normal-domain diffusion reconstruction.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--category", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--run-name", default="default")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    category = dataset_category(cfg, args.category)
    image_size = cfg_image_size(cfg, args.image_size)
    batch_size = args.batch_size or cfg_int(cfg, ("training.batch_size",), 8)
    epochs = args.epochs or cfg_int(cfg, ("training.epochs",), 20)
    seed_everything(cfg_seed(cfg, args.seed))
    device = resolve_device(cfg_device(cfg, args.device))
    run_dir = make_run_dir(cfg, args.run_name)
    save_run_metadata(run_dir, cfg, args, device, "train_diffusion")

    dataset = build_dataset(
        cfg_first(cfg, ("dataset.name",), "mvtec"),
        cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD"),
        category,
        "train",
        image_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg_int(cfg, ("dataset.num_workers",), 0),
        pin_memory=device.startswith("cuda"),
    )
    model, diffusion = build_diffusion_components(
        base_channels=diffusion_base_channels(cfg),
        timesteps=diffusion_timesteps(cfg),
        device=device,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg_first(cfg, ("training.lr",), 2e-4)), weight_decay=1e-4)
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        bar = tqdm(loader, desc=f"diffusion epoch {epoch + 1}/{epochs}", leave=False)
        epoch_losses: list[float] = []
        for batch in bar:
            x = batch["image"].to(device)
            loss = diffusion.loss(model, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            val = float(loss.detach().cpu())
            epoch_losses.append(val)
            bar.set_postfix(loss=f"{val:.4f}")
        losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))

    save_checkpoint(
        run_dir / "diffusion.pt",
        model_state=model.state_dict(),
        cfg=cfg,
        category=category,
        image_size=image_size,
        timesteps=diffusion_timesteps(cfg),
        base_channels=diffusion_base_channels(cfg),
    )
    save_json({"loss": losses, "num_train": len(dataset), "category": category}, run_dir / "metrics.json")
    print(f"Saved diffusion checkpoint to {run_dir / 'diffusion.pt'}")


if __name__ == "__main__":
    main()
