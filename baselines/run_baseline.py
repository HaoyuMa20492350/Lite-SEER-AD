from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, Wide_ResNet50_2_Weights, resnet18, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.registry import BASELINES, LOCAL_BASELINES
from seer_ad_v2.config import cfg_device, cfg_first, cfg_seed, dataset_category, image_size as cfg_image_size, load_config, make_run_dir, resolve_device
from seer_ad_v2.data.defect_synthesis import synthesize_anomaly
from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.metrics_efficiency import benchmark_callable
from seer_ad_v2.evaluation.metrics_plan import efficiency_summary
from seer_ad_v2.evaluation.pareto import write_pareto
from seer_ad_v2.models.diffusion.unet import TinyUNet
from seer_ad_v2.utils.image import heatmap_to_uint8, save_image, tensor_to_uint8
from seer_ad_v2.utils.io import save_json
from seer_ad_v2.utils.run import save_run_metadata
from seer_ad_v2.utils.seed import seed_everything


METHODS = set(LOCAL_BASELINES)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a local baseline with the Lite-SEER-AD output contract.")
    p.add_argument("--method", choices=sorted(METHODS), required=True)
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--category", default=None)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--train-max-samples", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-name", default=None)
    p.add_argument("--coreset-ratio", type=float, default=0.1)
    p.add_argument("--padim-dim", type=int, default=100)
    p.add_argument("--simplenet-epochs", type=int, default=2)
    p.add_argument("--simplenet-hidden-dim", type=int, default=384)
    p.add_argument("--simplenet-lr", type=float, default=1e-3)
    p.add_argument("--simplenet-noise-std", type=float, default=0.05)
    p.add_argument("--simplenet-max-patches", type=int, default=4096)
    p.add_argument("--draem-epochs", type=int, default=2)
    p.add_argument("--draem-lr", type=float, default=2e-4)
    p.add_argument("--draem-base-channels", type=int, default=16)
    p.add_argument("--draem-recon-weight", type=float, default=1.0)
    p.add_argument("--draem-seg-weight", type=float, default=1.0)
    p.add_argument("--uniad-epochs", type=int, default=2)
    p.add_argument("--uniad-lr", type=float, default=2e-4)
    p.add_argument("--uniad-patch-size", type=int, default=16)
    p.add_argument("--uniad-embed-dim", type=int, default=64)
    p.add_argument("--uniad-depth", type=int, default=2)
    p.add_argument("--uniad-heads", type=int, default=4)
    p.add_argument("--diffusionad-epochs", type=int, default=2)
    p.add_argument("--diffusionad-lr", type=float, default=2e-4)
    p.add_argument("--diffusionad-base-channels", type=int, default=16)
    p.add_argument("--diffusionad-timesteps", type=int, default=50)
    p.add_argument("--diffusionad-score-timestep", type=int, default=25)
    p.add_argument("--ddad-epochs", type=int, default=2)
    p.add_argument("--ddad-lr", type=float, default=2e-4)
    p.add_argument("--ddad-base-channels", type=int, default=16)
    p.add_argument("--ddad-timesteps", type=int, default=50)
    p.add_argument("--ddad-low-timestep", type=int, default=12)
    p.add_argument("--ddad-high-timestep", type=int, default=38)
    p.add_argument("--rd4ad-epochs", type=int, default=2)
    p.add_argument("--rd4ad-lr", type=float, default=1e-4)
    p.add_argument("--latency-warmups", type=int, default=50)
    p.add_argument("--latency-repeats", type=int, default=200)
    p.add_argument("--latency-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--allow-random-weights", action="store_true", help="Use only for debugging when ImageNet weights cannot be loaded.")
    return p.parse_args()


def _metric_csv(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


def _source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _preprocess(x: torch.Tensor, device: str) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    x01 = (x.clamp(-1, 1) + 1.0) * 0.5
    return (x01 - mean) / std


def _build_extractor(method: str, device: str, allow_random: bool) -> tuple[torch.nn.Module, list[str]]:
    try:
        if method in {"patchcore", "simplenet"}:
            backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
            return_nodes = {"layer2": "layer2", "layer3": "layer3"}
            names = ["layer2", "layer3"]
        else:
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            return_nodes = {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3"}
            names = ["layer1", "layer2", "layer3"]
    except Exception as exc:
        if not allow_random:
            raise RuntimeError(
                "Failed to load ImageNet pretrained weights. Re-run with network/cache available, "
                "or use --allow-random-weights only for pipeline debugging."
            ) from exc
        if method in {"patchcore", "simplenet"}:
            backbone = wide_resnet50_2(weights=None)
            return_nodes = {"layer2": "layer2", "layer3": "layer3"}
            names = ["layer2", "layer3"]
        else:
            backbone = resnet18(weights=None)
            return_nodes = {"layer1": "layer1", "layer2": "layer2", "layer3": "layer3"}
            names = ["layer1", "layer2", "layer3"]
    extractor = create_feature_extractor(backbone, return_nodes=return_nodes).to(device)
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad_(False)
    return extractor, names


def _build_resnet18_feature_extractor(device: str) -> tuple[torch.nn.Module, list[str]]:
    backbone = resnet18(weights=None)
    names = ["layer1", "layer2", "layer3"]
    extractor = create_feature_extractor(backbone, return_nodes={name: name for name in names}).to(device)
    return extractor, names


@torch.no_grad()
def _feature_map(extractor: torch.nn.Module, names: list[str], x: torch.Tensor, device: str) -> torch.Tensor:
    feats = extractor(_preprocess(x, device))
    target_size = feats[names[0]].shape[-2:]
    maps = []
    for name in names:
        value = feats[name]
        if value.shape[-2:] != target_size:
            value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
        maps.append(value)
    return torch.cat(maps, dim=1)


def _patch_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    return feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])


@torch.no_grad()
def _build_patchcore_bank(
    extractor: torch.nn.Module,
    names: list[str],
    loader: DataLoader,
    device: str,
    coreset_ratio: float,
    seed: int,
) -> torch.Tensor:
    chunks = []
    for batch in tqdm(loader, desc="patchcore:train", leave=False):
        fmap = _feature_map(extractor, names, batch["image"].to(device), device)
        chunks.append(_patch_matrix(fmap).cpu())
    bank = torch.cat(chunks, dim=0)
    ratio = max(0.0, min(1.0, float(coreset_ratio)))
    if 0.0 < ratio < 1.0 and len(bank) > 1:
        generator = torch.Generator().manual_seed(seed)
        keep = max(1, int(len(bank) * ratio))
        index = torch.randperm(len(bank), generator=generator)[:keep]
        bank = bank[index]
    return bank.to(device)


def _min_distances(patches: torch.Tensor, bank: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    mins = []
    for start in range(0, len(patches), chunk_size):
        chunk = patches[start : start + chunk_size]
        dists = torch.cdist(chunk, bank)
        mins.append(dists.min(dim=1).values)
    return torch.cat(mins, dim=0)


@torch.no_grad()
def _patchcore_scores(
    extractor: torch.nn.Module,
    names: list[str],
    bank: torch.Tensor,
    images: torch.Tensor,
    device: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    fmap = _feature_map(extractor, names, images.to(device), device)
    b, _, h, w = fmap.shape
    patches = _patch_matrix(fmap)
    scores = _min_distances(patches, bank).view(b, h, w)
    heatmaps = F.interpolate(scores.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
    heatmaps_np = heatmaps.detach().cpu().numpy().astype(np.float32)
    image_scores = np.asarray([float(hm.max()) for hm in heatmaps_np], dtype=np.float32)
    return image_scores, heatmaps_np


@torch.no_grad()
def _build_padim_stats(
    extractor: torch.nn.Module,
    names: list[str],
    loader: DataLoader,
    device: str,
    max_dim: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    chunks = []
    for batch in tqdm(loader, desc="padim:train", leave=False):
        chunks.append(_feature_map(extractor, names, batch["image"].to(device), device).cpu())
    features = torch.cat(chunks, dim=0)
    channels = features.shape[1]
    dim = min(int(max_dim), channels)
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(channels, generator=generator)[:dim]
    features = features[:, selected].to(device)
    n, d, h, w = features.shape
    flat = features.permute(2, 3, 0, 1).reshape(h * w, n, d)
    mean = flat.mean(dim=1)
    centered = flat - mean.unsqueeze(1)
    cov = centered.transpose(1, 2).matmul(centered) / max(1, n - 1)
    eye = torch.eye(d, device=device).unsqueeze(0)
    inv_cov = torch.linalg.pinv(cov + 0.01 * eye)
    return {"selected": selected.to(device), "mean": mean, "inv_cov": inv_cov, "height": torch.tensor(h), "width": torch.tensor(w)}


@torch.no_grad()
def _padim_scores(
    extractor: torch.nn.Module,
    names: list[str],
    stats: dict[str, torch.Tensor],
    images: torch.Tensor,
    device: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    fmap = _feature_map(extractor, names, images.to(device), device)[:, stats["selected"].long()]
    b, d, h, w = fmap.shape
    patches = fmap.permute(2, 3, 0, 1).reshape(h * w, b, d)
    diff = patches - stats["mean"].unsqueeze(1)
    dist2 = torch.einsum("lbd,lde,lbe->lb", diff, stats["inv_cov"], diff).clamp_min(0.0)
    scores = torch.sqrt(dist2 + 1e-8).transpose(0, 1).reshape(b, h, w)
    heatmaps = F.interpolate(scores.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
    heatmaps_np = heatmaps.detach().cpu().numpy().astype(np.float32)
    image_scores = np.asarray([float(hm.max()) for hm in heatmaps_np], dtype=np.float32)
    return image_scores, heatmaps_np


class SimpleNetHead(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 384) -> None:
        super().__init__()
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(max(32, hidden_dim // 2), 1),
        )

    def adapt(self, patches: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.adapter(patches), dim=1)

    def discriminate(self, features: torch.Tensor) -> torch.Tensor:
        return self.discriminator(features).squeeze(1)


def _sample_patches(patches: torch.Tensor, max_patches: int, seed: int, epoch: int, step: int) -> torch.Tensor:
    if max_patches <= 0 or len(patches) <= max_patches:
        return patches
    generator = torch.Generator(device=patches.device).manual_seed(int(seed + epoch * 100_003 + step))
    index = torch.randperm(len(patches), generator=generator, device=patches.device)[:max_patches]
    return patches[index]


@torch.no_grad()
def _feature_dim(extractor: torch.nn.Module, names: list[str], loader: DataLoader, device: str) -> int:
    batch = next(iter(loader))
    fmap = _feature_map(extractor, names, batch["image"].to(device), device)
    return int(fmap.shape[1])


def _train_simplenet(
    extractor: torch.nn.Module,
    names: list[str],
    loader: DataLoader,
    device: str,
    hidden_dim: int,
    epochs: int,
    lr: float,
    noise_std: float,
    max_patches: int,
    seed: int,
) -> SimpleNetHead:
    in_dim = _feature_dim(extractor, names, loader, device)
    head = SimpleNetHead(in_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=1e-4)
    for epoch in range(max(1, int(epochs))):
        head.train()
        for step, batch in enumerate(tqdm(loader, desc=f"simplenet:train:{epoch + 1}", leave=False)):
            with torch.no_grad():
                fmap = _feature_map(extractor, names, batch["image"].to(device), device)
                patches = _sample_patches(_patch_matrix(fmap), max_patches, seed, epoch, step)
            normal = head.adapt(patches)
            generated = normal + torch.randn_like(normal) * float(noise_std)
            logits = torch.cat([head.discriminate(normal), head.discriminate(generated)], dim=0)
            labels = torch.cat([torch.zeros(len(normal), device=device), torch.ones(len(generated), device=device)], dim=0)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def _simplenet_scores(
    extractor: torch.nn.Module,
    names: list[str],
    head: SimpleNetHead,
    images: torch.Tensor,
    device: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    fmap = _feature_map(extractor, names, images.to(device), device)
    b, _, h, w = fmap.shape
    patches = _patch_matrix(fmap)
    scores = torch.sigmoid(head.discriminate(head.adapt(patches))).view(b, h, w)
    heatmaps = F.interpolate(scores.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
    heatmaps_np = heatmaps.detach().cpu().numpy().astype(np.float32)
    image_scores = np.asarray([float(hm.max()) for hm in heatmaps_np], dtype=np.float32)
    return image_scores, heatmaps_np


class DraemLite(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self.reconstructor = TinyUNet(in_channels=3, base_channels=base_channels)
        c = int(base_channels)
        self.segmenter = nn.Sequential(
            nn.Conv2d(7, c, 3, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(),
            nn.Conv2d(c, c * 2, 3, padding=1),
            nn.GroupNorm(min(8, c * 2), c * 2),
            nn.SiLU(),
            nn.Conv2d(c * 2, c, 3, padding=1),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(),
            nn.Conv2d(c, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.zeros(x.shape[0], device=x.device)
        recon = torch.tanh(self.reconstructor(x, t))
        residual = (x - recon).abs().mean(dim=1, keepdim=True)
        logits = self.segmenter(torch.cat([x, recon, residual], dim=1))
        return recon, logits


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def _synth_batch(clean: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    synths = []
    masks = []
    for image in clean.detach().cpu():
        synth, mask = synthesize_anomaly(image)
        synths.append(synth)
        masks.append(mask)
    return torch.stack(synths).to(device), torch.stack(masks).to(device)


def _train_draem(
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    base_channels: int,
    recon_weight: float,
    seg_weight: float,
) -> DraemLite:
    model = DraemLite(base_channels=base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-5)
    for epoch in range(max(1, int(epochs))):
        model.train()
        for batch in tqdm(loader, desc=f"draem:train:{epoch + 1}", leave=False):
            clean = batch["image"].to(device)
            synth, target_mask = _synth_batch(clean, device)
            recon, logits = model(synth)
            loss_recon = F.l1_loss(recon, clean)
            loss_seg = F.binary_cross_entropy_with_logits(logits, target_mask) + _dice_loss(logits, target_mask)
            loss = float(recon_weight) * loss_recon + float(seg_weight) * loss_seg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def _normalize_heatmaps(maps: torch.Tensor) -> torch.Tensor:
    b = maps.shape[0]
    flat = maps.reshape(b, -1)
    lo = flat.min(dim=1).values.view(b, 1, 1)
    hi = flat.max(dim=1).values.view(b, 1, 1)
    return (maps - lo) / (hi - lo + 1e-6)


@torch.no_grad()
def _draem_scores(model: DraemLite, images: torch.Tensor, device: str) -> tuple[np.ndarray, np.ndarray]:
    x = images.to(device)
    recon, logits = model(x)
    residual = (x - recon).abs().mean(dim=1)
    prob = torch.sigmoid(logits)[:, 0]
    heatmaps = (0.5 * _normalize_heatmaps(residual) + 0.5 * prob).clamp(0.0, 1.0)
    heatmaps_np = heatmaps.detach().cpu().numpy().astype(np.float32)
    image_scores = np.asarray([float(hm.max()) for hm in heatmaps_np], dtype=np.float32)
    return image_scores, heatmaps_np


class UniADLite(nn.Module):
    def __init__(self, image_size: int, patch_size: int = 16, embed_dim: int = 64, depth: int = 2, heads: int = 4) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size must be divisible by patch_size, got {image_size} and {patch_size}")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        grid = self.image_size // self.patch_size
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(depth)))
        self.decoder = nn.Linear(embed_dim, 3 * self.patch_size * self.patch_size)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        if orig_size != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        h = self.image_size // self.patch_size
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens + self.pos)
        patches = self.decoder(tokens).transpose(1, 2)
        recon = F.fold(
            patches,
            output_size=(self.image_size, self.image_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        recon = torch.tanh(recon)
        if orig_size != recon.shape[-2:]:
            recon = F.interpolate(recon, size=orig_size, mode="bilinear", align_corners=False)
        return recon


def _train_uniad(
    loader: DataLoader,
    device: str,
    image_size: int,
    epochs: int,
    lr: float,
    patch_size: int,
    embed_dim: int,
    depth: int,
    heads: int,
) -> UniADLite:
    model = UniADLite(image_size=image_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, heads=heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    for epoch in range(max(1, int(epochs))):
        model.train()
        for batch in tqdm(loader, desc=f"uniad:train:{epoch + 1}", leave=False):
            clean = batch["image"].to(device)
            recon = model(clean)
            loss = F.l1_loss(recon, clean) + 0.5 * F.mse_loss(recon, clean)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def _uniad_scores(model: UniADLite, images: torch.Tensor, device: str) -> tuple[np.ndarray, np.ndarray]:
    x = images.to(device)
    recon = model(x)
    residual = (x - recon).abs().mean(dim=1)
    heatmaps = _normalize_heatmaps(residual)
    heatmaps_np = heatmaps.detach().cpu().numpy().astype(np.float32)
    raw_scores = residual.detach().cpu().numpy().astype(np.float32)
    image_scores = np.asarray([float(hm.max()) for hm in raw_scores], dtype=np.float32)
    return image_scores, heatmaps_np


class DiffusionADLite(nn.Module):
    def __init__(self, base_channels: int = 16, timesteps: int = 50) -> None:
        super().__init__()
        self.model = TinyUNet(in_channels=3, base_channels=base_channels)
        self.timesteps = max(2, int(timesteps))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x_t, t)


def _diffusion_alpha(t: torch.Tensor, timesteps: int) -> torch.Tensor:
    progress = t.float() / max(1, int(timesteps) - 1)
    return (1.0 - 0.85 * progress).clamp(0.05, 1.0).view(-1, 1, 1, 1)


def _train_diffusionad(
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    base_channels: int,
    timesteps: int,
    desc: str = "diffusionad",
) -> DiffusionADLite:
    model = DiffusionADLite(base_channels=base_channels, timesteps=timesteps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-5)
    for epoch in range(max(1, int(epochs))):
        model.train()
        for batch in tqdm(loader, desc=f"{desc}:train:{epoch + 1}", leave=False):
            clean = batch["image"].to(device)
            t = torch.randint(1, model.timesteps, (clean.shape[0],), device=device)
            noise = torch.randn_like(clean)
            alpha = _diffusion_alpha(t, model.timesteps)
            x_t = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def _diffusionad_scores(model: DiffusionADLite, images: torch.Tensor, device: str, score_timestep: int) -> tuple[np.ndarray, np.ndarray]:
    x = images.to(device)
    t_value = max(1, min(int(score_timestep), model.timesteps - 1))
    t = torch.full((x.shape[0],), t_value, device=device, dtype=torch.long)
    noise = torch.randn_like(x)
    alpha = _diffusion_alpha(t, model.timesteps)
    x_t = alpha.sqrt() * x + (1.0 - alpha).sqrt() * noise
    pred_noise = model(x_t, t)
    recon = ((x_t - (1.0 - alpha).sqrt() * pred_noise) / alpha.sqrt()).clamp(-1.0, 1.0)
    residual = (x - recon).abs().mean(dim=1)
    heatmaps = _normalize_heatmaps(residual)
    heatmaps_np = heatmaps.detach().cpu().numpy().astype(np.float32)
    raw_scores = residual.detach().cpu().numpy().astype(np.float32)
    image_scores = np.asarray([float(hm.max()) for hm in raw_scores], dtype=np.float32)
    return image_scores, heatmaps_np


@torch.no_grad()
def _ddad_scores(
    model: DiffusionADLite,
    images: torch.Tensor,
    device: str,
    low_timestep: int,
    high_timestep: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = images.to(device)
    residuals = []
    for value in [low_timestep, high_timestep]:
        t_value = max(1, min(int(value), model.timesteps - 1))
        t = torch.full((x.shape[0],), t_value, device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        alpha = _diffusion_alpha(t, model.timesteps)
        x_t = alpha.sqrt() * x + (1.0 - alpha).sqrt() * noise
        pred_noise = model(x_t, t)
        recon = ((x_t - (1.0 - alpha).sqrt() * pred_noise) / alpha.sqrt()).clamp(-1.0, 1.0)
        residuals.append((x - recon).abs().mean(dim=1))
    residual = torch.stack(residuals, dim=0).mean(dim=0)
    heatmaps = _normalize_heatmaps(residual)
    heatmaps_np = heatmaps.detach().cpu().numpy().astype(np.float32)
    raw_scores = residual.detach().cpu().numpy().astype(np.float32)
    image_scores = np.asarray([float(hm.max()) for hm in raw_scores], dtype=np.float32)
    return image_scores, heatmaps_np


def _rd4ad_loss(teacher_feats: dict[str, torch.Tensor], student_feats: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    losses = []
    for name in names:
        teacher = F.normalize(teacher_feats[name].detach(), dim=1)
        student = F.normalize(student_feats[name], dim=1)
        losses.append(1.0 - (teacher * student).sum(dim=1).mean())
    return torch.stack(losses).mean()


def _train_rd4ad(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    names: list[str],
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
) -> torch.nn.Module:
    teacher.eval()
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=1e-4)
    for epoch in range(max(1, int(epochs))):
        for batch in tqdm(loader, desc=f"rd4ad:train:{epoch + 1}", leave=False):
            x = _preprocess(batch["image"].to(device), device)
            with torch.no_grad():
                teacher_feats = teacher(x)
            student_feats = student(x)
            loss = _rd4ad_loss(teacher_feats, student_feats, names)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    student.eval()
    return student


@torch.no_grad()
def _rd4ad_scores(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    names: list[str],
    images: torch.Tensor,
    device: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = _preprocess(images.to(device), device)
    teacher_feats = teacher(x)
    student_feats = student(x)
    maps = []
    for name in names:
        teacher_feat = F.normalize(teacher_feats[name], dim=1)
        student_feat = F.normalize(student_feats[name], dim=1)
        anomaly = 1.0 - (teacher_feat * student_feat).sum(dim=1, keepdim=True)
        maps.append(F.interpolate(anomaly, size=(image_size, image_size), mode="bilinear", align_corners=False))
    heatmaps = torch.stack(maps, dim=0).mean(dim=0)[:, 0].clamp_min(0.0)
    heatmaps_np = heatmaps.detach().cpu().numpy().astype(np.float32)
    image_scores = np.asarray([float(hm.max()) for hm in heatmaps_np], dtype=np.float32)
    return image_scores, heatmaps_np


def _load_datasets(args: argparse.Namespace, cfg: dict[str, Any], category: str, image_size: int) -> tuple[Any, Any]:
    dataset_name = cfg_first(cfg, ("dataset.name",), "mvtec")
    root = cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD")
    train = build_dataset(dataset_name, root, category, "train", image_size, max_samples=args.train_max_samples)
    test = build_dataset(dataset_name, root, category, "test", image_size, max_samples=args.max_samples)
    return train, test


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg_seed(cfg, args.seed)
    seed_everything(seed)
    category = dataset_category(cfg, args.category)
    image_size = cfg_image_size(cfg, args.image_size)
    device = resolve_device(cfg_device(cfg, args.device))
    run_name = args.run_name or f"{args.method}_{category}"
    run_dir = make_run_dir(cfg, run_name)
    save_run_metadata(run_dir, cfg, args, device, "baselines/run_baseline")
    for sub in ["heatmaps", "masks", "images"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = _load_datasets(args, cfg, category, image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    extractor: torch.nn.Module | None = None
    names: list[str] = []
    if args.method not in {"draem", "uniad", "diffusionad", "ddad"}:
        extractor, names = _build_extractor(args.method, device, args.allow_random_weights)
    if args.method == "patchcore":
        assert extractor is not None
        model_state = _build_patchcore_bank(extractor, names, train_loader, device, args.coreset_ratio, seed)
    elif args.method == "padim":
        assert extractor is not None
        model_state = _build_padim_stats(extractor, names, train_loader, device, args.padim_dim, seed)
    elif args.method == "simplenet":
        assert extractor is not None
        model_state = _train_simplenet(
            extractor,
            names,
            train_loader,
            device,
            args.simplenet_hidden_dim,
            args.simplenet_epochs,
            args.simplenet_lr,
            args.simplenet_noise_std,
            args.simplenet_max_patches,
            seed,
        )
        torch.save(
            {
                "method": args.method,
                "head_state": model_state.state_dict(),
                "hidden_dim": args.simplenet_hidden_dim,
                "noise_std": args.simplenet_noise_std,
                "feature_layers": names,
            },
            run_dir / "baseline_model.pt",
        )
    elif args.method == "draem":
        model_state = _train_draem(
            train_loader,
            device,
            args.draem_epochs,
            args.draem_lr,
            args.draem_base_channels,
            args.draem_recon_weight,
            args.draem_seg_weight,
        )
        torch.save(
            {
                "method": args.method,
                "model_state": model_state.state_dict(),
                "base_channels": args.draem_base_channels,
                "recon_weight": args.draem_recon_weight,
                "seg_weight": args.draem_seg_weight,
            },
            run_dir / "baseline_model.pt",
        )
    elif args.method == "uniad":
        model_state = _train_uniad(
            train_loader,
            device,
            image_size,
            args.uniad_epochs,
            args.uniad_lr,
            args.uniad_patch_size,
            args.uniad_embed_dim,
            args.uniad_depth,
            args.uniad_heads,
        )
        torch.save(
            {
                "method": args.method,
                "model_state": model_state.state_dict(),
                "image_size": image_size,
                "patch_size": args.uniad_patch_size,
                "embed_dim": args.uniad_embed_dim,
                "depth": args.uniad_depth,
                "heads": args.uniad_heads,
            },
            run_dir / "baseline_model.pt",
        )
    elif args.method == "diffusionad":
        model_state = _train_diffusionad(
            train_loader,
            device,
            args.diffusionad_epochs,
            args.diffusionad_lr,
            args.diffusionad_base_channels,
            args.diffusionad_timesteps,
            "diffusionad",
        )
        torch.save(
            {
                "method": args.method,
                "model_state": model_state.state_dict(),
                "base_channels": args.diffusionad_base_channels,
                "timesteps": args.diffusionad_timesteps,
                "score_timestep": args.diffusionad_score_timestep,
            },
            run_dir / "baseline_model.pt",
        )
    elif args.method == "ddad":
        model_state = _train_diffusionad(
            train_loader,
            device,
            args.ddad_epochs,
            args.ddad_lr,
            args.ddad_base_channels,
            args.ddad_timesteps,
            "ddad",
        )
        torch.save(
            {
                "method": args.method,
                "model_state": model_state.state_dict(),
                "base_channels": args.ddad_base_channels,
                "timesteps": args.ddad_timesteps,
                "low_timestep": args.ddad_low_timestep,
                "high_timestep": args.ddad_high_timestep,
            },
            run_dir / "baseline_model.pt",
        )
    else:
        assert extractor is not None
        student, student_names = _build_resnet18_feature_extractor(device)
        if student_names != names:
            raise RuntimeError(f"RD4AD student feature names do not match teacher: {student_names} vs {names}")
        model_state = _train_rd4ad(extractor, student, names, train_loader, device, args.rd4ad_epochs, args.rd4ad_lr)
        torch.save(
            {
                "method": args.method,
                "student_state": model_state.state_dict(),
                "teacher_layers": names,
            },
            run_dir / "baseline_model.pt",
        )

    def score_batch(images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if args.method == "patchcore":
            assert extractor is not None
            return _patchcore_scores(
                extractor, names, model_state, images, device, image_size
            )
        if args.method == "padim":
            assert extractor is not None
            return _padim_scores(
                extractor, names, model_state, images, device, image_size
            )
        if args.method == "simplenet":
            assert extractor is not None
            return _simplenet_scores(
                extractor, names, model_state, images, device, image_size
            )
        if args.method == "draem":
            return _draem_scores(model_state, images, device)
        if args.method == "uniad":
            return _uniad_scores(model_state, images, device)
        if args.method == "diffusionad":
            return _diffusionad_scores(
                model_state,
                images,
                device,
                args.diffusionad_score_timestep,
            )
        if args.method == "ddad":
            return _ddad_scores(
                model_state,
                images,
                device,
                args.ddad_low_timestep,
                args.ddad_high_timestep,
            )
        assert extractor is not None
        return _rd4ad_scores(
            extractor, model_state, names, images, device, image_size
        )

    latency_batch = next(iter(test_loader))["image"][
        : args.latency_batch_size
    ].to(device)
    latency_benchmark = benchmark_callable(
        lambda: score_batch(latency_batch),
        device=device,
        warmups=args.latency_warmups,
        repeats=args.latency_repeats,
        batch_size=len(latency_batch),
    )
    save_json(latency_benchmark, run_dir / "latency_benchmark.json")

    labels: list[int] = []
    image_scores: list[float] = []
    masks: list[np.ndarray] = []
    heatmaps: list[np.ndarray] = []
    paths: list[str] = []
    score_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    offset = 0
    for batch in tqdm(test_loader, desc=f"{args.method}:test", leave=False):
        images = batch["image"].to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        batch_scores, batch_heatmaps = score_batch(images)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000.0
        per_image_latency = elapsed_ms / max(1, len(images))
        batch_masks = batch["mask"][:, 0].numpy().astype(np.uint8)
        batch_labels = batch["label"].numpy().astype(np.uint8)
        for j in range(len(images)):
            idx = offset + j
            stem = f"{idx:05d}"
            path = batch["path"][j]
            labels.append(int(batch_labels[j]))
            image_scores.append(float(batch_scores[j]))
            masks.append(batch_masks[j])
            heatmaps.append(batch_heatmaps[j])
            paths.append(path)
            save_image(run_dir / "heatmaps" / f"{stem}.png", heatmap_to_uint8(batch_heatmaps[j]))
            save_image(run_dir / "masks" / f"{stem}.png", (batch_masks[j] * 255).astype(np.uint8))
            image_dir = run_dir / "images" / stem
            image_dir.mkdir(parents=True, exist_ok=True)
            save_image(image_dir / "input.png", tensor_to_uint8(images[j]))
            save_image(image_dir / "mask.png", (batch_masks[j] * 255).astype(np.uint8))
            np.savez_compressed(image_dir / "residual_heatmap.npz", final=batch_heatmaps[j].astype(np.float32), residual=batch_heatmaps[j].astype(np.float32))
            score_rows.append(
                {
                    "index": idx,
                    "path": path,
                    "label": int(batch_labels[j]),
                    "image_score": float(batch_scores[j]),
                    "latency_ms": per_image_latency,
                    "nfe": 0,
                    "ablation": "baseline",
                    "method": args.method,
                }
            )
            pareto_rows.append({"index": idx, "latency_ms": per_image_latency, "nfe": 0, "image_score": float(batch_scores[j]), "ablation": "baseline", "method": args.method})
            efficiency_rows.append({"index": idx, "latency_ms": per_image_latency, "nfe": 0, "repaired_area_ratio": 0.0, "local_region_ratio": 0.0})
        offset += len(images)

    labels_np = np.asarray(labels, dtype=np.uint8)
    scores_np = np.asarray(image_scores, dtype=np.float32)
    masks_np = np.stack(masks).astype(np.uint8)
    heatmaps_np = np.stack(heatmaps).astype(np.float32)
    metrics = detection_metrics(labels_np, scores_np, masks_np, heatmaps_np)
    eff = efficiency_summary(efficiency_rows)
    eff.update(latency_benchmark)
    metrics.update({f"eff_{k}": v for k, v in eff.items()})
    metrics.update(latency_benchmark)
    spec = BASELINES[args.method]
    metrics["method"] = args.method
    metrics["method_id"] = args.method
    metrics["display_method"] = spec.display_name
    metrics["implementation_variant"] = spec.implementation_variant
    metrics["official_implementation"] = spec.official_implementation
    metrics["source_path"] = spec.source_path
    metrics["source_commit"] = _source_commit()
    metrics["reference_key"] = spec.reference_key
    metrics["category"] = category
    save_json(metrics, run_dir / "metrics.json")
    _metric_csv(metrics, run_dir / "metrics.csv")
    _metric_csv(eff, run_dir / "efficiency.csv")
    np.savez_compressed(
        run_dir / "predictions.npz",
        labels=labels_np,
        image_scores=scores_np,
        masks=masks_np,
        heatmaps=heatmaps_np,
        paths=np.asarray(paths),
        method=np.asarray(args.method),
        display_method=np.asarray(spec.display_name),
        implementation_variant=np.asarray(spec.implementation_variant),
        official_implementation=np.asarray(spec.official_implementation),
    )
    with (run_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "path", "label", "image_score", "latency_ms", "nfe", "ablation", "method"])
        writer.writeheader()
        writer.writerows(score_rows)
    save_json([], run_dir / "roi_budget.json")
    np.save(run_dir / "crv_score_drop.npy", np.asarray([], dtype=np.float32))
    write_pareto(run_dir / "pareto.csv", pareto_rows)
    print(f"Saved {args.method} baseline outputs to {run_dir}")


if __name__ == "__main__":
    main()
