from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, Wide_ResNet50_2_Weights, resnet18, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from seer_ad_v2.data.hard_negative_mining import ROI


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class FeaturePriorOutput:
    image_scores: np.ndarray
    heatmaps: np.ndarray
    distance_heatmaps: np.ndarray
    cosine_heatmaps: np.ndarray
    raw_heatmaps: np.ndarray
    raw_distance_heatmaps: np.ndarray
    raw_cosine_heatmaps: np.ndarray


def default_layers(backbone: str) -> list[str]:
    if backbone == "wide_resnet50_2":
        return ["layer2", "layer3"]
    return ["layer1", "layer2", "layer3"]


def parse_layers(value: str | list[str] | tuple[str, ...] | None, backbone: str) -> list[str]:
    if value is None:
        return default_layers(backbone)
    if isinstance(value, str):
        layers = [part.strip() for part in value.split(",") if part.strip()]
        return layers or default_layers(backbone)
    return [str(part) for part in value] or default_layers(backbone)


def build_extractor(
    backbone: str,
    layers: list[str],
    device: str,
    allow_random_weights: bool = False,
) -> torch.nn.Module:
    try:
        if backbone == "wide_resnet50_2":
            model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        elif backbone == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported feature backbone: {backbone}")
    except Exception as exc:
        if not allow_random_weights:
            raise RuntimeError(
                "Failed to load ImageNet pretrained weights. Use cached/network weights for paper runs, "
                "or pass --allow-random-weights only for plumbing smoke tests."
            ) from exc
        model = wide_resnet50_2(weights=None) if backbone == "wide_resnet50_2" else resnet18(weights=None)
    extractor = create_feature_extractor(model, return_nodes={name: name for name in layers}).to(device)
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad_(False)
    return extractor


def preprocess(x: torch.Tensor, device: str) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    x01 = (x.clamp(-1, 1) + 1.0) * 0.5
    return (x01 - mean) / std


@torch.no_grad()
def feature_map(extractor: torch.nn.Module, layers: list[str], x: torch.Tensor, device: str) -> torch.Tensor:
    feats = extractor(preprocess(x.to(device), device))
    target_size = feats[layers[0]].shape[-2:]
    maps = []
    for name in layers:
        value = feats[name]
        if value.shape[-2:] != target_size:
            value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
        maps.append(value)
    return torch.cat(maps, dim=1)


def patch_matrix(fmap: torch.Tensor) -> torch.Tensor:
    return fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])


def normalize_heatmaps(maps: torch.Tensor) -> torch.Tensor:
    b = maps.shape[0]
    flat = maps.reshape(b, -1)
    lo = flat.min(dim=1).values.view(b, 1, 1)
    hi = flat.max(dim=1).values.view(b, 1, 1)
    return (maps - lo) / (hi - lo + 1e-8)


def _top1_mean_np(heatmap: np.ndarray) -> float:
    flat = heatmap.reshape(-1)
    k = max(1, int(np.ceil(flat.size * 0.01)))
    return float(np.mean(np.partition(flat, flat.size - k)[-k:]))


def _sample_bank(bank: torch.Tensor, keep: int, seed: int) -> torch.Tensor:
    if keep <= 0 or len(bank) <= keep:
        return bank
    generator = torch.Generator().manual_seed(int(seed))
    index = torch.randperm(len(bank), generator=generator)[:keep]
    return bank[index]


def _reference_patch(
    image: torch.Tensor,
    grid_y: int,
    grid_x: int,
    grid_shape: tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    _, image_h, image_w = image.shape
    grid_h, grid_w = grid_shape
    center_y = int(round((grid_y + 0.5) * image_h / max(1, grid_h)))
    center_x = int(round((grid_x + 0.5) * image_w / max(1, grid_w)))
    half = max(1, int(patch_size) // 2)
    y1, y2 = max(0, center_y - half), min(image_h, center_y + half)
    x1, x2 = max(0, center_x - half), min(image_w, center_x + half)
    crop = image[:, y1:y2, x1:x2].unsqueeze(0)
    return F.interpolate(crop, size=(patch_size, patch_size), mode="bilinear", align_corners=False)[0]


@torch.no_grad()
def build_retrieval_reference_bank(
    extractor: torch.nn.Module,
    layers: list[str],
    loader: DataLoader,
    device: str,
    max_references: int,
    patch_size: int,
    seed: int,
) -> dict[str, Any]:
    if max_references <= 0:
        return {}
    dataset_size = max(1, len(loader.dataset))
    per_image = max(1, int(np.ceil(max_references / dataset_size)))
    generator = torch.Generator().manual_seed(int(seed) + 7919)
    features: list[torch.Tensor] = []
    patches: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    collected = 0
    for batch in tqdm(loader, desc="feature-prior:retrieval", leave=False):
        images = batch["image"]
        fmap = feature_map(extractor, layers, images.to(device), device).detach().cpu()
        _, _, grid_h, grid_w = fmap.shape
        for batch_idx in range(len(images)):
            take = min(per_image, grid_h * grid_w, max_references - collected)
            if take <= 0:
                break
            flat_indices = torch.randperm(grid_h * grid_w, generator=generator)[:take]
            for flat_idx in flat_indices.tolist():
                grid_y, grid_x = divmod(int(flat_idx), grid_w)
                features.append(fmap[batch_idx, :, grid_y, grid_x].float())
                positions.append(
                    torch.tensor(
                        [(grid_y + 0.5) / max(1, grid_h), (grid_x + 0.5) / max(1, grid_w)],
                        dtype=torch.float32,
                    )
                )
                patch = _reference_patch(images[batch_idx], grid_y, grid_x, (grid_h, grid_w), patch_size)
                patches.append((((patch.clamp(-1, 1) + 1.0) * 127.5).round()).to(torch.uint8))
                collected += 1
        if collected >= max_references:
            break
    if not features:
        return {}
    reference_features = torch.stack(features)
    return {
        "retrieval_features": reference_features,
        "retrieval_features_norm": F.normalize(reference_features, dim=1),
        "retrieval_patches": torch.stack(patches),
        "retrieval_positions": torch.stack(positions),
        "retrieval_patch_size": int(patch_size),
        "retrieval_reference_count": int(len(features)),
    }


@torch.no_grad()
def build_patchcore_prior(
    extractor: torch.nn.Module,
    layers: list[str],
    loader: DataLoader,
    device: str,
    coreset_ratio: float,
    max_bank_patches: int,
    seed: int,
) -> dict[str, Any]:
    chunks = []
    for batch in tqdm(loader, desc="feature-prior:patchcore", leave=False):
        fmap = feature_map(extractor, layers, batch["image"].to(device), device)
        chunks.append(patch_matrix(fmap).detach().cpu())
    bank = torch.cat(chunks, dim=0)
    ratio = max(0.0, min(1.0, float(coreset_ratio)))
    keep = len(bank)
    if 0.0 < ratio < 1.0:
        keep = max(1, int(len(bank) * ratio))
    if max_bank_patches > 0:
        keep = min(keep, int(max_bank_patches))
    bank = _sample_bank(bank, keep, seed)
    return {
        "method": "patchcore",
        "bank": bank.float(),
        "bank_norm": F.normalize(bank.float(), dim=1),
    }


@torch.no_grad()
def build_padim_prior(
    extractor: torch.nn.Module,
    layers: list[str],
    loader: DataLoader,
    device: str,
    max_dim: int,
    seed: int,
) -> dict[str, Any]:
    chunks = []
    for batch in tqdm(loader, desc="feature-prior:padim", leave=False):
        chunks.append(feature_map(extractor, layers, batch["image"].to(device), device).detach().cpu())
    features = torch.cat(chunks, dim=0)
    channels = features.shape[1]
    dim = min(int(max_dim), channels)
    generator = torch.Generator().manual_seed(int(seed))
    selected = torch.randperm(channels, generator=generator)[:dim]
    features = features[:, selected].to(device)
    n, d, h, w = features.shape
    flat = features.permute(2, 3, 0, 1).reshape(h * w, n, d)
    mean = flat.mean(dim=1)
    centered = flat - mean.unsqueeze(1)
    cov = centered.transpose(1, 2).matmul(centered) / max(1, n - 1)
    eye = torch.eye(d, device=device).unsqueeze(0)
    inv_cov = torch.linalg.pinv(cov + 0.01 * eye)
    return {
        "method": "padim",
        "selected": selected.cpu(),
        "mean": mean.detach().cpu(),
        "inv_cov": inv_cov.detach().cpu(),
        "height": int(h),
        "width": int(w),
    }


def build_feature_prior(
    extractor: torch.nn.Module,
    layers: list[str],
    loader: DataLoader,
    device: str,
    method: str,
    coreset_ratio: float,
    max_bank_patches: int,
    padim_dim: int,
    seed: int,
    retrieval_max_references: int = 0,
    retrieval_patch_size: int = 64,
) -> dict[str, Any]:
    if method == "patchcore":
        state = build_patchcore_prior(extractor, layers, loader, device, coreset_ratio, max_bank_patches, seed)
    elif method == "padim":
        state = build_padim_prior(extractor, layers, loader, device, padim_dim, seed)
    else:
        raise ValueError(f"Unsupported feature prior method: {method}")
    state.update(
        build_retrieval_reference_bank(
            extractor,
            layers,
            loader,
            device,
            max_references=int(retrieval_max_references),
            patch_size=int(retrieval_patch_size),
            seed=seed,
        )
    )
    return state


def load_feature_prior_components(
    checkpoint: dict[str, Any],
    device: str,
    allow_random_weights: bool = False,
) -> tuple[dict[str, Any], torch.nn.Module, list[str]]:
    backbone = str(checkpoint.get("backbone", "wide_resnet50_2"))
    layers = parse_layers(checkpoint.get("layers"), backbone)
    extractor = build_extractor(backbone, layers, device, allow_random_weights=allow_random_weights)
    return checkpoint, extractor, layers


def _min_distances(patches: torch.Tensor, bank: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    mins = []
    for start in range(0, len(patches), chunk_size):
        chunk = patches[start : start + chunk_size]
        mins.append(torch.cdist(chunk, bank).min(dim=1).values)
    return torch.cat(mins, dim=0)


def _max_cosine_gap(patches: torch.Tensor, bank_norm: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    patches = F.normalize(patches, dim=1)
    gaps = []
    for start in range(0, len(patches), chunk_size):
        chunk = patches[start : start + chunk_size]
        sims = chunk.matmul(bank_norm.T)
        gaps.append(1.0 - sims.max(dim=1).values)
    return torch.cat(gaps, dim=0)


@torch.no_grad()
def retrieve_normal_reference(
    checkpoint: dict[str, Any],
    extractor: torch.nn.Module,
    layers: list[str],
    image: torch.Tensor,
    roi: ROI,
    device: str,
    output_size: tuple[int, int] | None = None,
    spatial_weight: float = 0.25,
) -> tuple[torch.Tensor | None, float, int]:
    state = checkpoint.get("prior_state", checkpoint)
    reference_features = state.get("retrieval_features_norm")
    reference_patches = state.get("retrieval_patches")
    if reference_features is None or reference_patches is None or len(reference_features) == 0:
        return None, 0.0, -1
    if image.ndim == 3:
        image = image.unsqueeze(0)
    fmap = feature_map(extractor, layers, image[:1].to(device), device)
    _, _, grid_h, grid_w = fmap.shape
    image_h, image_w = image.shape[-2:]
    y1 = int(np.floor(roi.y1 * grid_h / max(1, image_h)))
    y2 = int(np.ceil(roi.y2 * grid_h / max(1, image_h)))
    x1 = int(np.floor(roi.x1 * grid_w / max(1, image_w)))
    x2 = int(np.ceil(roi.x2 * grid_w / max(1, image_w)))
    y1, y2 = max(0, min(grid_h - 1, y1)), max(1, min(grid_h, y2))
    x1, x2 = max(0, min(grid_w - 1, x1)), max(1, min(grid_w, x2))
    query = fmap[0, :, y1:y2, x1:x2].mean(dim=(1, 2))
    query = F.normalize(query.float(), dim=0)
    similarities = reference_features.to(device).matmul(query)
    match_scores = similarities
    reference_positions = state.get("retrieval_positions")
    if reference_positions is not None and len(reference_positions) == len(reference_features) and spatial_weight > 0.0:
        query_position = torch.tensor(
            [
                ((roi.y1 + roi.y2) * 0.5) / max(1, image_h),
                ((roi.x1 + roi.x2) * 0.5) / max(1, image_w),
            ],
            dtype=torch.float32,
            device=device,
        )
        spatial_distance = torch.linalg.vector_norm(reference_positions.to(device) - query_position, dim=1)
        match_scores = similarities - float(spatial_weight) * spatial_distance
    index = int(match_scores.argmax().item())
    similarity = float(similarities[index].item())
    patch = reference_patches[index].to(device=device, dtype=torch.float32) / 127.5 - 1.0
    patch = patch.unsqueeze(0)
    if output_size is not None:
        patch = F.interpolate(patch, size=output_size, mode="bilinear", align_corners=False)
    return patch, similarity, index


@torch.no_grad()
def feature_prior_scores(
    checkpoint: dict[str, Any],
    extractor: torch.nn.Module,
    layers: list[str],
    images: torch.Tensor,
    device: str,
    image_size: int,
) -> FeaturePriorOutput:
    state = checkpoint.get("prior_state", checkpoint)
    method = str(state.get("method", checkpoint.get("method", "patchcore")))
    fmap = feature_map(extractor, layers, images.to(device), device)
    b, _, h, w = fmap.shape
    patches = patch_matrix(fmap)
    if method == "patchcore":
        bank = state["bank"].to(device)
        bank_norm = state.get("bank_norm")
        if bank_norm is None:
            bank_norm = F.normalize(bank, dim=1)
        else:
            bank_norm = bank_norm.to(device)
        dist = _min_distances(patches, bank).view(b, h, w)
        cosine = _max_cosine_gap(patches, bank_norm).view(b, h, w).clamp_min(0.0)
    elif method == "padim":
        selected = state["selected"].long().to(device)
        fmap_sel = fmap[:, selected]
        target_h = int(state.get("height", fmap_sel.shape[-2]))
        target_w = int(state.get("width", fmap_sel.shape[-1]))
        if fmap_sel.shape[-2:] != (target_h, target_w):
            fmap_sel = F.interpolate(fmap_sel, size=(target_h, target_w), mode="bilinear", align_corners=False)
        _, d, h, w = fmap_sel.shape
        values = fmap_sel.permute(2, 3, 0, 1).reshape(h * w, b, d)
        mean = state["mean"].to(device)
        inv_cov = state["inv_cov"].to(device)
        diff = values - mean.unsqueeze(1)
        dist2 = torch.einsum("lbd,lde,lbe->lb", diff, inv_cov, diff).clamp_min(0.0)
        dist = torch.sqrt(dist2 + 1e-8).transpose(0, 1).reshape(b, h, w)
        cosine = dist
    else:
        raise ValueError(f"Unsupported feature prior method: {method}")

    raw = dist
    dist_norm = normalize_heatmaps(dist)
    cosine_norm = normalize_heatmaps(cosine)
    fused = normalize_heatmaps(0.75 * dist_norm + 0.25 * cosine_norm)
    heatmaps = F.interpolate(fused.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
    dist_up = F.interpolate(dist_norm.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
    cosine_up = F.interpolate(cosine_norm.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
    raw_up = F.interpolate(raw.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
    raw_dist_up = F.interpolate(dist.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
    raw_cosine_up = F.interpolate(cosine.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False)[:, 0]
    heatmaps_np = heatmaps.detach().cpu().numpy().astype(np.float32)
    raw_heatmaps_np = raw_up.detach().cpu().numpy().astype(np.float32)
    image_scores = np.asarray([_top1_mean_np(hm) for hm in raw_heatmaps_np], dtype=np.float32)
    return FeaturePriorOutput(
        image_scores=image_scores,
        heatmaps=heatmaps_np,
        distance_heatmaps=dist_up.detach().cpu().numpy().astype(np.float32),
        cosine_heatmaps=cosine_up.detach().cpu().numpy().astype(np.float32),
        raw_heatmaps=raw_heatmaps_np,
        raw_distance_heatmaps=raw_dist_up.detach().cpu().numpy().astype(np.float32),
        raw_cosine_heatmaps=raw_cosine_up.detach().cpu().numpy().astype(np.float32),
    )
