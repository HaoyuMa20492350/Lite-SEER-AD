from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.official_sources import load_official_source_manifest
from seer_ad_v2.data.defect_synthesis import synthesize_anomaly
from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.pixel_threshold_policy import (
    save_pixel_threshold_policy,
    select_synthetic_normal_threshold,
)
from tools.materialize_patchcore_pretrained import selected_categories
from tools.run_official_patchcore import split_ints, stable_seed


SYNTHETIC_MASK_MODES = ("blob", "scratch", "spot", "patch")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BACKBONE_FILENAME = "wide_resnet50_2-95faca4d.pth"
BACKBONE_URL = (
    "https://download.pytorch.org/models/"
    f"{BACKBONE_FILENAME}"
)
SIMPLENET_CORE_FILES = (
    "main.py",
    "simplenet.py",
    "common.py",
    "backbones.py",
    "datasets/mvtec.py",
    "run.sh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the pinned author-official SimpleNet architecture with a "
            "fixed final epoch and export strict evaluation evidence."
        )
    )
    parser.add_argument("--manifest", default="baselines/official_sources.json")
    parser.add_argument(
        "--source-root",
        default="third_party/official_baselines/simplenet",
    )
    parser.add_argument("--dataset-root", default="SEER-AD-dataset/MVTec-AD")
    parser.add_argument(
        "--external-root",
        default="baselines/external_outputs",
    )
    parser.add_argument("--categories", default="bottle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--meta-epochs", type=int, default=40)
    parser.add_argument("--gan-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--synthetic-seeds", default="7,13,23")
    parser.add_argument("--max-normal-images", type=int, default=16)
    parser.add_argument("--synthetic-variants", type=int, default=4)
    parser.add_argument("--max-normal-fpr", type=float, default=0.005)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart-training", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-learn",
        "timm",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_imagenet(images_01: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(images_01)
    std = IMAGENET_STD.to(images_01)
    return (images_01 - mean) / std


def denormalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(images)
    std = IMAGENET_STD.to(images)
    return images * std + mean


def margin_discriminator_loss(
    true_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    true_loss = torch.clamp(-true_scores + margin, min=0)
    fake_loss = torch.clamp(fake_scores + margin, min=0)
    return true_loss.mean() + fake_loss.mean()


def patch_logits_to_outputs(
    logits: torch.Tensor,
    *,
    batch_size: int,
    patch_shape: tuple[int, int],
    output_size: int,
    sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    anomaly = -logits
    anomaly = anomaly.reshape(batch_size, -1)
    image_scores = anomaly.amax(dim=1).detach().cpu().numpy()
    patch_maps = anomaly.reshape(
        batch_size,
        1,
        patch_shape[0],
        patch_shape[1],
    )
    patch_maps = F.interpolate(
        patch_maps,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    heatmaps = patch_maps.detach().cpu().numpy().astype(np.float32)
    heatmaps = np.stack(
        [gaussian_filter(item, sigma=sigma) for item in heatmaps],
        axis=0,
    )
    return image_scores.astype(np.float32), heatmaps.astype(np.float32)


def _load_official_modules(source_root: Path) -> tuple[Any, Any, Any]:
    source_path = str(source_root.resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    import backbones
    from datasets.mvtec import DatasetSplit, MVTecDataset
    from simplenet import SimpleNet

    return backbones, (DatasetSplit, MVTecDataset), SimpleNet


def build_model(
    source_root: Path,
    device: torch.device,
    *,
    meta_epochs: int,
    gan_epochs: int,
) -> Any:
    backbones, _, SimpleNet = _load_official_modules(source_root)
    backbone = backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    backbone.seed = None
    model = SimpleNet(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, 288, 288),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        patchstride=1,
        embedding_size=256,
        meta_epochs=meta_epochs,
        aed_meta_epochs=1,
        gan_epochs=gan_epochs,
        noise_std=0.015,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2,
        dsc_hidden=1024,
        dsc_margin=0.5,
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=0.001,
        pre_proj=1,
        proj_layer_type=0,
    )
    return model


def build_loaders(
    source_root: Path,
    dataset_root: Path,
    category: str,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, Any, Any]:
    _, (DatasetSplit, MVTecDataset), _ = _load_official_modules(source_root)
    train_dataset = MVTecDataset(
        str(dataset_root),
        classname=category,
        resize=329,
        imagesize=288,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
    )
    test_dataset = MVTecDataset(
        str(dataset_root),
        classname=category,
        resize=329,
        imagesize=288,
        split=DatasetSplit.TEST,
    )
    common = {
        "batch_size": max(1, batch_size),
        "num_workers": max(0, num_workers),
        "pin_memory": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, test_loader, train_dataset, test_dataset


@torch.inference_mode()
def cache_frozen_embeddings(
    model: Any,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    cached = None
    offset = 0
    patch_shape = None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        flat, patch_shapes = model._embed(images, evaluation=True)
        current_shape = tuple(int(x) for x in patch_shapes[0])
        patch_count = current_shape[0] * current_shape[1]
        batch_features = flat.reshape(
            len(images),
            patch_count,
            flat.shape[-1],
        ).detach()
        if cached is None:
            cache_device = device
            estimated_bytes = (
                len(loader.dataset)
                * patch_count
                * flat.shape[-1]
                * flat.element_size()
            )
            if device.type == "cuda":
                free_bytes, _ = torch.cuda.mem_get_info(device)
                reserve_bytes = 3 * 1024**3
                if estimated_bytes + reserve_bytes > free_bytes:
                    cache_device = torch.device("cpu")
            cached = torch.empty(
                len(loader.dataset),
                patch_count,
                flat.shape[-1],
                dtype=batch_features.dtype,
                device=cache_device,
            )
            patch_shape = current_shape
        if current_shape != patch_shape:
            raise ValueError("SimpleNet patch shape changed within one dataset")
        cached[offset : offset + len(images)].copy_(batch_features)
        offset += len(images)
    if cached is None or patch_shape is None:
        raise ValueError("Cannot cache an empty SimpleNet training dataset")
    return cached, patch_shape


def _mixed_gaussian_noise(model: Any, shape: torch.Size) -> torch.Tensor:
    noise_indices = torch.randint(
        0,
        model.mix_noise,
        (shape[0],),
        device=model.device,
    )
    one_hot = F.one_hot(
        noise_indices,
        num_classes=model.mix_noise,
    ).to(model.device)
    noise = torch.stack(
        [
            torch.normal(
                0,
                model.noise_std * 1.1**index,
                shape,
                device=model.device,
            )
            for index in range(model.mix_noise)
        ],
        dim=1,
    )
    return (noise * one_hot.unsqueeze(-1)).sum(1)


def train_cached_meta_epoch(
    model: Any,
    cached_embeddings: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    model.forward_modules.eval()
    model.pre_projection.train()
    model.discriminator.train()
    losses = []
    p_true_values = []
    p_fake_values = []
    sample_count = len(cached_embeddings)
    for _ in range(model.gan_epochs):
        order = torch.randperm(sample_count)
        for start in range(0, sample_count, max(1, batch_size)):
            indices = order[start : start + max(1, batch_size)]
            indices = indices.to(cached_embeddings.device)
            base = cached_embeddings[indices]
            if base.device != model.device:
                base = base.to(model.device, non_blocking=True)
            base = base.reshape(-1, base.shape[-1])
            model.dsc_opt.zero_grad(set_to_none=True)
            model.proj_opt.zero_grad(set_to_none=True)
            true_features = model.pre_projection(base)
            fake_features = true_features + _mixed_gaussian_noise(
                model,
                true_features.shape,
            )
            scores = model.discriminator(
                torch.cat([true_features, fake_features], dim=0)
            )
            true_scores = scores[: len(true_features)]
            fake_scores = scores[len(true_features) :]
            loss = margin_discriminator_loss(
                true_scores,
                fake_scores,
                model.dsc_margin,
            )
            loss.backward()
            model.proj_opt.step()
            model.dsc_opt.step()
            losses.append(float(loss.detach().cpu()))
            p_true_values.append(
                float(
                    (true_scores.detach() >= model.dsc_margin)
                    .float()
                    .mean()
                    .cpu()
                )
            )
            p_fake_values.append(
                float(
                    (fake_scores.detach() < -model.dsc_margin)
                    .float()
                    .mean()
                    .cpu()
                )
            )
    return {
        "loss": float(np.mean(losses)),
        "p_true": float(np.mean(p_true_values)),
        "p_fake": float(np.mean(p_fake_values)),
    }


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_training_checkpoint(
    path: Path,
    *,
    meta_epoch: int,
    model: Any,
    history: list[dict[str, float]],
) -> None:
    torch.save(
        {
            "meta_epoch": meta_epoch,
            "discriminator": model.discriminator.state_dict(),
            "pre_projection": model.pre_projection.state_dict(),
            "dsc_optimizer": model.dsc_opt.state_dict(),
            "projection_optimizer": model.proj_opt.state_dict(),
            "history": history,
            "rng_state": _rng_state(),
            "selection_protocol": "fixed_final_epoch_no_test_evaluation",
        },
        path,
    )


def train_model(
    model: Any,
    train_loader: DataLoader,
    checkpoint_path: Path,
    *,
    device: torch.device,
    meta_epochs: int,
    batch_size: int,
    checkpoint_every: int,
    restart_training: bool,
) -> tuple[list[dict[str, float]], int, tuple[int, int], str]:
    history: list[dict[str, float]] = []
    start_epoch = 0
    checkpoint = None
    if checkpoint_path.exists() and not restart_training:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model.discriminator.load_state_dict(
            checkpoint["discriminator"],
            strict=True,
        )
        model.pre_projection.load_state_dict(
            checkpoint["pre_projection"],
            strict=True,
        )
        model.dsc_opt.load_state_dict(checkpoint["dsc_optimizer"])
        model.proj_opt.load_state_dict(checkpoint["projection_optimizer"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["meta_epoch"])

    cached_embeddings, patch_shape = cache_frozen_embeddings(
        model,
        train_loader,
        device=device,
    )
    if checkpoint is not None and "rng_state" in checkpoint:
        _restore_rng_state(checkpoint["rng_state"])
    for meta_epoch in range(start_epoch, meta_epochs):
        stats = train_cached_meta_epoch(
            model,
            cached_embeddings,
            batch_size=batch_size,
        )
        record = {"meta_epoch": meta_epoch + 1, **stats}
        history.append(record)
        if (
            (meta_epoch + 1) % max(1, checkpoint_every) == 0
            or meta_epoch + 1 == meta_epochs
        ):
            save_training_checkpoint(
                checkpoint_path,
                meta_epoch=meta_epoch + 1,
                model=model,
                history=history,
            )
        logging.info(
            "SimpleNet meta-epoch %d/%d loss=%.6f p_true=%.3f p_fake=%.3f",
            meta_epoch + 1,
            meta_epochs,
            stats["loss"],
            stats["p_true"],
            stats["p_fake"],
        )
    embedding_cache_device = str(cached_embeddings.device)
    del cached_embeddings
    return history, start_epoch, patch_shape, embedding_cache_device


@torch.inference_mode()
def score_tensor_batch(
    model: Any,
    images: torch.Tensor,
    *,
    output_size: int = 288,
) -> tuple[np.ndarray, np.ndarray]:
    model.forward_modules.eval()
    model.pre_projection.eval()
    model.discriminator.eval()
    features, patch_shapes = model._embed(
        images.to(model.device, non_blocking=True),
        provide_patch_shapes=True,
        evaluation=True,
    )
    features = model.pre_projection(features)
    logits = model.discriminator(features)
    patch_shape = tuple(int(x) for x in patch_shapes[0])
    return patch_logits_to_outputs(
        logits,
        batch_size=len(images),
        patch_shape=patch_shape,
        output_size=output_size,
    )


@torch.inference_mode()
def score_loader(
    model: Any,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    labels = []
    masks = []
    paths = []
    for batch in loader:
        batch_scores, batch_heatmaps = score_tensor_batch(
            model,
            batch["image"],
        )
        scores.append(batch_scores)
        heatmaps.append(batch_heatmaps)
        labels.append(batch["is_anomaly"].numpy().astype(np.uint8))
        masks.append((batch["mask"][:, 0].numpy() > 0.5).astype(np.uint8))
        paths.extend(str(path) for path in batch["image_path"])
    return (
        np.concatenate(scores).astype(np.float32),
        np.concatenate(heatmaps).astype(np.float32),
        np.concatenate(labels).astype(np.uint8),
        np.concatenate(masks).astype(np.uint8),
        np.asarray(paths),
    )


@torch.inference_mode()
def score_images(
    model: Any,
    images: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    for start in range(0, len(images), max(1, batch_size)):
        batch_scores, batch_heatmaps = score_tensor_batch(
            model,
            images[start : start + max(1, batch_size)],
        )
        scores.append(batch_scores)
        heatmaps.append(batch_heatmaps)
    return (
        np.concatenate(scores).astype(np.float32),
        np.concatenate(heatmaps).astype(np.float32),
    )


def synthetic_evidence(
    model: Any,
    train_dataset: Any,
    category: str,
    seed: int,
    *,
    max_normal_images: int,
    synthetic_variants: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_dataset))[
        : min(len(train_dataset), max(1, max_normal_images))
    ]
    clean_items = []
    synthetic_items = []
    mask_items = []
    paths = []
    variant_ids = []
    for index in indices:
        item = train_dataset[int(index)]
        clean_normalized = item["image"].unsqueeze(0)
        clean_01 = denormalize_imagenet(clean_normalized).clamp(0, 1)[0]
        clean_pm1 = clean_01 * 2.0 - 1.0
        path = str(item["image_path"])
        for variant in range(max(1, synthetic_variants)):
            sample_rng = np.random.RandomState(
                stable_seed(seed, category, path, variant)
            )
            mode = SYNTHETIC_MASK_MODES[variant % len(SYNTHETIC_MASK_MODES)]
            synthetic_pm1, mask = synthesize_anomaly(
                clean_pm1,
                rng=sample_rng,
                mask_mode=mode,
            )
            clean_items.append(clean_normalized[0])
            synthetic_01 = (synthetic_pm1 + 1.0) * 0.5
            synthetic_items.append(
                normalize_imagenet(synthetic_01.unsqueeze(0))[0]
            )
            mask_items.append(mask[0])
            paths.append(path)
            variant_ids.append(variant)
    clean = torch.stack(clean_items)
    synthetic = torch.stack(synthetic_items)
    clean_scores, clean_heatmaps = score_images(
        model,
        clean,
        batch_size=batch_size,
    )
    synthetic_scores, synthetic_heatmaps = score_images(
        model,
        synthetic,
        batch_size=batch_size,
    )
    return {
        "clean_heatmaps": clean_heatmaps,
        "synthetic_heatmaps": synthetic_heatmaps,
        "synthetic_masks": torch.stack(mask_items).numpy().astype(np.uint8),
        "clean_image_scores": clean_scores,
        "synthetic_image_scores": synthetic_scores,
        "paths": np.asarray(paths),
        "variant_ids": np.asarray(variant_ids, dtype=np.int32),
        "seed": np.asarray(seed, dtype=np.int64),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def backbone_checkpoint() -> Path:
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    preferred = checkpoint_dir / BACKBONE_FILENAME
    if preferred.is_file():
        return preferred
    matches = sorted(checkpoint_dir.glob("wide_resnet50_2-*.pth"))
    if not matches:
        raise FileNotFoundError("Torchvision WideResNet50-2 weights are missing")
    return matches[0]


def run_category(
    args: argparse.Namespace,
    source: dict[str, Any],
    source_root: Path,
    dataset_root: Path,
    category: str,
    seeds: list[int],
) -> dict[str, Any]:
    artifact_dir = (
        Path(args.external_root) / "mvtec15" / "simplenet" / category
    )
    prediction_path = artifact_dir / "predictions.npz"
    policy_path = artifact_dir / "pixel_threshold_policy.json"
    provenance_path = artifact_dir / "provenance.json"
    if (
        args.resume
        and prediction_path.exists()
        and policy_path.exists()
        and provenance_path.exists()
    ):
        return {
            "category": category,
            "status": "cached",
            "out": str(artifact_dir),
        }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "training_checkpoint.pth"
    device = torch.device(args.device)
    setup_seed(args.seed)
    model = build_model(
        source_root,
        device,
        meta_epochs=args.meta_epochs,
        gan_epochs=args.gan_epochs,
    )
    train_loader, test_loader, train_dataset, _ = build_loaders(
        source_root,
        dataset_root,
        category,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_started = time.perf_counter()
    history, resumed_from_epoch, patch_shape, embedding_cache_device = train_model(
        model,
        train_loader,
        checkpoint_path,
        device=device,
        meta_epochs=args.meta_epochs,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        restart_training=args.restart_training,
    )
    training_seconds = time.perf_counter() - train_started
    _write_json(
        artifact_dir / "training_history.json",
        {
            "meta_epochs": args.meta_epochs,
            "gan_epochs_per_meta_epoch": args.gan_epochs,
            "resumed_from_meta_epoch": resumed_from_epoch,
            "training_seconds_this_invocation": training_seconds,
            "history": history,
        },
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    scores, heatmaps, labels, masks, paths = score_loader(model, test_loader)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        prediction_path,
        labels=labels,
        image_scores=scores,
        masks=masks,
        heatmaps=heatmaps,
        paths=paths,
    )

    evidence = []
    for seed in seeds:
        payload = synthetic_evidence(
            model,
            train_dataset,
            category,
            seed,
            max_normal_images=args.max_normal_images,
            synthetic_variants=args.synthetic_variants,
            batch_size=args.batch_size,
        )
        evidence_path = (
            artifact_dir / f"synthetic_validation_seed{seed}.npz"
        )
        np.savez_compressed(evidence_path, **payload)
        evidence.append(payload)
    policy = select_synthetic_normal_threshold(
        np.concatenate(
            [item["clean_heatmaps"] for item in evidence],
            axis=0,
        ),
        np.concatenate(
            [item["synthetic_heatmaps"] for item in evidence],
            axis=0,
        ),
        np.concatenate(
            [item["synthetic_masks"] for item in evidence],
            axis=0,
        ),
        max_normal_fpr=args.max_normal_fpr,
    )
    policy.update(
        {
            "method": "simplenet",
            "dataset": "mvtec15",
            "category": category,
            "synthetic_seeds": seeds,
            "source_artifacts": [
                str(
                    artifact_dir
                    / f"synthetic_validation_seed{seed}.npz"
                )
                for seed in seeds
            ],
        }
    )
    save_pixel_threshold_policy(policy, policy_path)
    metrics = detection_metrics(
        labels,
        scores,
        masks,
        heatmaps,
        pixel_threshold=float(policy["threshold"]),
        threshold_protocol=str(policy["protocol"]),
    )
    metrics.update(
        {
            "method": "simplenet",
            "display_method": source["display_name"],
            "dataset": "mvtec15",
            "category": category,
            "official_implementation": True,
            "source_kind": source["source_kind"],
            "source_commit": source["commit"],
            "latency_ms_mean": elapsed * 1000.0 / max(1, len(labels)),
            "prediction_count": int(len(labels)),
        }
    )
    _write_json(artifact_dir / "metrics.json", metrics)

    backbone_path = backbone_checkpoint()
    provenance = {
        "method": "simplenet",
        "dataset": "mvtec15",
        "category": category,
        "source_kind": source["source_kind"],
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "official_implementation": True,
        "execution_command": " ".join(sys.argv),
        "environment": package_versions(),
        "checkpoint_source": (
            "Trained locally from the author-official architecture because "
            "the pinned repository does not release MVTec model checkpoints"
        ),
        "training_checkpoint_path": str(checkpoint_path),
        "training_checkpoint_sha256": sha256_file(checkpoint_path),
        "backbone_checkpoint_path": str(backbone_path),
        "backbone_checkpoint_url": BACKBONE_URL,
        "backbone_checkpoint_sha256": sha256_file(backbone_path),
        "adapter_mode": (
            "author_architecture_fixed_final_epoch_with_frozen_feature_cache"
        ),
        "core_source_sha256": {
            relative: sha256_file(source_root / relative)
            for relative in SIMPLENET_CORE_FILES
        },
        "model_configuration": {
            "backbone": "wideresnet50",
            "layers": ["layer2", "layer3"],
            "resize": 329,
            "input_size": [288, 288],
            "pretrain_embed_dimension": 1536,
            "target_embed_dimension": 1536,
            "patch_size": 3,
            "patch_stride": 1,
            "patch_shape": list(patch_shape),
            "embedding_cache_device": embedding_cache_device,
            "meta_epochs": args.meta_epochs,
            "gan_epochs": args.gan_epochs,
            "noise_std": 0.015,
            "discriminator_hidden": 1024,
            "discriminator_layers": 2,
            "discriminator_margin": 0.5,
            "pre_projection_layers": 1,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "selection_protocol": "fixed_final_epoch_no_test_evaluation",
        "author_code_deviation": (
            "The author train() evaluates the MVTec test set after every "
            "meta-epoch and selects the checkpoint by test AUROC. This "
            "adapter trains the same 40x4 objective to a fixed final epoch "
            "and evaluates the test set once, preventing test-label leakage."
        ),
        "optimization_equivalence": (
            "Frozen deterministic backbone, patch extraction, preprocessing, "
            "and aggregation outputs are cached in float32 before the "
            "trainable pre-projection. The cache uses CUDA when at least "
            "3 GiB remains after allocation and otherwise uses CPU. Gaussian "
            "draws use the same distribution directly on the training "
            "device; no trainable operation or objective is changed."
        ),
        "prediction_export": (
            "raw negative discriminator patch scores, official bilinear "
            "upsampling and Gaussian sigma=4 smoothing, without test-set "
            "min-max normalization"
        ),
        "threshold_protocol": policy["protocol"],
        "uses_real_anomaly_labels_for_threshold": False,
        "uses_real_anomaly_masks_for_threshold": False,
        "uses_test_data_during_training": False,
    }
    _write_json(provenance_path, provenance)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "category": category,
        "status": "completed",
        "out": str(artifact_dir),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    manifest = load_official_source_manifest(args.manifest)
    source = manifest["sources"]["simplenet"]
    source_root = Path(args.source_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    marker = json.loads(
        (source_root / ".lite_seer_source.json").read_text(encoding="utf-8")
    )
    if marker.get("commit") != source["commit"]:
        raise ValueError(
            "Cached SimpleNet source does not match the pinned commit"
        )
    if not dataset_root.exists():
        raise FileNotFoundError(f"MVTec AD dataset is missing: {dataset_root}")
    categories = selected_categories(args.categories)
    seeds = split_ints(args.synthetic_seeds)
    records = []
    failures = []
    for category in categories:
        try:
            records.append(
                run_category(
                    args,
                    source,
                    source_root,
                    dataset_root,
                    category,
                    seeds,
                )
            )
        except Exception as exc:
            failures.append({"category": category, "error": str(exc)})
            logging.exception("SimpleNet failed for %s", category)
    report = {
        "method": "simplenet",
        "source_kind": source["source_kind"],
        "source_commit": source["commit"],
        "selection_protocol": "fixed_final_epoch_no_test_evaluation",
        "categories": categories,
        "completed": len(records),
        "failures": failures,
        "complete": not failures and len(records) == len(categories),
        "records": records,
    }
    report_path = (
        Path(args.external_root)
        / "mvtec15"
        / "simplenet"
        / "run_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
