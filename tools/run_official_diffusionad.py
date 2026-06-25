from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import logging
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as activation_checkpoint

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
from tools.materialize_diffusionad_foregrounds import (
    MVTEC_FOREGROUND_FOLDER_ID,
)
from tools.materialize_patchcore_pretrained import selected_categories
from tools.run_official_patchcore import split_ints, stable_seed


SYNTHETIC_MASK_MODES = ("blob", "scratch", "spot", "patch")
DIFFUSIONAD_CORE_FILES = (
    "train.py",
    "eval.py",
    "args/args1.json",
    "data/dataset_beta_thresh.py",
    "data/perlin.py",
    "models/DDPM.py",
    "models/Recon_subnetwork.py",
    "models/Seg_subnetwork.py",
)
FOREGROUND_AGGREGATE_SHA256 = (
    "990a97dc2f516c8938decb1e2307c65546b9266bfd56bf7f71b4a9f029906fbd"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the pinned author-official DiffusionAD architecture with "
            "fixed-final-epoch selection and strict evaluation evidence."
        )
    )
    parser.add_argument("--manifest", default="baselines/official_sources.json")
    parser.add_argument(
        "--source-root",
        default="third_party/official_baselines/diffusionad",
    )
    parser.add_argument("--dataset-root", default="SEER-AD-dataset/MVTec-AD")
    parser.add_argument(
        "--dtd-root",
        default=(
            "SEER-AD-dataset/DTD/"
            "Describable-Textures-Dataset-DTD"
        ),
    )
    parser.add_argument(
        "--foreground-report",
        default=(
            "third_party/official_baselines/diffusionad/"
            "foreground_assets/mvtec/materialization_report.json"
        ),
    )
    parser.add_argument(
        "--external-root",
        default="baselines/external_outputs",
    )
    parser.add_argument("--categories", default="bottle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help=(
            "Split each author batch into micro-batches for backward while "
            "keeping one optimizer and scheduler step per author batch."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use CUDA float16 autocast with GradScaler while preserving the "
            "requested full batch size and optimizer-step schedule."
        ),
    )
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help=(
            "Recompute the two reconstruction-UNet forwards during backward "
            "to reduce activation memory without changing batch size."
        ),
    )
    parser.add_argument(
        "--offload-saved-tensors",
        action="store_true",
        help=(
            "Offload tensors saved for backward to pinned CPU memory while "
            "retaining the requested full training batch."
        ),
    )
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
        "opencv-python",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "imgaug",
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


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy(inputs, targets, reduction="none")
        probability = torch.exp(-bce)
        return (
            self.alpha
            * (1.0 - probability) ** self.gamma
            * bce
        ).mean()


def official_image_scores(heatmaps: torch.Tensor) -> torch.Tensor:
    if heatmaps.ndim != 4 or heatmaps.shape[1] != 1:
        raise ValueError(
            "DiffusionAD heatmaps must have shape Bx1xHxW, got "
            f"{tuple(heatmaps.shape)}"
        )
    flattened = heatmaps.flatten(1)
    topk = min(50, flattened.shape[1])
    return flattened.topk(topk, dim=1, largest=True).values.mean(dim=1)


def load_author_args(
    source_root: Path,
    dataset_root: Path,
    dtd_root: Path,
) -> defaultdict[str, Any]:
    payload = json.loads(
        (source_root / "args" / "args1.json").read_text(encoding="utf-8")
    )
    payload["mvtec_root_path"] = str(dataset_root)
    payload["anomaly_source_path"] = str(dtd_root)
    payload["arg_num"] = "1"
    return defaultdict(str, payload)


def _load_official_modules(source_root: Path) -> tuple[Any, Any, Any, Any]:
    source_path = str(source_root.resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from data.dataset_beta_thresh import MVTecTestDataset, MVTecTrainDataset
    from models.DDPM import GaussianDiffusionModel, get_beta_schedule
    from models.Recon_subnetwork import UNetModel
    from models.Seg_subnetwork import SegmentationSubNetwork

    return (
        (MVTecTrainDataset, MVTecTestDataset),
        (GaussianDiffusionModel, get_beta_schedule),
        UNetModel,
        SegmentationSubNetwork,
    )


def build_models(
    source_root: Path,
    args: defaultdict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, Any]:
    (
        _,
        (GaussianDiffusionModel, get_beta_schedule),
        UNetModel,
        SegmentationSubNetwork,
    ) = _load_official_modules(source_root)
    unet = UNetModel(
        args["img_size"][0],
        args["base_channels"],
        channel_mults=args["channel_mults"],
        dropout=args["dropout"],
        n_heads=args["num_heads"],
        n_head_channels=args["num_head_channels"],
        in_channels=args["channels"],
    ).to(device)
    segmentation = SegmentationSubNetwork(
        in_channels=6,
        out_channels=1,
    ).to(device)
    betas = get_beta_schedule(args["T"], args["beta_schedule"])
    diffusion = GaussianDiffusionModel(
        args["img_size"],
        betas,
        loss_weight=args["loss_weight"],
        loss_type=args["loss-type"],
        noise=args["noise_fn"],
        img_channels=args["channels"],
    )
    return unet, segmentation, diffusion


def build_loaders(
    source_root: Path,
    args: defaultdict[str, Any],
    dataset_root: Path,
    category: str,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, Any, Any]:
    (MVTecTrainDataset, MVTecTestDataset), _, _, _ = (
        _load_official_modules(source_root)
    )
    category_root = dataset_root / category
    train_dataset = MVTecTrainDataset(
        str(category_root),
        category,
        img_size=args["img_size"],
        args=args,
    )
    test_dataset = MVTecTestDataset(
        str(category_root),
        category,
        img_size=args["img_size"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, batch_size),
        shuffle=True,
        num_workers=max(0, num_workers),
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=True,
    )
    return train_loader, test_loader, train_dataset, test_dataset


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
    epoch: int,
    unet: torch.nn.Module,
    segmentation: torch.nn.Module,
    optimizer_unet: torch.optim.Optimizer,
    optimizer_segmentation: torch.optim.Optimizer,
    scheduler_segmentation: Any,
    scaler: torch.amp.GradScaler,
    history: list[dict[str, float]],
    args: defaultdict[str, Any],
    amp_enabled: bool,
    activation_checkpointing: bool,
    offload_saved_tensors: bool,
    micro_batch_size: int | None,
    max_train_batches: int | None,
    training_seed: int,
) -> None:
    payload = {
        "epoch": epoch,
        "unet_model_state_dict": unet.state_dict(),
        "seg_model_state_dict": segmentation.state_dict(),
        "optimizer_unet": optimizer_unet.state_dict(),
        "optimizer_segmentation": optimizer_segmentation.state_dict(),
        "scheduler_segmentation": scheduler_segmentation.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "history": history,
        "rng_state": _rng_state(),
        "args": dict(args),
        "amp_enabled": amp_enabled,
        "activation_checkpointing": activation_checkpointing,
        "offload_saved_tensors": offload_saved_tensors,
        "micro_batch_size": micro_batch_size,
        "max_train_batches": max_train_batches,
        "training_seed": training_seed,
        "selection_protocol": "fixed_final_epoch_no_test_evaluation",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_models(
    unet: torch.nn.Module,
    segmentation: torch.nn.Module,
    diffusion: Any,
    loader: DataLoader,
    checkpoint_path: Path,
    args: defaultdict[str, Any],
    *,
    device: torch.device,
    epochs: int,
    checkpoint_every: int,
    max_train_batches: int | None,
    restart_training: bool,
    amp_enabled: bool,
    activation_checkpointing: bool,
    offload_saved_tensors: bool,
    micro_batch_size: int | None,
    training_seed: int,
) -> tuple[list[dict[str, float]], int]:
    optimizer_unet = optim.Adam(
        unet.parameters(),
        lr=args["diffusion_lr"],
        weight_decay=args["weight_decay"],
    )
    optimizer_segmentation = optim.Adam(
        segmentation.parameters(),
        lr=args["seg_lr"],
        weight_decay=args["weight_decay"],
    )
    scheduler_segmentation = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_segmentation,
        T_max=10,
        eta_min=0,
    )
    focal = BinaryFocalLoss().to(device)
    smooth_l1 = nn.SmoothL1Loss().to(device)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
    )
    history: list[dict[str, float]] = []
    start_epoch = 0
    last_checkpoint_epoch = 0
    if checkpoint_path.exists() and not restart_training:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        unet.load_state_dict(
            checkpoint["unet_model_state_dict"],
            strict=True,
        )
        segmentation.load_state_dict(
            checkpoint["seg_model_state_dict"],
            strict=True,
        )
        optimizer_unet.load_state_dict(checkpoint["optimizer_unet"])
        optimizer_segmentation.load_state_dict(
            checkpoint["optimizer_segmentation"]
        )
        scheduler_segmentation.load_state_dict(
            checkpoint["scheduler_segmentation"]
        )
        checkpoint_amp = bool(checkpoint.get("amp_enabled", False))
        if checkpoint_amp != amp_enabled:
            raise ValueError(
                "DiffusionAD checkpoint precision mode does not match the "
                f"request: checkpoint amp={checkpoint_amp}, request "
                f"amp={amp_enabled}"
            )
        checkpoint_activation = bool(
            checkpoint.get("activation_checkpointing", False)
        )
        if checkpoint_activation != activation_checkpointing:
            raise ValueError(
                "DiffusionAD checkpoint activation-checkpointing mode does "
                "not match the request: checkpoint "
                f"activation_checkpointing={checkpoint_activation}, request "
                f"activation_checkpointing={activation_checkpointing}"
            )
        checkpoint_offload = bool(
            checkpoint.get("offload_saved_tensors", False)
        )
        if checkpoint_offload != offload_saved_tensors:
            raise ValueError(
                "DiffusionAD checkpoint saved-tensor offload mode does not "
                "match the request: checkpoint "
                f"offload_saved_tensors={checkpoint_offload}, request "
                f"offload_saved_tensors={offload_saved_tensors}"
            )
        checkpoint_micro_batch = checkpoint.get("micro_batch_size")
        if checkpoint_micro_batch != micro_batch_size:
            raise ValueError(
                "DiffusionAD checkpoint micro-batch mode does not match the "
                f"request: checkpoint micro_batch_size="
                f"{checkpoint_micro_batch}, request "
                f"micro_batch_size={micro_batch_size}"
            )
        checkpoint_max_batches = checkpoint.get("max_train_batches")
        if checkpoint_max_batches != max_train_batches:
            raise ValueError(
                "DiffusionAD checkpoint max-train-batches setting does not "
                "match the request: checkpoint max_train_batches="
                f"{checkpoint_max_batches}, request "
                f"max_train_batches={max_train_batches}"
            )
        checkpoint_batch_size = int(
            checkpoint.get("args", {}).get("Batch_Size", -1)
        )
        if checkpoint_batch_size != int(args["Batch_Size"]):
            raise ValueError(
                "DiffusionAD checkpoint author batch size does not match the "
                f"request: {checkpoint_batch_size} vs {args['Batch_Size']}"
            )
        checkpoint_seed = int(
            checkpoint.get("training_seed", training_seed)
        )
        if checkpoint_seed != training_seed:
            raise ValueError(
                "DiffusionAD checkpoint seed does not match the request: "
                f"{checkpoint_seed} vs {training_seed}"
            )
        if "grad_scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["grad_scaler"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"])
        last_checkpoint_epoch = start_epoch
        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])

    for epoch in range(start_epoch, epochs):
        unet.train()
        segmentation.train()
        losses = []
        noise_losses = []
        focal_losses = []
        smooth_losses = []
        for batch_index, sample in enumerate(loader):
            if (
                max_train_batches is not None
                and batch_index >= max_train_batches
            ):
                break
            augmented_cpu = sample["augmented_image"].float()
            anomaly_mask_cpu = sample["anomaly_mask"].float()
            anomaly_label_cpu = sample["has_anomaly"].float().reshape(-1)
            full_batch_size = len(augmented_cpu)
            requested_micro_batch = (
                full_batch_size
                if micro_batch_size is None
                else int(micro_batch_size)
            )
            if requested_micro_batch < 1:
                raise ValueError("micro_batch_size must be positive")
            if full_batch_size % requested_micro_batch != 0:
                raise ValueError(
                    "DiffusionAD micro_batch_size must divide each full "
                    f"batch exactly: {requested_micro_batch} does not divide "
                    f"{full_batch_size}"
                )
            total_normal = int(
                torch.count_nonzero(anomaly_label_cpu == 0).item()
            )
            optimizer_unet.zero_grad(set_to_none=True)
            optimizer_segmentation.zero_grad(set_to_none=True)
            batch_loss = 0.0
            batch_noise_loss = 0.0
            batch_focal_loss = 0.0
            batch_smooth_loss = 0.0
            for micro_start in range(
                0,
                full_batch_size,
                requested_micro_batch,
            ):
                micro_end = micro_start + requested_micro_batch
                augmented = augmented_cpu[micro_start:micro_end].to(
                    device,
                    non_blocking=True,
                )
                anomaly_mask = anomaly_mask_cpu[micro_start:micro_end].to(
                    device,
                    non_blocking=True,
                )
                anomaly_label = anomaly_label_cpu[
                    micro_start:micro_end
                ].to(
                    device,
                    non_blocking=True,
                )
                micro_normal = int(
                    torch.count_nonzero(anomaly_label == 0).item()
                )
                sample_weight = len(augmented) / full_batch_size
                normal_weight = (
                    micro_normal / total_normal
                    if total_normal > 0
                    else 0.0
                )
                saved_tensor_context = (
                    torch.autograd.graph.save_on_cpu(pin_memory=True)
                    if offload_saved_tensors
                    else contextlib.nullcontext()
                )
                with saved_tensor_context:
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=amp_enabled,
                    ):
                        reconstruction_model = unet
                        if activation_checkpointing:
                            reconstruction_model = (
                                lambda image, timestep: activation_checkpoint(
                                    unet,
                                    image,
                                    timestep,
                                    use_reentrant=False,
                                )
                            )
                        noise_loss, reconstruction, _, _, _ = (
                            diffusion.norm_guided_one_step_denoising(
                                reconstruction_model,
                                augmented,
                                anomaly_label,
                                args,
                            )
                        )
                        predicted_mask = segmentation(
                            torch.cat((augmented, reconstruction), dim=1)
                        )
                    with torch.amp.autocast(
                        device_type=device.type,
                        enabled=False,
                    ):
                        focal_loss = focal(
                            predicted_mask.float(),
                            anomaly_mask.float(),
                        )
                        smooth_loss = smooth_l1(
                            predicted_mask.float(),
                            anomaly_mask.float(),
                        )
                        weighted_noise = noise_loss.float() * normal_weight
                        weighted_focal = (
                            5.0 * focal_loss * sample_weight
                        )
                        weighted_smooth = smooth_loss * sample_weight
                        loss = (
                            weighted_noise
                            + weighted_focal
                            + weighted_smooth
                        )
                scaler.scale(loss).backward()
                batch_loss += float(loss.detach().cpu())
                batch_noise_loss += float(weighted_noise.detach().cpu())
                batch_focal_loss += float(weighted_focal.detach().cpu())
                batch_smooth_loss += float(weighted_smooth.detach().cpu())
            scaler.step(optimizer_unet)
            scaler.step(optimizer_segmentation)
            scaler.update()
            scheduler_segmentation.step()
            losses.append(batch_loss)
            noise_losses.append(batch_noise_loss)
            focal_losses.append(batch_focal_loss)
            smooth_losses.append(batch_smooth_loss)
        if not losses:
            raise ValueError("DiffusionAD training loader produced no batches")
        record = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "noise_loss": float(np.mean(noise_losses)),
            "weighted_focal_loss": float(np.mean(focal_losses)),
            "smooth_l1_loss": float(np.mean(smooth_losses)),
            "batches": len(losses),
        }
        history.append(record)
        should_checkpoint = (
            (epoch + 1) % max(1, checkpoint_every) == 0
            or epoch + 1 == epochs
        )
        if should_checkpoint:
            save_training_checkpoint(
                checkpoint_path,
                epoch=epoch + 1,
                unet=unet,
                segmentation=segmentation,
                optimizer_unet=optimizer_unet,
                optimizer_segmentation=optimizer_segmentation,
                scheduler_segmentation=scheduler_segmentation,
                scaler=scaler,
                history=history,
                args=args,
                amp_enabled=amp_enabled,
                activation_checkpointing=activation_checkpointing,
                offload_saved_tensors=offload_saved_tensors,
                micro_batch_size=micro_batch_size,
                max_train_batches=max_train_batches,
                training_seed=training_seed,
            )
            last_checkpoint_epoch = epoch + 1
        progress = {
            "last_completed_epoch": epoch + 1,
            "target_epochs": epochs,
            "last_checkpoint_epoch": last_checkpoint_epoch,
            "checkpoint_every": checkpoint_every,
            "last_record": record,
            "amp_enabled": amp_enabled,
            "activation_checkpointing": activation_checkpointing,
            "offload_saved_tensors": offload_saved_tensors,
            "micro_batch_size": micro_batch_size,
            "max_train_batches": max_train_batches,
            "training_seed": training_seed,
        }
        _write_json(
            checkpoint_path.with_name("training_progress.json"),
            progress,
        )
        logging.info(
            "DiffusionAD epoch %d/%d loss=%.6f batches=%d",
            epoch + 1,
            epochs,
            record["loss"],
            len(losses),
        )
    return history, start_epoch


@torch.inference_mode()
def score_tensor_batch(
    unet: torch.nn.Module,
    segmentation: torch.nn.Module,
    diffusion: Any,
    images: torch.Tensor,
    args: defaultdict[str, Any],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    unet.eval()
    segmentation.eval()
    images = images.float().to(device, non_blocking=True)
    normal_t = torch.full(
        (len(images),),
        int(args["eval_normal_t"]),
        device=device,
        dtype=torch.long,
    )
    noisier_t = torch.full(
        (len(images),),
        int(args["eval_noisier_t"]),
        device=device,
        dtype=torch.long,
    )
    (
        _,
        reconstruction,
        _,
        _,
        _,
        _,
        _,
    ) = diffusion.norm_guided_one_step_denoising_eval(
        unet,
        images,
        normal_t,
        noisier_t,
        args,
    )
    heatmaps = segmentation(torch.cat((images, reconstruction), dim=1))
    scores = official_image_scores(heatmaps)
    return (
        scores.detach().cpu().numpy().astype(np.float32),
        heatmaps[:, 0].detach().cpu().numpy().astype(np.float32),
    )


@torch.inference_mode()
def score_loader(
    unet: torch.nn.Module,
    segmentation: torch.nn.Module,
    diffusion: Any,
    loader: DataLoader,
    args: defaultdict[str, Any],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    labels = []
    masks = []
    paths = []
    for sample in loader:
        batch_scores, batch_heatmaps = score_tensor_batch(
            unet,
            segmentation,
            diffusion,
            sample["image"],
            args,
            device=device,
        )
        scores.append(batch_scores)
        heatmaps.append(batch_heatmaps)
        labels.append(
            sample["has_anomaly"].reshape(-1).numpy().astype(np.uint8)
        )
        masks.append(
            (sample["mask"][:, 0].numpy() > 0.5).astype(np.uint8)
        )
        paths.extend(str(path) for path in sample["file_name"])
    return (
        np.concatenate(scores),
        np.concatenate(heatmaps),
        np.concatenate(labels),
        np.concatenate(masks),
        np.asarray(paths),
    )


def read_rgb_image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1))


@torch.inference_mode()
def score_images(
    unet: torch.nn.Module,
    segmentation: torch.nn.Module,
    diffusion: Any,
    images: torch.Tensor,
    args: defaultdict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    for start in range(0, len(images), max(1, batch_size)):
        batch_scores, batch_heatmaps = score_tensor_batch(
            unet,
            segmentation,
            diffusion,
            images[start : start + max(1, batch_size)],
            args,
            device=device,
        )
        scores.append(batch_scores)
        heatmaps.append(batch_heatmaps)
    return np.concatenate(scores), np.concatenate(heatmaps)


def synthetic_evidence(
    unet: torch.nn.Module,
    segmentation: torch.nn.Module,
    diffusion: Any,
    train_paths: list[Path],
    category: str,
    seed: int,
    args: defaultdict[str, Any],
    *,
    device: torch.device,
    max_normal_images: int,
    synthetic_variants: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_paths))[
        : min(len(train_paths), max(1, max_normal_images))
    ]
    clean_items = []
    synthetic_items = []
    mask_items = []
    paths = []
    variant_ids = []
    for index in indices:
        path = train_paths[int(index)]
        clean_01 = read_rgb_image(path)
        clean_pm1 = clean_01 * 2.0 - 1.0
        for variant in range(max(1, synthetic_variants)):
            sample_rng = np.random.RandomState(
                stable_seed(seed, category, str(path), variant)
            )
            mode = SYNTHETIC_MASK_MODES[variant % len(SYNTHETIC_MASK_MODES)]
            synthetic_pm1, mask = synthesize_anomaly(
                clean_pm1,
                rng=sample_rng,
                mask_mode=mode,
            )
            clean_items.append(clean_01)
            synthetic_items.append((synthetic_pm1 + 1.0) * 0.5)
            mask_items.append(mask[0])
            paths.append(str(path))
            variant_ids.append(variant)
    setup_seed(seed)
    clean_scores, clean_heatmaps = score_images(
        unet,
        segmentation,
        diffusion,
        torch.stack(clean_items),
        args,
        device=device,
        batch_size=batch_size,
    )
    setup_seed(seed + 1)
    synthetic_scores, synthetic_heatmaps = score_images(
        unet,
        segmentation,
        diffusion,
        torch.stack(synthetic_items),
        args,
        device=device,
        batch_size=batch_size,
    )
    return {
        "clean_heatmaps": clean_heatmaps.astype(np.float32),
        "synthetic_heatmaps": synthetic_heatmaps.astype(np.float32),
        "synthetic_masks": torch.stack(mask_items).numpy().astype(np.uint8),
        "clean_image_scores": clean_scores.astype(np.float32),
        "synthetic_image_scores": synthetic_scores.astype(np.float32),
        "paths": np.asarray(paths),
        "variant_ids": np.asarray(variant_ids, dtype=np.int32),
        "seed": np.asarray(seed, dtype=np.int64),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def completed_output_matches_request(
    provenance_path: Path,
    *,
    epochs: int,
    max_train_batches: int | None,
    amp_enabled: bool,
    activation_checkpointing: bool,
    offload_saved_tensors: bool,
    micro_batch_size: int | None,
    batch_size: int,
    seed: int,
    synthetic_seeds: list[int],
    max_normal_images: int,
    synthetic_variants: int,
    max_normal_fpr: float,
) -> bool:
    if not provenance_path.exists():
        return False
    try:
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    configuration = provenance.get("model_configuration", {})
    if not isinstance(configuration, dict):
        return False
    expected_paper_eligible = epochs == 3000 and max_train_batches is None
    return (
        int(configuration.get("epochs", -1)) == int(epochs)
        and configuration.get("max_train_batches") == max_train_batches
        and bool(configuration.get("amp_enabled", False)) == amp_enabled
        and bool(
            configuration.get("activation_checkpointing", False)
        )
        == activation_checkpointing
        and bool(configuration.get("offload_saved_tensors", False))
        == offload_saved_tensors
        and configuration.get("micro_batch_size") == micro_batch_size
        and int(configuration.get("batch_size", -1)) == batch_size
        and int(configuration.get("seed", -1)) == seed
        and provenance.get("synthetic_seeds") == synthetic_seeds
        and int(provenance.get("max_normal_images", -1))
        == max_normal_images
        and int(provenance.get("synthetic_variants", -1))
        == synthetic_variants
        and math.isclose(
            float(provenance.get("max_normal_fpr", -1.0)),
            max_normal_fpr,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and bool(provenance.get("paper_eligible_full_training"))
        == expected_paper_eligible
    )


def run_category(
    cli_args: argparse.Namespace,
    source: dict[str, Any],
    source_root: Path,
    dataset_root: Path,
    dtd_root: Path,
    foreground_report: dict[str, Any],
    category: str,
    seeds: list[int],
) -> dict[str, Any]:
    artifact_dir = (
        Path(cli_args.external_root)
        / "mvtec15"
        / "diffusionad"
        / category
    )
    prediction_path = artifact_dir / "predictions.npz"
    policy_path = artifact_dir / "pixel_threshold_policy.json"
    provenance_path = artifact_dir / "provenance.json"
    if (
        cli_args.resume
        and prediction_path.exists()
        and policy_path.exists()
        and provenance_path.exists()
        and completed_output_matches_request(
            provenance_path,
            epochs=cli_args.epochs,
            max_train_batches=cli_args.max_train_batches,
            amp_enabled=bool(cli_args.amp),
            activation_checkpointing=bool(
                cli_args.activation_checkpointing
            ),
            offload_saved_tensors=bool(
                cli_args.offload_saved_tensors
            ),
            micro_batch_size=cli_args.micro_batch_size,
            batch_size=cli_args.batch_size,
            seed=cli_args.seed,
            synthetic_seeds=seeds,
            max_normal_images=cli_args.max_normal_images,
            synthetic_variants=cli_args.synthetic_variants,
            max_normal_fpr=cli_args.max_normal_fpr,
        )
    ):
        return {
            "category": category,
            "status": "cached",
            "out": str(artifact_dir),
        }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "training_checkpoint.pth"
    device = torch.device(cli_args.device)
    setup_seed(cli_args.seed)
    args = load_author_args(source_root, dataset_root, dtd_root)
    args["EPOCHS"] = cli_args.epochs
    args["Batch_Size"] = cli_args.batch_size
    unet, segmentation, diffusion = build_models(
        source_root,
        args,
        device,
    )
    train_loader, test_loader, _, _ = build_loaders(
        source_root,
        args,
        dataset_root,
        category,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
    )
    train_started = time.perf_counter()
    history, resumed_from_epoch = train_models(
        unet,
        segmentation,
        diffusion,
        train_loader,
        checkpoint_path,
        args,
        device=device,
        epochs=cli_args.epochs,
        checkpoint_every=cli_args.checkpoint_every,
        max_train_batches=cli_args.max_train_batches,
        restart_training=cli_args.restart_training,
        amp_enabled=bool(cli_args.amp),
        activation_checkpointing=bool(
            cli_args.activation_checkpointing
        ),
        offload_saved_tensors=bool(
            cli_args.offload_saved_tensors
        ),
        micro_batch_size=cli_args.micro_batch_size,
        training_seed=cli_args.seed,
    )
    training_seconds = time.perf_counter() - train_started
    _write_json(
        artifact_dir / "training_history.json",
        {
            "epochs": cli_args.epochs,
            "resumed_from_epoch": resumed_from_epoch,
            "max_train_batches": cli_args.max_train_batches,
            "amp_enabled": bool(cli_args.amp),
            "activation_checkpointing": bool(
                cli_args.activation_checkpointing
            ),
            "offload_saved_tensors": bool(
                cli_args.offload_saved_tensors
            ),
            "micro_batch_size": cli_args.micro_batch_size,
            "training_seconds_this_invocation": training_seconds,
            "history": history,
        },
    )

    setup_seed(cli_args.seed + 100000)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    scores, heatmaps, labels, masks, paths = score_loader(
        unet,
        segmentation,
        diffusion,
        test_loader,
        args,
        device=device,
    )
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

    train_paths = sorted(
        (dataset_root / category / "train" / "good").glob("*.png")
    )
    evidence = []
    for seed in seeds:
        payload = synthetic_evidence(
            unet,
            segmentation,
            diffusion,
            train_paths,
            category,
            seed,
            args,
            device=device,
            max_normal_images=cli_args.max_normal_images,
            synthetic_variants=cli_args.synthetic_variants,
            batch_size=1,
        )
        np.savez_compressed(
            artifact_dir / f"synthetic_validation_seed{seed}.npz",
            **payload,
        )
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
        max_normal_fpr=cli_args.max_normal_fpr,
    )
    policy.update(
        {
            "method": "diffusionad",
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
    paper_eligible = (
        cli_args.epochs == 3000
        and cli_args.max_train_batches is None
    )
    metrics.update(
        {
            "method": "diffusionad",
            "display_method": source["display_name"],
            "dataset": "mvtec15",
            "category": category,
            "official_implementation": True,
            "source_kind": source["source_kind"],
            "source_commit": source["commit"],
            "latency_ms_mean": elapsed * 1000.0 / max(1, len(labels)),
            "prediction_count": int(len(labels)),
            "paper_eligible_full_training": paper_eligible,
        }
    )
    _write_json(artifact_dir / "metrics.json", metrics)

    provenance = {
        "method": "diffusionad",
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
            "the repository does not release MVTec checkpoints"
        ),
        "training_checkpoint_path": str(checkpoint_path),
        "training_checkpoint_sha256": sha256_file(checkpoint_path),
        "adapter_mode": "author_architecture_fixed_final_epoch",
        "core_source_sha256": {
            relative: sha256_file(source_root / relative)
            for relative in DIFFUSIONAD_CORE_FILES
        },
        "foreground_folder_id": MVTEC_FOREGROUND_FOLDER_ID,
        "foreground_aggregate_sha256": FOREGROUND_AGGREGATE_SHA256,
        "foreground_materialization_report": foreground_report,
        "model_configuration": {
            **dict(args),
            "epochs": cli_args.epochs,
            "batch_size": cli_args.batch_size,
            "max_train_batches": cli_args.max_train_batches,
            "amp_enabled": bool(cli_args.amp),
            "activation_checkpointing": bool(
                cli_args.activation_checkpointing
            ),
            "offload_saved_tensors": bool(
                cli_args.offload_saved_tensors
            ),
            "micro_batch_size": cli_args.micro_batch_size,
            "seed": cli_args.seed,
        },
        "selection_protocol": "fixed_final_epoch_no_test_evaluation",
        "author_code_deviation": (
            "The author train.py evaluates the MVTec test set every 50 "
            "epochs and selects params-best.pt using test AUROC. This adapter "
            "does not access the test set until fixed-final-epoch training "
            "has completed. When --amp is enabled, CUDA float16 autocast and "
            "GradScaler are used as a disclosed memory-execution adaptation; "
            "the author batch size and optimizer-step schedule are retained. "
            "When --activation-checkpointing is enabled, reconstruction-UNet "
            "activations are recomputed during backward without changing the "
            "forward model, losses, batch, or optimizer schedule. When "
            "--offload-saved-tensors is enabled, tensors saved for backward "
            "are stored in pinned CPU memory to retain the full author batch."
            " When --micro-batch-size is lower than --batch-size, gradients "
            "are accumulated with one optimizer/scheduler step per author "
            "batch and loss terms are weighted to full-batch means; "
            "BatchNorm statistics use the disclosed micro-batch size."
        ),
        "mixed_precision_training": bool(cli_args.amp),
        "activation_checkpointing": bool(
            cli_args.activation_checkpointing
        ),
        "offload_saved_tensors": bool(
            cli_args.offload_saved_tensors
        ),
        "micro_batch_size": cli_args.micro_batch_size,
        "exact_author_batchnorm": (
            cli_args.micro_batch_size is None
            or cli_args.micro_batch_size == cli_args.batch_size
        ),
        "synthetic_seeds": seeds,
        "max_normal_images": cli_args.max_normal_images,
        "synthetic_variants": cli_args.synthetic_variants,
        "max_normal_fpr": cli_args.max_normal_fpr,
        "prediction_export": (
            "raw segmentation-subnetwork probabilities; image score is the "
            "author top-50 pixel mean; no test-set min-max normalization"
        ),
        "threshold_protocol": policy["protocol"],
        "uses_real_anomaly_labels_for_threshold": False,
        "uses_real_anomaly_masks_for_threshold": False,
        "uses_test_data_during_training": False,
        "paper_eligible_full_training": paper_eligible,
    }
    _write_json(provenance_path, provenance)
    del unet, segmentation, diffusion
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
    cli_args = parse_args()
    logging.basicConfig(level=logging.INFO)
    manifest = load_official_source_manifest(cli_args.manifest)
    source = manifest["sources"]["diffusionad"]
    source_root = Path(cli_args.source_root).resolve()
    dataset_root = Path(cli_args.dataset_root).resolve()
    dtd_root = Path(cli_args.dtd_root).resolve()
    foreground_report_path = Path(cli_args.foreground_report).resolve()
    marker = json.loads(
        (source_root / ".lite_seer_source.json").read_text(encoding="utf-8")
    )
    if marker.get("commit") != source["commit"]:
        raise ValueError(
            "Cached DiffusionAD source does not match the pinned commit"
        )
    if not dataset_root.exists():
        raise FileNotFoundError(f"MVTec AD dataset is missing: {dataset_root}")
    if not list((dtd_root / "images").glob("*/*.jpg")):
        raise FileNotFoundError(f"DTD images are missing: {dtd_root}")
    foreground_report = json.loads(
        foreground_report_path.read_text(encoding="utf-8")
    )
    if (
        not foreground_report.get("complete")
        or foreground_report.get("aggregate_sha256")
        != FOREGROUND_AGGREGATE_SHA256
    ):
        raise ValueError("DiffusionAD foreground materialization is incomplete")
    categories = selected_categories(cli_args.categories)
    seeds = split_ints(cli_args.synthetic_seeds)
    records = []
    failures = []
    for category in categories:
        try:
            records.append(
                run_category(
                    cli_args,
                    source,
                    source_root,
                    dataset_root,
                    dtd_root,
                    foreground_report,
                    category,
                    seeds,
                )
            )
        except Exception as exc:
            failures.append({"category": category, "error": str(exc)})
            logging.exception("DiffusionAD failed for %s", category)
    report = {
        "method": "diffusionad",
        "source_kind": source["source_kind"],
        "source_commit": source["commit"],
        "selection_protocol": "fixed_final_epoch_no_test_evaluation",
        "epochs": cli_args.epochs,
        "max_train_batches": cli_args.max_train_batches,
        "amp_enabled": bool(cli_args.amp),
        "activation_checkpointing": bool(
            cli_args.activation_checkpointing
        ),
        "offload_saved_tensors": bool(
            cli_args.offload_saved_tensors
        ),
        "micro_batch_size": cli_args.micro_batch_size,
        "paper_eligible_full_training": (
            cli_args.epochs == 3000
            and cli_args.max_train_batches is None
        ),
        "categories": categories,
        "completed": len(records),
        "failures": failures,
        "complete": not failures and len(records) == len(categories),
        "records": records,
    }
    report_path = (
        Path(cli_args.external_root)
        / "mvtec15"
        / "diffusionad"
        / "run_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
