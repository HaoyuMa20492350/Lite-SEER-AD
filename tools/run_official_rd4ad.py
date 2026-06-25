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
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.official_sources import load_official_source_manifest
from seer_ad_v2.data.datasets import build_dataset
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
RD4AD_CORE_FILES = (
    "main.py",
    "dataset.py",
    "test.py",
    "resnet.py",
    "de_resnet.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the pinned author-official RD4AD architecture without "
            "test-label model selection and export strict evaluation evidence."
        )
    )
    parser.add_argument("--manifest", default="baselines/official_sources.json")
    parser.add_argument(
        "--source-root",
        default="third_party/official_baselines/rd4ad",
    )
    parser.add_argument("--dataset-root", default="SEER-AD-dataset/MVTec-AD")
    parser.add_argument(
        "--external-root",
        default="baselines/external_outputs",
    )
    parser.add_argument("--categories", default="bottle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--checkpoint-every", type=int, default=10)
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


def rd4ad_loss(
    teacher_features: list[torch.Tensor],
    student_features: list[torch.Tensor],
) -> torch.Tensor:
    if len(teacher_features) != len(student_features):
        raise ValueError("RD4AD teacher and student feature counts must match")
    loss = torch.zeros((), device=teacher_features[0].device)
    cosine = torch.nn.CosineSimilarity(dim=1)
    for teacher, student in zip(teacher_features, student_features):
        loss = loss + torch.mean(
            1.0
            - cosine(
                teacher.reshape(teacher.shape[0], -1),
                student.reshape(student.shape[0], -1),
            )
        )
    return loss


def raw_anomaly_maps(
    teacher_features: list[torch.Tensor],
    student_features: list[torch.Tensor],
    *,
    out_size: int,
    sigma: float = 4.0,
) -> np.ndarray:
    if len(teacher_features) != len(student_features):
        raise ValueError("RD4AD teacher and student feature counts must match")
    batch_size = teacher_features[0].shape[0]
    anomaly = torch.zeros(
        batch_size,
        out_size,
        out_size,
        device=teacher_features[0].device,
    )
    for teacher, student in zip(teacher_features, student_features):
        distance = 1.0 - F.cosine_similarity(teacher, student, dim=1)
        anomaly += F.interpolate(
            distance.unsqueeze(1),
            size=out_size,
            mode="bilinear",
            align_corners=True,
        )[:, 0]
    maps = anomaly.detach().cpu().numpy().astype(np.float32)
    return np.stack(
        [gaussian_filter(item, sigma=sigma) for item in maps],
        axis=0,
    ).astype(np.float32)


def official_image_scores(heatmaps: np.ndarray) -> np.ndarray:
    heatmaps = np.asarray(heatmaps)
    if heatmaps.ndim != 3:
        raise ValueError(
            f"RD4AD heatmaps must have shape BxHxW, got {heatmaps.shape}"
        )
    return heatmaps.reshape(len(heatmaps), -1).max(axis=1).astype(np.float32)


def _load_official_modules(source_root: Path) -> tuple[Any, Any, Any]:
    source_path = str(source_root.resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from dataset import MVTecDataset, get_data_transforms
    from de_resnet import de_wide_resnet50_2
    from resnet import wide_resnet50_2

    return (
        (MVTecDataset, get_data_transforms),
        wide_resnet50_2,
        de_wide_resnet50_2,
    )


def build_models(
    source_root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    _, wide_resnet50_2, de_wide_resnet50_2 = _load_official_modules(
        source_root
    )
    encoder, bottleneck = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)
    encoder.to(device).eval()
    bottleneck.to(device)
    decoder.to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, bottleneck, decoder


def build_loaders(
    source_root: Path,
    dataset_root: Path,
    category: str,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, Any]:
    (MVTecDataset, get_data_transforms), _, _ = _load_official_modules(
        source_root
    )
    image_transform, mask_transform = get_data_transforms(256, 256)
    train_dataset = ImageFolder(
        root=dataset_root / category / "train",
        transform=image_transform,
    )
    test_dataset = MVTecDataset(
        root=str(dataset_root / category),
        transform=image_transform,
        gt_transform=mask_transform,
        phase="test",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(0, num_workers),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=True,
    )
    return train_loader, test_loader, test_dataset


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


def _save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    bottleneck: torch.nn.Module,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, float]],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "bottleneck": bottleneck.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "rng_state": _rng_state(),
            "selection_protocol": "fixed_final_epoch_no_test_evaluation",
        },
        path,
    )


def train_models(
    encoder: torch.nn.Module,
    bottleneck: torch.nn.Module,
    decoder: torch.nn.Module,
    train_loader: DataLoader,
    checkpoint_path: Path,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    checkpoint_every: int,
    restart_training: bool,
) -> tuple[list[dict[str, float]], int]:
    optimizer = torch.optim.Adam(
        list(decoder.parameters()) + list(bottleneck.parameters()),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )
    history: list[dict[str, float]] = []
    start_epoch = 0
    if checkpoint_path.exists() and not restart_training:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        bottleneck.load_state_dict(checkpoint["bottleneck"], strict=True)
        decoder.load_state_dict(checkpoint["decoder"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"])
        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])

    encoder.eval()
    cache_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    cached_features: list[list[torch.Tensor]] = [[], [], []]
    with torch.no_grad():
        for images, _ in cache_loader:
            teacher = encoder(images.to(device, non_blocking=True))
            for index, feature in enumerate(teacher):
                cached_features[index].append(feature)
    teacher_features = [
        torch.cat(features, dim=0)
        for features in cached_features
    ]
    sample_count = teacher_features[0].shape[0]
    batch_size = int(train_loader.batch_size or 1)
    for epoch in range(start_epoch, epochs):
        bottleneck.train()
        decoder.train()
        losses = []
        order = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            teacher = [feature[indices] for feature in teacher_features]
            student = decoder(bottleneck(teacher))
            loss = rd4ad_loss(teacher, student)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        if (
            (epoch + 1) % max(1, checkpoint_every) == 0
            or epoch + 1 == epochs
        ):
            _save_training_checkpoint(
                checkpoint_path,
                epoch=epoch + 1,
                bottleneck=bottleneck,
                decoder=decoder,
                optimizer=optimizer,
                history=history,
            )
        logging.info(
            "RD4AD epoch %d/%d loss=%.6f",
            epoch + 1,
            epochs,
            mean_loss,
        )
    del teacher_features
    return history, start_epoch


@torch.inference_mode()
def score_tensor_batch(
    encoder: torch.nn.Module,
    bottleneck: torch.nn.Module,
    decoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    out_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    teacher = encoder(images)
    student = decoder(bottleneck(teacher))
    heatmaps = raw_anomaly_maps(
        teacher,
        student,
        out_size=out_size,
    )
    return official_image_scores(heatmaps), heatmaps


@torch.inference_mode()
def score_loader(
    encoder: torch.nn.Module,
    bottleneck: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: DataLoader,
    test_dataset: Any,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bottleneck.eval()
    decoder.eval()
    scores = []
    heatmaps = []
    labels = []
    masks = []
    for images, mask, label, _ in loader:
        batch_scores, batch_heatmaps = score_tensor_batch(
            encoder,
            bottleneck,
            decoder,
            images.to(device, non_blocking=True),
        )
        scores.append(batch_scores)
        heatmaps.append(batch_heatmaps)
        labels.extend(label.numpy().astype(np.uint8).tolist())
        masks.append((mask[:, 0].numpy() > 0.5).astype(np.uint8))
    return (
        np.concatenate(scores).astype(np.float32),
        np.concatenate(heatmaps).astype(np.float32),
        np.asarray(labels, dtype=np.uint8),
        np.concatenate(masks).astype(np.uint8),
        np.asarray(test_dataset.img_paths),
    )


def _to_imagenet(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = (images.to(device, non_blocking=True) + 1.0) * 0.5
    return (
        images - IMAGENET_MEAN.to(device)
    ) / IMAGENET_STD.to(device)


@torch.inference_mode()
def score_images(
    encoder: torch.nn.Module,
    bottleneck: torch.nn.Module,
    decoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    for start in range(0, len(images), max(1, batch_size)):
        batch_scores, batch_heatmaps = score_tensor_batch(
            encoder,
            bottleneck,
            decoder,
            _to_imagenet(
                images[start : start + max(1, batch_size)],
                device,
            ),
        )
        scores.append(batch_scores)
        heatmaps.append(batch_heatmaps)
    return (
        np.concatenate(scores).astype(np.float32),
        np.concatenate(heatmaps).astype(np.float32),
    )


def synthetic_evidence(
    encoder: torch.nn.Module,
    bottleneck: torch.nn.Module,
    decoder: torch.nn.Module,
    train_dataset: Any,
    category: str,
    seed: int,
    *,
    device: torch.device,
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
        clean = item["image"]
        path = str(item["path"])
        for variant in range(max(1, synthetic_variants)):
            sample_rng = np.random.RandomState(
                stable_seed(seed, category, path, variant)
            )
            mode = SYNTHETIC_MASK_MODES[variant % len(SYNTHETIC_MASK_MODES)]
            synthetic, mask = synthesize_anomaly(
                clean,
                rng=sample_rng,
                mask_mode=mode,
            )
            clean_items.append(clean)
            synthetic_items.append(synthetic)
            mask_items.append(mask[0])
            paths.append(path)
            variant_ids.append(variant)
    clean = torch.stack(clean_items)
    synthetic = torch.stack(synthetic_items)
    clean_scores, clean_heatmaps = score_images(
        encoder,
        bottleneck,
        decoder,
        clean,
        device=device,
        batch_size=batch_size,
    )
    synthetic_scores, synthetic_heatmaps = score_images(
        encoder,
        bottleneck,
        decoder,
        synthetic,
        device=device,
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


def rd4ad_checkpoint_source() -> str:
    return (
        "Trained locally from the pinned author-official architecture because "
        "the repository does not release MVTec model checkpoints"
    )


def backfill_cached_provenance(path: Path) -> bool:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    if not provenance.get("checkpoint_source"):
        provenance["checkpoint_source"] = rd4ad_checkpoint_source()
        changed = True
    training = provenance.get("training_configuration", {})
    if "paper_eligible_full_training" not in provenance:
        provenance["paper_eligible_full_training"] = (
            int(training.get("epochs", 0)) == 200
        )
        changed = True
    if changed:
        _write_json(path, provenance)
    return changed


def run_category(
    args: argparse.Namespace,
    source: dict[str, Any],
    source_root: Path,
    dataset_root: Path,
    category: str,
    seeds: list[int],
) -> dict[str, Any]:
    artifact_dir = Path(args.external_root) / "mvtec15" / "rd4ad" / category
    prediction_path = artifact_dir / "predictions.npz"
    policy_path = artifact_dir / "pixel_threshold_policy.json"
    provenance_path = artifact_dir / "provenance.json"
    if (
        args.resume
        and prediction_path.exists()
        and policy_path.exists()
        and provenance_path.exists()
    ):
        backfill_cached_provenance(provenance_path)
        return {"category": category, "status": "cached", "out": str(artifact_dir)}

    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "training_checkpoint.pth"
    device = torch.device(args.device)
    setup_seed(args.seed)
    encoder, bottleneck, decoder = build_models(source_root, device)
    train_loader, test_loader, test_dataset = build_loaders(
        source_root,
        dataset_root,
        category,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_started = time.perf_counter()
    history, resumed_from_epoch = train_models(
        encoder,
        bottleneck,
        decoder,
        train_loader,
        checkpoint_path,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
        restart_training=args.restart_training,
    )
    training_seconds = time.perf_counter() - train_started
    _write_json(
        artifact_dir / "training_history.json",
        {
            "epochs": args.epochs,
            "resumed_from_epoch": resumed_from_epoch,
            "training_seconds_this_invocation": training_seconds,
            "history": history,
        },
    )

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    scores, heatmaps, labels, masks, paths = score_loader(
        encoder,
        bottleneck,
        decoder,
        test_loader,
        test_dataset,
        device=device,
    )
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

    threshold_dataset = build_dataset(
        "mvtec",
        dataset_root,
        category,
        "train",
        256,
    )
    evidence = []
    for seed in seeds:
        payload = synthetic_evidence(
            encoder,
            bottleneck,
            decoder,
            threshold_dataset,
            category,
            seed,
            device=device,
            max_normal_images=args.max_normal_images,
            synthetic_variants=args.synthetic_variants,
            batch_size=args.batch_size,
        )
        np.savez_compressed(
            artifact_dir / f"synthetic_validation_seed{seed}.npz",
            **payload,
        )
        evidence.append(payload)
    policy = select_synthetic_normal_threshold(
        np.concatenate([item["clean_heatmaps"] for item in evidence], axis=0),
        np.concatenate(
            [item["synthetic_heatmaps"] for item in evidence],
            axis=0,
        ),
        np.concatenate([item["synthetic_masks"] for item in evidence], axis=0),
        max_normal_fpr=args.max_normal_fpr,
    )
    policy.update(
        {
            "method": "rd4ad",
            "dataset": "mvtec15",
            "category": category,
            "synthetic_seeds": seeds,
            "source_artifacts": [
                str(artifact_dir / f"synthetic_validation_seed{seed}.npz")
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
            "method": "rd4ad",
            "display_method": source["display_name"],
            "dataset": "mvtec15",
            "category": category,
            "official_implementation": True,
            "source_kind": source["source_kind"],
            "source_commit": source["commit"],
            "latency_ms_mean": elapsed * 1000.0 / max(1, len(labels)),
            "prediction_count": int(len(labels)),
            "training_epochs": args.epochs,
        }
    )
    _write_json(artifact_dir / "metrics.json", metrics)

    backbone_path = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / BACKBONE_FILENAME
    )
    provenance = {
        "method": "rd4ad",
        "dataset": "mvtec15",
        "category": category,
        "source_kind": source["source_kind"],
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "official_implementation": True,
        "execution_command": " ".join(sys.argv),
        "environment": package_versions(),
        "checkpoint_source": rd4ad_checkpoint_source(),
        "paper_eligible_full_training": args.epochs == 200,
        "adapter_mode": (
            "pinned_author_architecture_fixed_final_epoch_no_test_selection"
        ),
        "core_source_sha256": {
            relative: sha256_file(source_root / relative)
            for relative in RD4AD_CORE_FILES
        },
        "backbone": {
            "name": "wide_resnet50_2",
            "source_url": BACKBONE_URL,
            "path": str(backbone_path),
            "size_bytes": backbone_path.stat().st_size,
            "sha256": sha256_file(backbone_path),
        },
        "trained_checkpoint": {
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "training_configuration": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "betas": [0.5, 0.999],
            "seed": args.seed,
            "input_size": [256, 256],
            "model_selection": "fixed_final_epoch",
            "test_evaluation_during_training": False,
            "uses_test_labels_for_model_selection": False,
            "frozen_teacher_feature_cache": (
                "exact float32 features precomputed once because the author "
                "teacher is frozen and training transforms are deterministic"
            ),
        },
        "prediction_export": (
            "Raw sum of three teacher-student cosine-distance maps after "
            "bilinear upsampling and author sigma=4 Gaussian smoothing"
        ),
        "threshold_protocol": policy["protocol"],
        "uses_real_anomaly_labels_for_threshold": False,
        "uses_real_anomaly_masks_for_threshold": False,
    }
    _write_json(provenance_path, provenance)
    del encoder, bottleneck, decoder
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "category": category,
        "status": "completed",
        "out": str(artifact_dir),
        "metrics": metrics,
        "training_seconds": training_seconds,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The official RD4AD training adapter requires CUDA")
    if args.epochs < 1:
        raise ValueError("RD4AD epochs must be positive")

    manifest = load_official_source_manifest(args.manifest)
    source = manifest["sources"]["rd4ad"]
    source_root = Path(args.source_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    marker = json.loads(
        (source_root / ".lite_seer_source.json").read_text(encoding="utf-8")
    )
    if marker.get("commit") != source["commit"]:
        raise ValueError("Cached RD4AD source does not match the pinned commit")
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
            logging.exception("RD4AD failed for %s", category)
            gc.collect()
            torch.cuda.empty_cache()
    report = {
        "method": "rd4ad",
        "source_kind": source["source_kind"],
        "source_commit": source["commit"],
        "categories": categories,
        "completed": len(records),
        "failures": failures,
        "complete": not failures and len(records) == len(categories),
        "records": records,
    }
    out_path = Path(args.external_root) / "mvtec15" / "rd4ad" / "run_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_path, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
