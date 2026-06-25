from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torch.utils.data import DataLoader

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
from tools.materialize_ddad_pretrained import (
    DDAD_FOLDER_ID,
    DDAD_MVTEC_FILES,
    DDAD_MVTEC_SETTINGS,
)
from tools.run_official_patchcore import split_ints, stable_seed


SYNTHETIC_MASK_MODES = ("blob", "scratch", "spot", "patch")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DDAD_CORE_FILES = (
    "config.yaml",
    "main.py",
    "ddad.py",
    "dataset.py",
    "reconstruction.py",
    "anomaly_map.py",
    "feature_extractor.py",
    "resnet.py",
    "unet.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run author-released DDAD MVTec checkpoints and export raw "
            "predictions plus label-free fixed-threshold evidence."
        )
    )
    parser.add_argument("--manifest", default="baselines/official_sources.json")
    parser.add_argument(
        "--source-root",
        default="third_party/official_baselines/ddad",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="third_party/official_baselines/ddad/pretrained/MVTec",
    )
    parser.add_argument("--dataset-root", default="SEER-AD-dataset/MVTec-AD")
    parser.add_argument(
        "--external-root",
        default="baselines/external_outputs",
    )
    parser.add_argument("--categories", default="bottle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--synthetic-seeds", default="7,13,23")
    parser.add_argument("--max-normal-images", type=int, default=16)
    parser.add_argument("--synthetic-variants", type=int, default=4)
    parser.add_argument("--max-normal-fpr", type=float, default=0.005)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
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
        "kornia",
        "numpy",
        "scipy",
        "scikit-learn",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def selected_categories(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return sorted(DDAD_MVTEC_SETTINGS)
    categories = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(categories) - set(DDAD_MVTEC_SETTINGS))
    if unknown:
        raise ValueError("Unknown DDAD categories: " + ", ".join(unknown))
    if not categories:
        raise ValueError("At least one DDAD category is required")
    return categories


def center_crop_224(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[-2:] != (256, 256):
        raise ValueError(
            f"DDAD center crop expects 256x256 tensors, got {tensor.shape[-2:]}"
        )
    return tensor[..., 16:240, 16:240]


def official_image_scores(heatmaps: torch.Tensor) -> torch.Tensor:
    if heatmaps.ndim != 4 or heatmaps.shape[1] != 1:
        raise ValueError(
            "DDAD heatmaps must have shape Bx1xHxW, got "
            f"{tuple(heatmaps.shape)}"
        )
    return heatmaps.flatten(1).amax(dim=1)


def neighborhood_mean(features: torch.Tensor) -> torch.Tensor:
    """Author patchify's 3x3 unfolded mean without its large temporary tensor."""
    return F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)


def _strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def checkpoint_paths(
    checkpoint_root: Path,
    category: str,
) -> tuple[Path, Path]:
    setting = DDAD_MVTEC_SETTINGS[category]
    category_root = checkpoint_root / category
    return (
        category_root / str(setting["unet_checkpoint"]),
        category_root / str(setting["feature_checkpoint"]),
    )


def load_models(
    source_root: Path,
    checkpoint_root: Path,
    category: str,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, Path, Path]:
    source_path = str(source_root.resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from resnet import wide_resnet101_2
    from unet import UNetModel

    unet_path, feature_path = checkpoint_paths(checkpoint_root, category)
    missing = [
        str(path)
        for path in (unet_path, feature_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Official DDAD checkpoints are missing: " + ", ".join(missing)
        )

    unet = UNetModel(
        256,
        64,
        dropout=0.0,
        n_heads=4,
        in_channels=3,
    )
    unet_state = torch.load(
        unet_path,
        map_location="cpu",
        weights_only=True,
    )
    unet.load_state_dict(_strip_module_prefix(unet_state), strict=True)
    unet.to(device).eval()

    feature_extractor = wide_resnet101_2(pretrained=False)
    feature_state = torch.load(
        feature_path,
        map_location="cpu",
        weights_only=True,
    )
    feature_extractor.load_state_dict(
        _strip_module_prefix(feature_state),
        strict=True,
    )
    feature_extractor.to(device).eval()
    return unet, feature_extractor, unet_path, feature_path


def diffusion_alphas(device: torch.device) -> torch.Tensor:
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
    betas_tensor = torch.tensor(
        betas,
        dtype=torch.float32,
        device=device,
    )
    beta = torch.cat(
        [torch.zeros(1, device=device), betas_tensor],
        dim=0,
    )
    return (1.0 - beta).cumprod(dim=0)


def alpha_at(
    alphas: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    return alphas.index_select(0, timestep.long() + 1).view(-1, 1, 1, 1)


@torch.inference_mode()
def reconstruct(
    unet: torch.nn.Module,
    images: torch.Tensor,
    *,
    w: float,
    alphas: torch.Tensor,
) -> torch.Tensor:
    device = images.device
    batch_size = images.shape[0]
    start = torch.full(
        (batch_size,),
        250,
        device=device,
        dtype=torch.long,
    )
    at_start = alpha_at(alphas, start)
    current = (
        at_start.sqrt() * images
        + (1.0 - at_start).sqrt() * torch.randn_like(images)
    )
    sequence = list(range(0, 250, 25))
    sequence_next = [-1] + sequence[:-1]
    for current_step, next_step in zip(
        reversed(sequence),
        reversed(sequence_next),
    ):
        timestep = torch.full(
            (batch_size,),
            current_step,
            device=device,
            dtype=torch.long,
        )
        next_timestep = torch.full(
            (batch_size,),
            next_step,
            device=device,
            dtype=torch.long,
        )
        at = alpha_at(alphas, timestep)
        at_next = alpha_at(alphas, next_timestep)
        predicted_noise = unet(current, timestep.float())
        conditioned = (
            at.sqrt() * images
            + (1.0 - at).sqrt() * predicted_noise
        )
        corrected_noise = (
            predicted_noise
            - (1.0 - at).sqrt() * w * (conditioned - current)
        )
        predicted_x0 = (
            current - corrected_noise * (1.0 - at).sqrt()
        ) / at.sqrt()
        stochastic = (
            (1.0 - at / at_next)
            * (1.0 - at_next)
            / (1.0 - at)
        ).sqrt()
        direction = ((1.0 - at_next) - stochastic.square()).sqrt()
        current = (
            at_next.sqrt() * predicted_x0
            + stochastic * torch.randn_like(images)
            + direction * corrected_noise
        )
    return current


def _imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (((images + 1.0) * 0.5) - mean) / std


@torch.inference_mode()
def official_heatmap(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    feature_extractor: torch.nn.Module,
) -> torch.Tensor:
    pixel_distance = torch.mean(
        torch.abs(reconstructed - target),
        dim=1,
        keepdim=True,
    )
    target_features = feature_extractor(_imagenet_normalize(target))
    output_features = feature_extractor(
        _imagenet_normalize(reconstructed)
    )
    feature_distance = torch.zeros_like(pixel_distance)
    for target_feature, output_feature in zip(
        target_features[1:],
        output_features[1:],
    ):
        target_patch = neighborhood_mean(target_feature)
        output_patch = neighborhood_mean(output_feature)
        distance = 1.0 - F.cosine_similarity(
            target_patch,
            output_patch,
            dim=1,
        )
        feature_distance += F.interpolate(
            distance.unsqueeze(1),
            size=256,
            mode="bilinear",
            align_corners=True,
        )
    anomaly_map = (
        feature_distance
        + (feature_distance.amax() / pixel_distance.amax())
        * pixel_distance
    )
    anomaly_map = gaussian_blur2d(
        anomaly_map,
        kernel_size=(33, 33),
        sigma=(4.0, 4.0),
    )
    return center_crop_224(anomaly_map.sum(dim=1, keepdim=True))


@torch.inference_mode()
def score_batch(
    unet: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    images: torch.Tensor,
    *,
    w: float,
    alphas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reconstructed = reconstruct(unet, images, w=w, alphas=alphas)
    heatmaps = official_heatmap(
        reconstructed,
        images,
        feature_extractor,
    )
    return official_image_scores(heatmaps), heatmaps[:, 0]


@torch.inference_mode()
def score_images(
    unet: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    w: float,
    alphas: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(
            device,
            non_blocking=True,
        )
        batch_scores, batch_heatmaps = score_batch(
            unet,
            feature_extractor,
            batch,
            w=w,
            alphas=alphas,
        )
        scores.append(batch_scores.cpu())
        heatmaps.append(batch_heatmaps.cpu())
    return (
        torch.cat(scores).numpy().astype(np.float32),
        torch.cat(heatmaps).numpy().astype(np.float32),
    )


@torch.inference_mode()
def score_loader(
    unet: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    w: float,
    alphas: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    labels = []
    masks = []
    paths = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        batch_scores, batch_heatmaps = score_batch(
            unet,
            feature_extractor,
            images,
            w=w,
            alphas=alphas,
        )
        scores.append(batch_scores.cpu())
        heatmaps.append(batch_heatmaps.cpu())
        labels.append(batch["label"].reshape(-1))
        masks.append(center_crop_224(batch["mask"])[:, 0])
        paths.extend(str(path) for path in batch["path"])
    return (
        torch.cat(scores).numpy().astype(np.float32),
        torch.cat(heatmaps).numpy().astype(np.float32),
        torch.cat(labels).numpy().astype(np.uint8),
        torch.cat(masks).numpy().astype(np.uint8),
        np.asarray(paths),
    )


def synthetic_evidence(
    unet: torch.nn.Module,
    feature_extractor: torch.nn.Module,
    train_dataset: Any,
    category: str,
    seed: int,
    *,
    device: torch.device,
    max_normal_images: int,
    synthetic_variants: int,
    batch_size: int,
    w: float,
    alphas: torch.Tensor,
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
            mask_items.append(center_crop_224(mask)[0])
            paths.append(path)
            variant_ids.append(variant)

    clean = torch.stack(clean_items)
    synthetic = torch.stack(synthetic_items)
    torch.manual_seed(stable_seed(seed, category, "clean", 0))
    torch.cuda.manual_seed_all(stable_seed(seed, category, "clean", 0))
    clean_scores, clean_heatmaps = score_images(
        unet,
        feature_extractor,
        clean,
        device=device,
        batch_size=batch_size,
        w=w,
        alphas=alphas,
    )
    torch.manual_seed(stable_seed(seed, category, "synthetic", 0))
    torch.cuda.manual_seed_all(stable_seed(seed, category, "synthetic", 0))
    synthetic_scores, synthetic_heatmaps = score_images(
        unet,
        feature_extractor,
        synthetic,
        device=device,
        batch_size=batch_size,
        w=w,
        alphas=alphas,
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


def run_category(
    args: argparse.Namespace,
    source: dict[str, Any],
    source_root: Path,
    checkpoint_root: Path,
    materialization_report: dict[str, Any],
    dataset_root: Path,
    category: str,
    seeds: list[int],
) -> dict[str, Any]:
    artifact_dir = Path(args.external_root) / "mvtec15" / "ddad" / category
    prediction_path = artifact_dir / "predictions.npz"
    policy_path = artifact_dir / "pixel_threshold_policy.json"
    provenance_path = artifact_dir / "provenance.json"
    if (
        args.resume
        and prediction_path.exists()
        and policy_path.exists()
        and provenance_path.exists()
    ):
        return {"category": category, "status": "cached", "out": str(artifact_dir)}

    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    setting = DDAD_MVTEC_SETTINGS[category]
    w = float(setting["w"])
    unet, feature_extractor, unet_path, feature_path = load_models(
        source_root,
        checkpoint_root,
        category,
        device,
    )
    alphas = diffusion_alphas(device)
    train_dataset = build_dataset(
        "mvtec",
        dataset_root,
        category,
        "train",
        256,
    )
    test_dataset = build_dataset(
        "mvtec",
        dataset_root,
        category,
        "test",
        256,
        max_samples=args.max_test_samples,
        sample_seed=42 if args.max_test_samples is not None else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=True,
    )
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    scores, heatmaps, labels, masks, paths = score_loader(
        unet,
        feature_extractor,
        test_loader,
        device=device,
        w=w,
        alphas=alphas,
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

    evidence = []
    for seed in seeds:
        payload = synthetic_evidence(
            unet,
            feature_extractor,
            train_dataset,
            category,
            seed,
            device=device,
            max_normal_images=args.max_normal_images,
            synthetic_variants=args.synthetic_variants,
            batch_size=args.batch_size,
            w=w,
            alphas=alphas,
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
            "method": "ddad",
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
            "method": "ddad",
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

    materialized = {
        (record["category"], record["filename"]): record
        for record in materialization_report["records"]
    }
    unet_record = materialized[
        (category, str(setting["unet_checkpoint"]))
    ]
    feature_record = materialized[
        (category, str(setting["feature_checkpoint"]))
    ]
    provenance = {
        "method": "ddad",
        "dataset": "mvtec15",
        "category": category,
        "source_kind": source["source_kind"],
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "official_implementation": True,
        "execution_command": " ".join(sys.argv),
        "environment": package_versions(),
        "checkpoint_source": (
            "Author-released DDAD MVTec folder from Google Drive folder ID "
            f"{DDAD_FOLDER_ID}"
        ),
        "checkpoint_url": (
            "https://drive.google.com/drive/folders/"
            f"{DDAD_FOLDER_ID}"
        ),
        "checkpoint_files": {
            "unet": {
                "path": str(unet_path),
                "google_drive_id": DDAD_MVTEC_FILES[
                    f"{category}/{setting['unet_checkpoint']}"
                ],
                "size_bytes": unet_record["size_bytes"],
                "sha256": unet_record["sha256"],
            },
            "feature_extractor": {
                "path": str(feature_path),
                "google_drive_id": DDAD_MVTEC_FILES[
                    f"{category}/{setting['feature_checkpoint']}"
                ],
                "size_bytes": feature_record["size_bytes"],
                "sha256": feature_record["sha256"],
            },
        },
        "checkpoint_release_note": (
            "Actual author-released filenames are authoritative. The release "
            "contains bottle/feat8 and leather/feat5, while the README table "
            "lists bottle FE epoch 5 and leather FE epoch 8."
        ),
        "adapter_mode": "pinned_author_models_direct_inference",
        "core_source_sha256": {
            relative: sha256_file(source_root / relative)
            for relative in DDAD_CORE_FILES
        },
        "model_configuration": {
            "input_size": [256, 256],
            "evaluation_crop": [224, 224],
            "unet_base_channels": 64,
            "unet_attention_heads": 4,
            "feature_extractor": "wide_resnet101_2",
            "conditioning_w": w,
            "pixel_feature_weight_v": 1.0,
            "trajectory_steps": 1000,
            "test_start_step": 250,
            "skip": 25,
            "denoising_steps": 10,
            "eta": 1.0,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "test_batch_size": args.batch_size,
        },
        "operator_equivalence": (
            "The official 3x3 Unfold followed by spatial mean is evaluated as "
            "avg_pool2d(kernel=3,stride=1,padding=1) to reduce peak memory."
        ),
        "prediction_export": (
            "Raw author-protocol DDAD anomaly maps without test-set min-max; "
            "image score is the raw map maximum after 224 center crop."
        ),
        "threshold_protocol": policy["protocol"],
        "uses_real_anomaly_labels_for_threshold": False,
        "uses_real_anomaly_masks_for_threshold": False,
    }
    _write_json(provenance_path, provenance)
    del unet, feature_extractor, alphas
    gc.collect()
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
    if args.batch_size != 16:
        raise ValueError(
            "DDAD must use the author's MVTec test batch size of 16 because "
            "its pixel/feature distance scaling is batch-global"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The pinned official DDAD adapter requires CUDA")

    manifest = load_official_source_manifest(args.manifest)
    source = manifest["sources"]["ddad"]
    source_root = Path(args.source_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    marker = json.loads(
        (source_root / ".lite_seer_source.json").read_text(encoding="utf-8")
    )
    if marker.get("commit") != source["commit"]:
        raise ValueError("Cached DDAD source does not match the pinned commit")
    missing_core = [
        str(source_root / relative)
        for relative in DDAD_CORE_FILES
        if not (source_root / relative).is_file()
    ]
    if missing_core:
        raise FileNotFoundError(
            "Pinned DDAD core files are missing: " + ", ".join(missing_core)
        )
    report_path = checkpoint_root / "materialization_report.json"
    materialization_report = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    if (
        not materialization_report.get("complete")
        or int(materialization_report.get("files", 0)) != 30
    ):
        raise ValueError("DDAD checkpoint materialization report is incomplete")
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
                    checkpoint_root,
                    materialization_report,
                    dataset_root,
                    category,
                    seeds,
                )
            )
        except Exception as exc:
            failures.append({"category": category, "error": str(exc)})
            logging.exception("DDAD failed for %s", category)
            gc.collect()
            torch.cuda.empty_cache()
    report = {
        "method": "ddad",
        "source_kind": source["source_kind"],
        "source_commit": source["commit"],
        "checkpoint_folder_id": DDAD_FOLDER_ID,
        "categories": categories,
        "completed": len(records),
        "failures": failures,
        "complete": not failures and len(records) == len(categories),
        "records": records,
    }
    out_path = Path(args.external_root) / "mvtec15" / "ddad" / "run_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_path, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
