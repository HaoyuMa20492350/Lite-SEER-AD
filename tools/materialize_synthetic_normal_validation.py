from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.config import cfg_device, load_config, resolve_device
from seer_ad_v2.data.datasets import DTDTextureDataset, build_dataset
from seer_ad_v2.data.defect_synthesis import synthesize_anomaly
from seer_ad_v2.evaluation.pixel_policy import (
    candidate_pixel_maps,
    candidate_score_maps,
    multiscale_candidate_maps,
)
from seer_ad_v2.evaluation.score_aggregation import image_scores_from_heatmaps
from seer_ad_v2.evaluation.synthetic_validation import evaluate_synthetic_validation
from seer_ad_v2.models.feature_prior import feature_prior_scores, load_feature_prior_components
from seer_ad_v2.utils.io import load_checkpoint, save_json
from seer_ad_v2.utils.seed import seed_everything


SYNTHETIC_MASK_MODES = ("blob", "scratch", "spot", "patch")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate one saved pixel-policy candidate on held-out normal images "
            "and deterministic synthetic defects without reading real anomaly masks."
        )
    )
    p.add_argument("--candidate-run-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--max-normal-images", type=int, default=None)
    p.add_argument("--synthetic-variants", type=int, default=None)
    p.add_argument(
        "--strength-profile",
        choices=["standard", "broad"],
        default=None,
        help=(
            "Synthetic severity schedule. 'broad' alternates standard and "
            "weak defects for each mask mode."
        ),
    )
    p.add_argument("--canonical-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--disable-texture-bank",
        action="store_true",
        help="Use deterministic random textures instead of the configured DTD bank.",
    )
    p.add_argument("--allow-random-feature-weights", action="store_true")
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def stable_seed(seed: int, category: str, path: str, variant: int) -> int:
    digest = hashlib.sha256(f"{seed}:{category}:{path}:{variant}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_normal_image(path: str, size: int) -> torch.Tensor:
    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = REPO_ROOT / image_path
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(image)


def resize_images(images: torch.Tensor, size: int) -> torch.Tensor:
    if images.shape[-2:] == (size, size):
        return images
    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def resize_masks(masks: torch.Tensor, size: int) -> torch.Tensor:
    if masks.shape[-2:] == (size, size):
        return masks
    return F.interpolate(masks, size=(size, size), mode="nearest")


def mild_photometric_view(images: torch.Tensor) -> torch.Tensor:
    values = (images.clamp(-1, 1) + 1.0) * 0.5
    values = values * 0.9 + 0.05
    return (values.clamp(0, 1) * 2.0 - 1.0).clamp(-1, 1)


def calibration_strength(
    mask_mode: str,
    variant: int,
    rng: np.random.RandomState,
    profile: str,
) -> float | None:
    if profile == "standard":
        return None
    cycle = variant // len(SYNTHETIC_MASK_MODES)
    if cycle % 2 == 0:
        return None
    weak_ranges = {
        "blob": (0.08, 0.30),
        "scratch": (0.05, 0.25),
        "spot": (0.05, 0.25),
        "patch": (0.05, 0.25),
    }
    lo, hi = weak_ranges[mask_mode]
    return float(rng.uniform(lo, hi))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.candidate_run_dir)
    out_path = Path(args.out) if args.out else run_dir / "synthetic_validation.npz"
    metrics_path = out_path.with_name(f"{out_path.stem}_metrics.json")
    payload = read_json(run_dir / "run_args.json")
    run_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    if not isinstance(run_args, dict):
        raise ValueError(f"Invalid run_args.json in {run_dir}")

    config_path = resolve_path(str(run_args.get("config") or "configs/mvtec.yaml"))
    if config_path is None:
        raise ValueError("Candidate does not record a config path")
    cfg = load_config(config_path)
    policy_cfg = cfg.get("policy_selection", {}) or {}
    max_normal_images = int(
        args.max_normal_images
        if args.max_normal_images is not None
        else policy_cfg.get("max_normal_images", 16)
    )
    synthetic_variants = int(
        args.synthetic_variants
        if args.synthetic_variants is not None
        else policy_cfg.get("synthetic_variants", len(SYNTHETIC_MASK_MODES))
    )
    if synthetic_variants < 1:
        raise ValueError("synthetic_variants must be at least 1")
    strength_profile = str(
        args.strength_profile
        or policy_cfg.get("synthetic_strength_profile", "broad")
    )
    category = str(run_args.get("category") or "")
    if not category:
        raise ValueError("Candidate does not record a category")
    seed = int(args.seed if args.seed is not None else run_args.get("seed", cfg.get("seed", 7)))
    seed_everything(seed)
    image_size = int(run_args.get("image_size") or cfg.get("dataset", {}).get("image_size", 256))
    device = resolve_device(str(args.device or run_args.get("device") or cfg_device(cfg)))
    feature_checkpoint = resolve_path(run_args.get("feature_prior_checkpoint"))
    if feature_checkpoint is None or not feature_checkpoint.exists():
        raise FileNotFoundError(f"Feature-prior checkpoint is unavailable: {feature_checkpoint}")

    dataset_cfg = cfg.get("dataset", {}) or {}
    dataset_root = resolve_path(str(dataset_cfg.get("root") or ""))
    if dataset_root is None:
        raise ValueError(f"Dataset root is unavailable in {config_path}")
    train_dataset = build_dataset(
        str(dataset_cfg.get("name") or ""),
        dataset_root,
        category,
        "train",
        image_size,
    )
    if any(int(record.label) != 0 for record in train_dataset.records):
        raise ValueError(
            f"Official training split contains non-normal records: {category}"
        )
    normal_paths = np.asarray(
        [str(record.image_path) for record in train_dataset.records]
    )
    if len(normal_paths) == 0:
        raise ValueError(
            f"{category} has no official training-normal images for calibration"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(normal_paths))
    normal_paths = normal_paths[
        order[: min(len(order), max(1, max_normal_images))]
    ]

    texture_source = str(
        policy_cfg.get(
            "synthetic_texture_source",
            "deterministic_random",
        )
    )
    if texture_source not in {
        "deterministic_random",
        "configured_dtd",
    }:
        raise ValueError(
            f"Unsupported synthetic_texture_source: {texture_source}"
        )
    texture_root = resolve_path(
        str((cfg.get("hn_sev", {}) or {}).get("texture_root") or "")
    )
    texture_bank = None
    if (
        texture_source == "configured_dtd"
        and not args.disable_texture_bank
        and texture_root is not None
    ):
        texture_bank = DTDTextureDataset(texture_root)
        if not texture_bank.paths:
            raise FileNotFoundError(
                f"Configured synthetic texture bank is empty: {texture_root}"
            )

    clean_items: list[torch.Tensor] = []
    synthetic_items: list[torch.Tensor] = []
    mask_items: list[torch.Tensor] = []
    sample_paths: list[str] = []
    variant_ids: list[int] = []
    for sample_path in normal_paths:
        canonical_clean = load_normal_image(sample_path, int(args.canonical_size))
        for variant in range(synthetic_variants):
            sample_rng = np.random.RandomState(stable_seed(seed, category, sample_path, variant))
            mask_mode = SYNTHETIC_MASK_MODES[variant % len(SYNTHETIC_MASK_MODES)]
            synthetic, mask = synthesize_anomaly(
                canonical_clean,
                texture_bank=texture_bank,
                strength=calibration_strength(
                    mask_mode,
                    variant,
                    sample_rng,
                    strength_profile,
                ),
                rng=sample_rng,
                mask_mode=mask_mode,
            )
            clean_items.append(canonical_clean)
            synthetic_items.append(synthetic)
            mask_items.append(mask)
            sample_paths.append(sample_path)
            variant_ids.append(variant)

    clean = resize_images(torch.stack(clean_items), image_size)
    synthetic = resize_images(torch.stack(synthetic_items), image_size)
    masks = resize_masks(torch.stack(mask_items), image_size)[:, 0]
    checkpoint = load_checkpoint(feature_checkpoint)
    feature_prior, extractor, layers = load_feature_prior_components(
        checkpoint,
        device,
        allow_random_weights=bool(args.allow_random_feature_weights or run_args.get("allow_random_feature_weights", False)),
    )
    secondary_components = None
    secondary_checkpoint_path = resolve_path(run_args.get("secondary_feature_prior_checkpoint"))
    if secondary_checkpoint_path is not None:
        secondary_checkpoint = load_checkpoint(secondary_checkpoint_path)
        secondary_components = load_feature_prior_components(
            secondary_checkpoint,
            device,
            allow_random_weights=bool(args.allow_random_feature_weights or run_args.get("allow_random_feature_weights", False)),
        )
    image_score_mode = str(run_args.get("image_score_mode") or "top5")

    def score(
        images: torch.Tensor,
        *,
        measure_latency: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        heatmaps: list[np.ndarray] = []
        score_maps: list[np.ndarray] = []
        elapsed = 0.0
        for start in range(0, len(images), max(1, int(args.batch_size))):
            batch = images[start : start + max(1, int(args.batch_size))].to(device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            output = feature_prior_scores(feature_prior, extractor, layers, batch, device, image_size)
            if secondary_components is not None:
                secondary_output = feature_prior_scores(*secondary_components, batch, device, image_size)
                pixel_maps = multiscale_candidate_maps(output, secondary_output, run_args, run_dir)
                selected_score_maps = (
                    pixel_maps
                    if str(run_args.get("image_score_source")) == "multiscale_fusion"
                    else candidate_score_maps(output, run_args)
                )
            else:
                pixel_maps = candidate_pixel_maps(output, run_args, run_dir)
                selected_score_maps = candidate_score_maps(output, run_args)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            if measure_latency:
                elapsed += time.perf_counter() - started
            heatmaps.append(np.asarray(pixel_maps, dtype=np.float32))
            score_maps.append(np.asarray(selected_score_maps, dtype=np.float32))
        heatmaps_np = np.concatenate(heatmaps, axis=0)
        score_maps_np = np.concatenate(score_maps, axis=0)
        scores_np = image_scores_from_heatmaps(score_maps_np, mode=image_score_mode)
        latency_ms = elapsed * 1000.0 / max(1, len(images)) if measure_latency else 0.0
        return heatmaps_np, score_maps_np, scores_np, latency_ms

    clean_heatmaps, clean_score_heatmaps, clean_scores, _ = score(clean)
    synthetic_heatmaps, synthetic_score_heatmaps, synthetic_scores, latency_ms = score(
        synthetic,
        measure_latency=True,
    )
    flipped_heatmaps, _, _, _ = score(torch.flip(synthetic, dims=[3]))
    flipped_heatmaps = np.flip(flipped_heatmaps, axis=2).copy()
    photometric_heatmaps, _, _, _ = score(mild_photometric_view(synthetic))

    metrics = evaluate_synthetic_validation(
        clean_heatmaps,
        synthetic_heatmaps,
        masks.detach().cpu().numpy().astype(np.uint8),
        clean_scores,
        synthetic_scores,
        [flipped_heatmaps, photometric_heatmaps],
    )
    metrics.update(
        {
            "latency_ms": float(latency_ms),
            "candidate_run": str(run_dir),
            "category": category,
            "seed": seed,
            "canonical_size": int(args.canonical_size),
            "candidate_image_size": image_size,
            "normal_images": int(len(normal_paths)),
            "synthetic_variants": synthetic_variants,
            "synthetic_strength_profile": strength_profile,
            "synthetic_mask_modes": sorted(
                {
                    SYNTHETIC_MASK_MODES[
                        variant % len(SYNTHETIC_MASK_MODES)
                    ]
                    for variant in range(synthetic_variants)
                }
            ),
            "synthetic_texture_source": (
                "deterministic_random"
                if texture_bank is None
                else "configured_dtd"
            ),
            "synthetic_texture_root": (
                None if texture_bank is None else str(texture_root)
            ),
            "synthetic_texture_images": (
                0 if texture_bank is None else len(texture_bank.paths)
            ),
            "selection_data": "official_train_normal_images_plus_synthetic_masks",
            "normal_source_split": "train",
            "normal_source_dataset": str(dataset_cfg.get("name") or ""),
            "official_train_normal_pool": int(len(train_dataset.records)),
            "uses_real_anomaly_labels_for_selection": False,
            "uses_real_anomaly_masks_for_selection": False,
            "feature_prior_checkpoint": str(feature_checkpoint),
            "secondary_feature_prior_checkpoint": None
            if secondary_checkpoint_path is None
            else str(secondary_checkpoint_path),
            "pixel_heatmap_source": run_args.get("pixel_heatmap_source", ""),
            "image_score_source": run_args.get("image_score_source", ""),
            "image_score_mode": image_score_mode,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        clean_heatmaps=clean_heatmaps.astype(np.float32),
        synthetic_heatmaps=synthetic_heatmaps.astype(np.float32),
        clean_score_heatmaps=clean_score_heatmaps.astype(np.float32),
        synthetic_score_heatmaps=synthetic_score_heatmaps.astype(np.float32),
        synthetic_masks=masks.detach().cpu().numpy().astype(np.uint8),
        clean_image_scores=clean_scores.astype(np.float32),
        synthetic_image_scores=synthetic_scores.astype(np.float32),
        flipped_synthetic_heatmaps=flipped_heatmaps.astype(np.float32),
        photometric_synthetic_heatmaps=photometric_heatmaps.astype(np.float32),
        paths=np.asarray(sample_paths),
        variant_ids=np.asarray(variant_ids, dtype=np.int32),
        mask_modes=np.asarray(
            [SYNTHETIC_MASK_MODES[variant % len(SYNTHETIC_MASK_MODES)] for variant in variant_ids]
        ),
        seed=np.asarray(seed, dtype=np.int64),
    )
    save_json(metrics, metrics_path)
    print(json.dumps({"artifact": str(out_path), "metrics": metrics}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
