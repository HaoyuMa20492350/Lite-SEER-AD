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
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.official_sources import load_official_source_manifest
from seer_ad_v2.data.defect_synthesis import synthesize_anomaly
from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.pixel_threshold_policy import (
    save_pixel_threshold_policy,
    select_synthetic_normal_threshold,
)
from tools.materialize_patchcore_pretrained import selected_categories
from tools.run_official_patchcore import split_ints, stable_seed


CHECKPOINT_ID = "1v282ZlibC-b0H9sjLUlOSCFNzEv-TIuh"
CHECKPOINT_URL = (
    "https://drive.google.com/file/d/"
    f"{CHECKPOINT_ID}/view?usp=sharing"
)
SYNTHETIC_MASK_MODES = ("blob", "scratch", "spot", "patch")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
UNIAD_CORE_FILES = (
    "experiments/MVTec-AD/config.yaml",
    "models/model_helper.py",
    "models/initializer.py",
    "models/necks/mfcn.py",
    "models/reconstructions/uniad.py",
    "models/backbones/efficientnet/__init__.py",
    "models/backbones/efficientnet/model.py",
    "models/backbones/efficientnet/utils.py",
    "utils/misc_helper.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned author-official UniAD model and export raw "
            "predictions plus label-free fixed-threshold evidence."
        )
    )
    parser.add_argument("--manifest", default="baselines/official_sources.json")
    parser.add_argument(
        "--source-root",
        default="third_party/official_baselines/uniad",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "third_party/official_baselines/uniad/pretrained/"
            "mvtec_1gpu_ckpt.pth.tar"
        ),
    )
    parser.add_argument("--dataset-root", default="SEER-AD-dataset/MVTec-AD")
    parser.add_argument(
        "--external-root",
        default="baselines/external_outputs",
    )
    parser.add_argument("--categories", default="bottle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--synthetic-seeds", default="7,13,23")
    parser.add_argument("--max-normal-images", type=int, default=16)
    parser.add_argument("--synthetic-variants", type=int, default=4)
    parser.add_argument("--max-normal-fpr", type=float, default=0.005)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    packages = (
        "torch",
        "torchvision",
        "einops",
        "numpy",
        "opencv-python",
        "PyYAML",
        "scikit-learn",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def official_image_scores(heatmaps: torch.Tensor) -> torch.Tensor:
    if heatmaps.ndim != 4 or heatmaps.shape[1] != 1:
        raise ValueError(
            "UniAD heatmaps must have shape Bx1xHxW, got "
            f"{tuple(heatmaps.shape)}"
        )
    pooled = F.avg_pool2d(heatmaps, kernel_size=(16, 16), stride=1)
    return pooled.flatten(1).amax(dim=1)


def _preprocess(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    images = (images.clamp(-1.0, 1.0) + 1.0) * 0.5
    return (
        images - IMAGENET_MEAN.to(device)
    ) / IMAGENET_STD.to(device)


def load_uniad_model(
    source_root: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    source_root = source_root.resolve()
    missing = [
        str(source_root / relative)
        for relative in UNIAD_CORE_FILES
        if not (source_root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Pinned UniAD core files are missing: " + ", ".join(missing)
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Official UniAD checkpoint is missing: {checkpoint_path}"
        )
    source_path = str(source_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    from models.model_helper import ModelHelper
    from utils.misc_helper import update_config

    config_path = source_root / "experiments" / "MVTec-AD" / "config.yaml"
    config = EasyDict(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config = update_config(config)
    # The official checkpoint contains every backbone tensor.
    config.net[0].kwargs.pretrained = False
    config.net[2].kwargs.save_recon = False
    model = ModelHelper(config.net).to(device)
    model.device = device

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def score_images(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    model.eval()
    for start in range(0, len(images), max(1, batch_size)):
        batch = _preprocess(
            images[start : start + max(1, batch_size)],
            device,
        )
        output = model({"image": batch})
        pred = output["pred"]
        scores.append(official_image_scores(pred).detach().cpu())
        heatmaps.append(pred[:, 0].detach().cpu())
    return (
        torch.cat(scores).numpy().astype(np.float32),
        torch.cat(heatmaps).numpy().astype(np.float32),
    )


@torch.inference_mode()
def score_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    labels = []
    masks = []
    paths = []
    model.eval()
    for batch in loader:
        output = model({"image": _preprocess(batch["image"], device)})
        pred = output["pred"]
        scores.append(official_image_scores(pred).detach().cpu())
        heatmaps.append(pred[:, 0].detach().cpu())
        labels.append(batch["label"].reshape(-1))
        masks.append(batch["mask"][:, 0])
        paths.extend(str(path) for path in batch["path"])
    return (
        torch.cat(scores).numpy().astype(np.float32),
        torch.cat(heatmaps).numpy().astype(np.float32),
        torch.cat(labels).numpy().astype(np.uint8),
        torch.cat(masks).numpy().astype(np.uint8),
        np.asarray(paths),
    )


def synthetic_evidence(
    model: torch.nn.Module,
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
        model,
        clean,
        device=device,
        batch_size=batch_size,
    )
    synthetic_scores, synthetic_heatmaps = score_images(
        model,
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


def run_category(
    args: argparse.Namespace,
    source: dict[str, Any],
    source_root: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    dataset_root: Path,
    category: str,
    seeds: list[int],
    model: torch.nn.Module,
) -> dict[str, Any]:
    artifact_dir = Path(args.external_root) / "mvtec15" / "uniad" / category
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
    train_dataset = build_dataset(
        "mvtec",
        dataset_root,
        category,
        "train",
        224,
    )
    test_dataset = build_dataset(
        "mvtec",
        dataset_root,
        category,
        "test",
        224,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    scores, heatmaps, labels, masks, paths = score_loader(
        model,
        test_loader,
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

    evidence = []
    for seed in seeds:
        payload = synthetic_evidence(
            model,
            train_dataset,
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
            "method": "uniad",
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
            "method": "uniad",
            "display_method": source["display_name"],
            "dataset": "mvtec15",
            "category": category,
            "official_implementation": True,
            "source_kind": source["source_kind"],
            "source_commit": source["commit"],
            "checkpoint_sha256": checkpoint_sha256,
            "latency_ms_mean": elapsed * 1000.0 / max(1, len(labels)),
            "prediction_count": int(len(labels)),
        }
    )
    _write_json(artifact_dir / "metrics.json", metrics)
    provenance = {
        "method": "uniad",
        "dataset": "mvtec15",
        "category": category,
        "source_kind": source["source_kind"],
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "official_implementation": True,
        "execution_command": " ".join(sys.argv),
        "environment": package_versions(),
        "checkpoint_source": (
            "Author-released 1-GPU MVTec AD checkpoint from Google Drive "
            f"file ID {CHECKPOINT_ID}"
        ),
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "adapter_mode": "pinned_author_model_single_gpu_direct_inference",
        "core_source_sha256": {
            relative: sha256_file(source_root / relative)
            for relative in UNIAD_CORE_FILES
        },
        "model_configuration": {
            "input_size": [224, 224],
            "backbone": "efficientnet_b4",
            "backbone_outblocks": [1, 5, 9, 21],
            "feature_size": [14, 14],
            "hidden_dim": 256,
            "nhead": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "neighbor_size": [7, 7],
        },
        "prediction_export": (
            "raw official UniAD L2 feature-reconstruction maps without "
            "test-set min-max; image score is official 16x16 average-pool max"
        ),
        "threshold_protocol": policy["protocol"],
        "uses_real_anomaly_labels_for_threshold": False,
        "uses_real_anomaly_masks_for_threshold": False,
    }
    _write_json(provenance_path, provenance)
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
    source = manifest["sources"]["uniad"]
    source_root = Path(args.source_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    marker_path = source_root / ".lite_seer_source.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("commit") != source["commit"]:
        raise ValueError("Cached UniAD source does not match the pinned commit")
    if not dataset_root.exists():
        raise FileNotFoundError(f"MVTec AD dataset is missing: {dataset_root}")
    categories = selected_categories(args.categories)
    seeds = split_ints(args.synthetic_seeds)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(
            "The pinned official UniAD implementation requires a CUDA device"
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    model = load_uniad_model(source_root, checkpoint_path, device)
    records = []
    failures = []
    for category in categories:
        try:
            records.append(
                run_category(
                    args,
                    source,
                    source_root,
                    checkpoint_path,
                    checkpoint_sha256,
                    dataset_root,
                    category,
                    seeds,
                    model,
                )
            )
        except Exception as exc:
            failures.append({"category": category, "error": str(exc)})
            logging.exception("UniAD failed for %s", category)
    report = {
        "method": "uniad",
        "source_kind": source["source_kind"],
        "source_commit": source["commit"],
        "checkpoint_sha256": checkpoint_sha256,
        "categories": categories,
        "completed": len(records),
        "failures": failures,
        "complete": not failures and len(records) == len(categories),
        "records": records,
    }
    report_path = Path(args.external_root) / "mvtec15" / "uniad" / "run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    print(json.dumps(report, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
