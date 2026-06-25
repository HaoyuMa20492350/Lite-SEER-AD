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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


CHECKPOINT_ID = "1eOE8wXNihjsiDvDANHFbg_mQkLesDrs1"
CHECKPOINT_URL = f"https://drive.google.com/uc?id={CHECKPOINT_ID}"
CHECKPOINT_BUNDLE_SHA256 = (
    "83075056b97c1a5b9550b74c222997ba1dbf455bc43fda41ff52774999d0ded7"
)
BASE_MODEL_NAME = "DRAEM_seg_large_ae_large_0.0001_800_bs8"
SYNTHETIC_MASK_MODES = ("blob", "scratch", "spot", "patch")
DRAEM_CORE_FILES = (
    "model_unet.py",
    "test_DRAEM.py",
    "data_loader.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run author-released DRAEM checkpoints and export raw "
            "predictions plus label-free fixed-threshold evidence."
        )
    )
    parser.add_argument("--manifest", default="baselines/official_sources.json")
    parser.add_argument(
        "--source-root",
        default="third_party/official_baselines/draem",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=(
            "third_party/official_baselines/draem/pretrained/"
            "DRAEM_checkpoints"
        ),
    )
    parser.add_argument(
        "--checkpoint-bundle",
        default=(
            "third_party/official_baselines/draem/pretrained/"
            "DRAEM_checkpoints.zip"
        ),
    )
    parser.add_argument("--dataset-root", default="SEER-AD-dataset/MVTec-AD")
    parser.add_argument(
        "--external-root",
        default="baselines/external_outputs",
    )
    parser.add_argument("--categories", default="bottle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
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
        "numpy",
        "opencv-python",
        "scikit-learn",
        "scipy",
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
            "DRAEM heatmaps must have shape Bx1xHxW, got "
            f"{tuple(heatmaps.shape)}"
        )
    pooled = F.avg_pool2d(heatmaps, kernel_size=21, stride=1, padding=10)
    return pooled.flatten(1).amax(dim=1)


def read_bgr_image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    image = cv2.resize(image, dsize=(256, 256))
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1))


def read_official_mask(path: Path | None, image_shape: tuple[int, int]) -> torch.Tensor:
    if path is None:
        return torch.zeros(image_shape, dtype=torch.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")
    mask = cv2.resize(mask, dsize=(image_shape[1], image_shape[0]))
    return torch.from_numpy((mask.astype(np.float32) / 255.0).astype(np.uint8))


class DRAEMTestDataset(Dataset):
    def __init__(self, dataset_root: Path, category: str) -> None:
        self.category_root = dataset_root / category
        self.images = sorted((self.category_root / "test").glob("*/*.png"))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.images[index]
        defect = image_path.parent.name
        label = int(defect != "good")
        mask_path = None
        if label:
            mask_path = (
                self.category_root
                / "ground_truth"
                / defect
                / f"{image_path.stem}_mask.png"
            )
        return {
            "image": read_bgr_image(image_path),
            "mask": read_official_mask(mask_path, (256, 256)),
            "label": label,
            "path": str(image_path),
        }


def checkpoint_paths(
    checkpoint_root: Path,
    category: str,
) -> tuple[Path, Path]:
    stem = f"{BASE_MODEL_NAME}_{category}_"
    return (
        checkpoint_root / f"{stem}.pckl",
        checkpoint_root / f"{stem}_seg.pckl",
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
    from model_unet import DiscriminativeSubNetwork, ReconstructiveSubNetwork

    reconstruction_path, segmentation_path = checkpoint_paths(
        checkpoint_root,
        category,
    )
    missing = [
        str(path)
        for path in (reconstruction_path, segmentation_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Official DRAEM checkpoints are missing: " + ", ".join(missing)
        )
    reconstruction = ReconstructiveSubNetwork(
        in_channels=3,
        out_channels=3,
    ).to(device)
    reconstruction.load_state_dict(
        torch.load(
            reconstruction_path,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    segmentation = DiscriminativeSubNetwork(
        in_channels=6,
        out_channels=2,
    ).to(device)
    segmentation.load_state_dict(
        torch.load(
            segmentation_path,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    reconstruction.eval()
    segmentation.eval()
    return reconstruction, segmentation, reconstruction_path, segmentation_path


@torch.inference_mode()
def score_images(
    reconstruction: torch.nn.Module,
    segmentation: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    for start in range(0, len(images), max(1, batch_size)):
        batch = images[start : start + max(1, batch_size)].to(
            device,
            non_blocking=True,
        )
        reconstructed = reconstruction(batch)
        logits = segmentation(
            torch.cat((reconstructed.detach(), batch), dim=1)
        )
        pred = torch.softmax(logits, dim=1)[:, 1:2]
        scores.append(official_image_scores(pred).detach().cpu())
        heatmaps.append(pred[:, 0].detach().cpu())
    return (
        torch.cat(scores).numpy().astype(np.float32),
        torch.cat(heatmaps).numpy().astype(np.float32),
    )


@torch.inference_mode()
def score_loader(
    reconstruction: torch.nn.Module,
    segmentation: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    labels = []
    masks = []
    paths = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        reconstructed = reconstruction(image)
        logits = segmentation(
            torch.cat((reconstructed.detach(), image), dim=1)
        )
        pred = torch.softmax(logits, dim=1)[:, 1:2]
        scores.append(official_image_scores(pred).detach().cpu())
        heatmaps.append(pred[:, 0].detach().cpu())
        labels.append(batch["label"].reshape(-1))
        masks.append(batch["mask"])
        paths.extend(str(path) for path in batch["path"])
    return (
        torch.cat(scores).numpy().astype(np.float32),
        torch.cat(heatmaps).numpy().astype(np.float32),
        torch.cat(labels).numpy().astype(np.uint8),
        torch.cat(masks).numpy().astype(np.uint8),
        np.asarray(paths),
    )


def synthetic_evidence(
    reconstruction: torch.nn.Module,
    segmentation: torch.nn.Module,
    train_paths: list[Path],
    category: str,
    seed: int,
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
        clean_01 = read_bgr_image(path)
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
    clean = torch.stack(clean_items)
    synthetic = torch.stack(synthetic_items)
    clean_scores, clean_heatmaps = score_images(
        reconstruction,
        segmentation,
        clean,
        device=device,
        batch_size=batch_size,
    )
    synthetic_scores, synthetic_heatmaps = score_images(
        reconstruction,
        segmentation,
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
    checkpoint_root: Path,
    checkpoint_bundle: Path,
    dataset_root: Path,
    category: str,
    seeds: list[int],
) -> dict[str, Any]:
    artifact_dir = Path(args.external_root) / "mvtec15" / "draem" / category
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
    reconstruction, segmentation, reconstruction_path, segmentation_path = (
        load_models(source_root, checkpoint_root, category, device)
    )
    test_dataset = DRAEMTestDataset(dataset_root, category)
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
        reconstruction,
        segmentation,
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

    train_paths = sorted((dataset_root / category / "train" / "good").glob("*.png"))
    evidence = []
    for seed in seeds:
        payload = synthetic_evidence(
            reconstruction,
            segmentation,
            train_paths,
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
            "method": "draem",
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
            "method": "draem",
            "display_method": source["display_name"],
            "dataset": "mvtec15",
            "category": category,
            "official_implementation": True,
            "source_kind": source["source_kind"],
            "source_commit": source["commit"],
            "checkpoint_bundle_sha256": CHECKPOINT_BUNDLE_SHA256,
            "latency_ms_mean": elapsed * 1000.0 / max(1, len(labels)),
            "prediction_count": int(len(labels)),
        }
    )
    _write_json(artifact_dir / "metrics.json", metrics)
    provenance = {
        "method": "draem",
        "dataset": "mvtec15",
        "category": category,
        "source_kind": source["source_kind"],
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "official_implementation": True,
        "execution_command": " ".join(sys.argv),
        "environment": package_versions(),
        "checkpoint_source": (
            "Author-released 15-class DRAEM checkpoint bundle from Google "
            f"Drive file ID {CHECKPOINT_ID}"
        ),
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_bundle_path": str(checkpoint_bundle),
        "checkpoint_bundle_sha256": CHECKPOINT_BUNDLE_SHA256,
        "checkpoint_bundle_size_bytes": checkpoint_bundle.stat().st_size,
        "checkpoint_files": {
            str(reconstruction_path): sha256_file(reconstruction_path),
            str(segmentation_path): sha256_file(segmentation_path),
        },
        "adapter_mode": "pinned_author_models_direct_inference",
        "core_source_sha256": {
            relative: sha256_file(source_root / relative)
            for relative in DRAEM_CORE_FILES
        },
        "model_configuration": {
            "input_size": [256, 256],
            "input_color_order": "BGR",
            "reconstructive_base_width": 128,
            "discriminative_base_channels": 64,
            "base_model_name": BASE_MODEL_NAME,
        },
        "prediction_export": (
            "raw official DRAEM anomaly-class softmax maps without test-set "
            "min-max; image score is official 21x21 average-pool max"
        ),
        "threshold_protocol": policy["protocol"],
        "uses_real_anomaly_labels_for_threshold": False,
        "uses_real_anomaly_masks_for_threshold": False,
    }
    _write_json(provenance_path, provenance)
    del reconstruction, segmentation
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
    source = manifest["sources"]["draem"]
    source_root = Path(args.source_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    checkpoint_bundle = Path(args.checkpoint_bundle).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    marker = json.loads(
        (source_root / ".lite_seer_source.json").read_text(encoding="utf-8")
    )
    if marker.get("commit") != source["commit"]:
        raise ValueError("Cached DRAEM source does not match the pinned commit")
    if not dataset_root.exists():
        raise FileNotFoundError(f"MVTec AD dataset is missing: {dataset_root}")
    if not checkpoint_bundle.is_file():
        raise FileNotFoundError(
            f"DRAEM checkpoint bundle is missing: {checkpoint_bundle}"
        )
    bundle_sha256 = sha256_file(checkpoint_bundle)
    if bundle_sha256 != CHECKPOINT_BUNDLE_SHA256:
        raise ValueError(
            "DRAEM checkpoint bundle SHA256 mismatch: "
            f"{bundle_sha256} != {CHECKPOINT_BUNDLE_SHA256}"
        )
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
                    checkpoint_bundle,
                    dataset_root,
                    category,
                    seeds,
                )
            )
        except Exception as exc:
            failures.append({"category": category, "error": str(exc)})
            logging.exception("DRAEM failed for %s", category)
    report = {
        "method": "draem",
        "source_kind": source["source_kind"],
        "source_commit": source["commit"],
        "checkpoint_bundle_sha256": bundle_sha256,
        "categories": categories,
        "completed": len(records),
        "failures": failures,
        "complete": not failures and len(records) == len(categories),
        "records": records,
    }
    report_path = Path(args.external_root) / "mvtec15" / "draem" / "run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
