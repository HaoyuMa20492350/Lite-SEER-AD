from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import sys
import time
import types
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import torch
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


SYNTHETIC_MASK_MODES = ("blob", "scratch", "spot", "patch")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PADIM_CORE_FILES = (
    "src/anomalib/models/components/base/dynamic_buffer.py",
    "src/anomalib/models/components/feature_extractors/utils.py",
    "src/anomalib/models/components/feature_extractors/timm.py",
    "src/anomalib/models/components/filters/blur.py",
    "src/anomalib/models/components/stats/multi_variate_gaussian.py",
    "src/anomalib/models/image/padim/anomaly_map.py",
    "src/anomalib/models/image/padim/torch_model.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned Anomalib PaDiM reference core and export raw "
            "predictions plus label-free fixed-threshold evidence."
        )
    )
    parser.add_argument("--manifest", default="baselines/official_sources.json")
    parser.add_argument(
        "--source-root",
        default="third_party/official_baselines/padim",
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
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--n-features", type=int, default=100)
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
        "timm",
        "kornia",
        "numpy",
        "scipy",
        "scikit-learn",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _package(name: str, path: Path | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)] if path is not None else []
    sys.modules[name] = module
    return module


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_padim_model_class(source_root: Path) -> type[torch.nn.Module]:
    source_root = source_root.resolve()
    src = source_root / "src" / "anomalib"
    missing = [
        str(source_root / relative)
        for relative in PADIM_CORE_FILES
        if not (source_root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Pinned PaDiM core files are missing: " + ", ".join(missing)
        )
    try:
        import kornia  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PaDiM core requires kornia. Install kornia==0.8.2."
        ) from exc

    for name in list(sys.modules):
        if name == "anomalib" or name.startswith("anomalib."):
            del sys.modules[name]

    _package("anomalib", src)
    data_module = _package("anomalib.data")
    inference_batch = namedtuple(
        "InferenceBatch",
        ("pred_score", "pred_label", "anomaly_map", "pred_mask"),
        defaults=(None, None, None, None),
    )
    data_module.InferenceBatch = inference_batch

    models = _package("anomalib.models", src / "models")
    components = _package(
        "anomalib.models.components",
        src / "models" / "components",
    )
    models.components = components

    base = _package(
        "anomalib.models.components.base",
        src / "models" / "components" / "base",
    )
    dynamic = _load_module(
        "anomalib.models.components.base.dynamic_buffer",
        source_root / PADIM_CORE_FILES[0],
    )
    base.DynamicBufferMixin = dynamic.DynamicBufferMixin

    feature_extractors = _package(
        "anomalib.models.components.feature_extractors",
        src / "models" / "components" / "feature_extractors",
    )
    feature_utils = _load_module(
        "anomalib.models.components.feature_extractors.utils",
        source_root / PADIM_CORE_FILES[1],
    )
    timm_module = _load_module(
        "anomalib.models.components.feature_extractors.timm",
        source_root / PADIM_CORE_FILES[2],
    )
    feature_extractors.TimmFeatureExtractor = timm_module.TimmFeatureExtractor
    feature_extractors.dryrun_find_featuremap_dims = (
        feature_utils.dryrun_find_featuremap_dims
    )

    filters = _package(
        "anomalib.models.components.filters",
        src / "models" / "components" / "filters",
    )
    blur = _load_module(
        "anomalib.models.components.filters.blur",
        source_root / PADIM_CORE_FILES[3],
    )
    filters.GaussianBlur2d = blur.GaussianBlur2d

    stats = _package(
        "anomalib.models.components.stats",
        src / "models" / "components" / "stats",
    )
    gaussian = _load_module(
        "anomalib.models.components.stats.multi_variate_gaussian",
        source_root / PADIM_CORE_FILES[4],
    )
    stats.MultiVariateGaussian = gaussian.MultiVariateGaussian

    components.DynamicBufferMixin = dynamic.DynamicBufferMixin
    components.TimmFeatureExtractor = timm_module.TimmFeatureExtractor
    components.GaussianBlur2d = blur.GaussianBlur2d
    components.MultiVariateGaussian = gaussian.MultiVariateGaussian

    image = _package(
        "anomalib.models.image",
        src / "models" / "image",
    )
    padim = _package(
        "anomalib.models.image.padim",
        src / "models" / "image" / "padim",
    )
    image.padim = padim
    anomaly_map = _load_module(
        "anomalib.models.image.padim.anomaly_map",
        source_root / PADIM_CORE_FILES[5],
    )
    padim.AnomalyMapGenerator = anomaly_map.AnomalyMapGenerator
    torch_model = _load_module(
        "anomalib.models.image.padim.torch_model",
        source_root / PADIM_CORE_FILES[6],
    )
    return torch_model.PadimModel


def _preprocess(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    images = (images.clamp(-1.0, 1.0) + 1.0) * 0.5
    return (
        images - IMAGENET_MEAN.to(device)
    ) / IMAGENET_STD.to(device)


@torch.no_grad()
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
        output = model(batch)
        scores.append(output.pred_score.detach().cpu().reshape(-1))
        heatmaps.append(output.anomaly_map[:, 0].detach().cpu())
    return (
        torch.cat(scores).numpy().astype(np.float32),
        torch.cat(heatmaps).numpy().astype(np.float32),
    )


def fit_model(
    model_class: type[torch.nn.Module],
    train_loader: DataLoader,
    *,
    device: torch.device,
    model_seed: int,
    n_features: int,
) -> torch.nn.Module:
    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
    model = model_class(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
        n_features=n_features,
    ).to(device)
    model.train()
    with torch.no_grad():
        for batch in train_loader:
            model(_preprocess(batch["image"], device))
    model.fit()
    model.eval()
    return model


@torch.no_grad()
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
        output = model(_preprocess(batch["image"], device))
        scores.append(output.pred_score.detach().cpu().reshape(-1))
        heatmaps.append(output.anomaly_map[:, 0].detach().cpu())
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
    dataset_root: Path,
    category: str,
    seeds: list[int],
    model_class: type[torch.nn.Module],
) -> dict[str, Any]:
    artifact_dir = Path(args.external_root) / "mvtec15" / "padim" / category
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
        256,
    )
    test_dataset = build_dataset(
        "mvtec",
        dataset_root,
        category,
        "test",
        256,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda",
    )
    fit_started = time.perf_counter()
    model = fit_model(
        model_class,
        train_loader,
        device=device,
        model_seed=args.model_seed,
        n_features=args.n_features,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - fit_started

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_started = time.perf_counter()
    scores, heatmaps, labels, masks, paths = score_loader(
        model,
        test_loader,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_seconds = time.perf_counter() - test_started
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
            "method": "padim",
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
            "method": "padim",
            "display_method": source["display_name"],
            "dataset": "mvtec15",
            "category": category,
            "official_implementation": False,
            "source_kind": source["source_kind"],
            "source_commit": source["commit"],
            "model_seed": args.model_seed,
            "n_features": args.n_features,
            "fit_seconds": fit_seconds,
            "latency_ms_mean": test_seconds * 1000.0 / max(1, len(labels)),
            "prediction_count": int(len(labels)),
        }
    )
    _write_json(artifact_dir / "metrics.json", metrics)
    core_hashes = {
        relative: sha256_file(source_root / relative)
        for relative in PADIM_CORE_FILES
    }
    provenance = {
        "method": "padim",
        "dataset": "mvtec15",
        "category": category,
        "source_kind": source["source_kind"],
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "official_implementation": False,
        "execution_command": " ".join(sys.argv),
        "environment": package_versions(),
        "checkpoint_source": (
            "ImageNet ResNet18 backbone plus category-specific Gaussian "
            "statistics fitted from MVTec train/good"
        ),
        "adapter_mode": "pinned_anomalib_core_files_without_lightning_cli",
        "core_source_sha256": core_hashes,
        "model_configuration": {
            "backbone": "resnet18",
            "layers": ["layer1", "layer2", "layer3"],
            "n_features": args.n_features,
            "model_seed": args.model_seed,
            "image_size": [256, 256],
            "gaussian_blur_sigma": 4,
        },
        "prediction_export": (
            "raw Anomalib PaDiM Mahalanobis maps after official sigma-4 blur "
            "without test-set min-max"
        ),
        "threshold_protocol": policy["protocol"],
        "uses_real_anomaly_labels_for_threshold": False,
        "uses_real_anomaly_masks_for_threshold": False,
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
    source = manifest["sources"]["padim"]
    source_root = Path(args.source_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    marker_path = source_root / ".lite_seer_source.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("commit") != source["commit"]:
        raise ValueError("Cached PaDiM source does not match the pinned commit")
    if not dataset_root.exists():
        raise FileNotFoundError(f"MVTec AD dataset is missing: {dataset_root}")
    categories = selected_categories(args.categories)
    seeds = split_ints(args.synthetic_seeds)
    model_class = load_padim_model_class(source_root)
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
                    model_class,
                )
            )
        except Exception as exc:
            failures.append({"category": category, "error": str(exc)})
            logging.exception("PaDiM failed for %s", category)
    report = {
        "method": "padim",
        "source_kind": source["source_kind"],
        "source_commit": source["commit"],
        "categories": categories,
        "completed": len(records),
        "failures": failures,
        "complete": not failures and len(records) == len(categories),
        "records": records,
    }
    report_path = Path(args.external_root) / "mvtec15" / "padim" / "run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
