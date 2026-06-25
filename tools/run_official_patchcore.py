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
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

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
from tools.audit_official_source_environments import parse_lfs_pointer
from tools.materialize_patchcore_pretrained import (
    DEFAULT_BUNDLE,
    selected_categories,
)


SYNTHETIC_MASK_MODES = ("blob", "scratch", "spot", "patch")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned author-official PatchCore implementation and export "
            "raw predictions plus label-free fixed-threshold evidence."
        )
    )
    parser.add_argument("--manifest", default="baselines/official_sources.json")
    parser.add_argument(
        "--source-root",
        default="third_party/official_baselines/patchcore",
    )
    parser.add_argument("--dataset-root", default="SEER-AD-dataset/MVTec-AD")
    parser.add_argument(
        "--external-root",
        default="baselines/external_outputs",
    )
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--categories", default="bottle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--faiss-workers", type=int, default=8)
    parser.add_argument("--synthetic-seeds", default="7,13,23")
    parser.add_argument("--max-normal-images", type=int, default=16)
    parser.add_argument("--synthetic-variants", type=int, default=4)
    parser.add_argument("--max-normal-fpr", type=float, default=0.005)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def split_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one synthetic seed is required")
    return values


def stable_seed(seed: int, category: str, path: str, variant: int) -> int:
    payload = f"{seed}:{category}:{path}:{variant}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def package_versions() -> dict[str, str]:
    packages = (
        "torch",
        "torchvision",
        "faiss-cpu",
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


def _load_official_modules(source_root: Path) -> tuple[Any, Any, Any]:
    source_path = str(source_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    try:
        import patchcore.common as common
        import patchcore.patchcore as patchcore_model
        from patchcore.datasets import mvtec
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Official PatchCore dependencies are incomplete. Install "
            "faiss-cpu before running this adapter."
        ) from exc
    return common, patchcore_model, mvtec


def _assert_model_materialized(model_dir: Path) -> None:
    required = (
        model_dir / "patchcore_params.pkl",
        model_dir / "nnscorer_search_index.faiss",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Official PatchCore model files are missing: " + ", ".join(missing)
        )
    pointers = [str(path) for path in required if parse_lfs_pointer(path)]
    if pointers:
        raise ValueError(
            "PatchCore model files are still Git LFS pointers. Run "
            "tools/materialize_patchcore_pretrained.py first: "
            + ", ".join(pointers)
        )


def _load_model(
    model_dir: Path,
    device: torch.device,
    common: Any,
    patchcore_model: Any,
    faiss_workers: int,
) -> Any:
    _assert_model_materialized(model_dir)
    nn_method = common.FaissNN(False, faiss_workers)
    model = patchcore_model.PatchCore(device)
    model.load_from_path(
        load_path=str(model_dir),
        device=device,
        nn_method=nn_method,
    )
    return model


def _score_batches(
    model: Any,
    images: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    heatmaps = []
    for start in range(0, len(images), max(1, batch_size)):
        batch = images[start : start + max(1, batch_size)]
        batch_scores, batch_heatmaps = model.predict(batch)
        scores.extend(batch_scores)
        heatmaps.extend(batch_heatmaps)
    return (
        np.asarray(scores, dtype=np.float32),
        np.asarray(heatmaps, dtype=np.float32),
    )


def _synthetic_evidence(
    model: Any,
    mvtec: Any,
    dataset_root: Path,
    category: str,
    seed: int,
    *,
    max_normal_images: int,
    synthetic_variants: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    train_dataset = mvtec.MVTecDataset(
        str(dataset_root),
        classname=category,
        resize=366,
        imagesize=320,
        split=mvtec.DatasetSplit.TRAIN,
    )
    normal_paths = [str(item[2]) for item in train_dataset.data_to_iterate]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(normal_paths))
    normal_paths = [
        normal_paths[index]
        for index in order[: min(len(order), max(1, max_normal_images))]
    ]
    spatial = transforms.Compose(
        [
            transforms.Resize(366),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
        ]
    )
    clean_items = []
    synthetic_items = []
    mask_items = []
    sample_paths = []
    variant_ids = []
    for path in normal_paths:
        clean_01 = spatial(Image.open(path).convert("RGB"))
        clean_pm1 = clean_01 * 2.0 - 1.0
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
            clean_items.append((clean_01 - IMAGENET_MEAN) / IMAGENET_STD)
            synthetic_01 = (synthetic_pm1 + 1.0) * 0.5
            synthetic_items.append(
                (synthetic_01 - IMAGENET_MEAN) / IMAGENET_STD
            )
            mask_items.append(mask[0])
            sample_paths.append(path)
            variant_ids.append(variant)
    clean = torch.stack(clean_items)
    synthetic = torch.stack(synthetic_items)
    clean_scores, clean_heatmaps = _score_batches(model, clean, batch_size)
    synthetic_scores, synthetic_heatmaps = _score_batches(
        model,
        synthetic,
        batch_size,
    )
    return {
        "clean_heatmaps": clean_heatmaps,
        "synthetic_heatmaps": synthetic_heatmaps,
        "synthetic_masks": np.asarray(mask_items, dtype=np.uint8),
        "clean_image_scores": clean_scores,
        "synthetic_image_scores": synthetic_scores,
        "paths": np.asarray(sample_paths),
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
    common: Any,
    patchcore_model: Any,
    mvtec: Any,
) -> dict[str, Any]:
    artifact_dir = (
        Path(args.external_root) / "mvtec15" / "patchcore" / category
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
        return {"category": category, "status": "cached", "out": str(artifact_dir)}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir = (
        source_root
        / "models"
        / args.bundle
        / "models"
        / f"mvtec_{category}"
    )
    device = torch.device(args.device)
    model = _load_model(
        model_dir,
        device,
        common,
        patchcore_model,
        args.faiss_workers,
    )
    test_dataset = mvtec.MVTecDataset(
        str(dataset_root),
        classname=category,
        resize=366,
        imagesize=320,
        split=mvtec.DatasetSplit.TEST,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    scores, heatmaps, labels, masks = model.predict(loader)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    labels_np = np.asarray(labels, dtype=np.uint8)
    scores_np = np.asarray(scores, dtype=np.float32)
    heatmaps_np = np.asarray(heatmaps, dtype=np.float32)
    masks_np = np.asarray(masks, dtype=np.uint8)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    paths = np.asarray([str(item[2]) for item in test_dataset.data_to_iterate])
    np.savez_compressed(
        prediction_path,
        labels=labels_np,
        image_scores=scores_np,
        masks=masks_np,
        heatmaps=heatmaps_np,
        paths=paths,
    )

    evidence = []
    for seed in seeds:
        seed_payload = _synthetic_evidence(
            model,
            mvtec,
            dataset_root,
            category,
            seed,
            max_normal_images=args.max_normal_images,
            synthetic_variants=args.synthetic_variants,
            batch_size=args.batch_size,
        )
        np.savez_compressed(
            artifact_dir / f"synthetic_validation_seed{seed}.npz",
            **seed_payload,
        )
        evidence.append(seed_payload)
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
            "method": "patchcore",
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
        labels_np,
        scores_np,
        masks_np,
        heatmaps_np,
        pixel_threshold=float(policy["threshold"]),
        threshold_protocol=str(policy["protocol"]),
    )
    metrics.update(
        {
            "method": "patchcore",
            "display_method": source["display_name"],
            "dataset": "mvtec15",
            "category": category,
            "official_implementation": True,
            "source_commit": source["commit"],
            "model_bundle": args.bundle,
            "latency_ms_mean": elapsed * 1000.0 / max(1, len(labels_np)),
            "prediction_count": int(len(labels_np)),
        }
    )
    _write_json(artifact_dir / "metrics.json", metrics)
    provenance = {
        "method": "patchcore",
        "dataset": "mvtec15",
        "category": category,
        "source_kind": source["source_kind"],
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "official_implementation": True,
        "execution_command": " ".join(sys.argv),
        "environment": package_versions(),
        "checkpoint_source": (
            f"Git LFS bundle {args.bundle} shipped by the pinned official repository"
        ),
        "model_bundle": args.bundle,
        "model_directory": str(model_dir),
        "prediction_export": "raw PatchCore distance scores without test-set min-max",
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
    source = manifest["sources"]["patchcore"]
    source_root = Path(args.source_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    marker_path = source_root / ".lite_seer_source.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("commit") != source["commit"]:
        raise ValueError("Cached PatchCore source does not match the pinned commit")
    if not dataset_root.exists():
        raise FileNotFoundError(f"MVTec AD dataset is missing: {dataset_root}")
    categories = selected_categories(args.categories)
    seeds = split_ints(args.synthetic_seeds)
    common, patchcore_model, mvtec = _load_official_modules(source_root)
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
                    common,
                    patchcore_model,
                    mvtec,
                )
            )
        except Exception as exc:
            failures.append({"category": category, "error": str(exc)})
            logging.exception("PatchCore failed for %s", category)
    report = {
        "method": "patchcore",
        "source_commit": source["commit"],
        "bundle": args.bundle,
        "categories": categories,
        "completed": len(records),
        "failures": failures,
        "complete": not failures and len(records) == len(categories),
        "records": records,
    }
    report_path = Path(args.external_root) / "mvtec15" / "patchcore" / "run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(report_path, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
