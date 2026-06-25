from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.config import cfg_device, cfg_first, cfg_int, load_config, resolve_device
from seer_ad_v2.data.datasets import MVTEC_AD2_CATEGORIES, build_dataset
from seer_ad_v2.evaluation.mvtec_ad2_submission import (
    assert_export_root_available,
    assert_path_outside_submission,
    assert_submission_root,
    create_submission_archive,
    default_metadata_dir,
    run_official_checker,
    sha256_file,
)
from seer_ad_v2.models.feature_prior import feature_prior_scores, load_feature_prior_components
from seer_ad_v2.utils.io import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export an official-format MVTec AD 2 private-test submission.")
    p.add_argument("--config", default="configs/mvtec_ad2.yaml")
    p.add_argument("--checkpoint", required=True, help="Checkpoint template containing {category}.")
    p.add_argument("--categories", default="all")
    p.add_argument("--splits", default="test_private,test_private_mixed")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold-sigma", type=float, default=3.0)
    p.add_argument(
        "--submission-resolution",
        choices=("native", "model"),
        default="native",
        help=(
            "Write maps at source-image resolution or at the model output "
            "resolution. The official server bilinearly upsamples model-size maps."
        ),
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--metadata-out",
        default=None,
        help="Metadata directory outside the checker-visible submission root.",
    )
    p.add_argument(
        "--official-checker",
        default=(
            "official_mvtec_ad2_utils/MVTecAD2_public_code_utils/"
            "check_and_prepare_data_for_upload.py"
        ),
    )
    p.add_argument("--skip-official-check", action="store_true")
    p.add_argument(
        "--archive-out",
        default=None,
        help="Optional .tar.gz path, created only after the official checker passes.",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def prepare_submission_map(
    heatmap: np.ndarray,
    source_size: tuple[int, int],
    submission_resolution: str,
) -> np.ndarray:
    heatmap = heatmap.astype(np.float32, copy=False)
    if submission_resolution == "native":
        output = cv2.resize(
            heatmap,
            source_size,
            interpolation=cv2.INTER_LINEAR,
        )
    elif submission_resolution == "model":
        output = heatmap
    else:
        raise ValueError(
            f"Unsupported submission resolution: {submission_resolution}"
        )
    return np.nan_to_num(
        output,
        nan=0.0,
        posinf=np.finfo(np.float16).max,
        neginf=np.finfo(np.float16).min,
    )


def heatmaps_for_split(
    cfg: dict[str, Any],
    category: str,
    split: str,
    checkpoint_template: str,
    image_size: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    checkpoint_path = Path(checkpoint_template.format(category=category))
    checkpoint = load_checkpoint(checkpoint_path)
    prior, extractor, layers = load_feature_prior_components(checkpoint, device)
    dataset = build_dataset(
        cfg_first(cfg, ("dataset.name",), "mvtec_ad2"),
        cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD2"),
        category,
        split,
        image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg_int(cfg, ("dataset.num_workers",), 0),
    )
    chunks = []
    paths: list[str] = []
    for batch in loader:
        output = feature_prior_scores(
            prior,
            extractor,
            layers,
            batch["image"].to(device),
            device,
            image_size,
        )
        chunks.append(output.raw_heatmaps.astype(np.float32))
        paths.extend(str(path) for path in batch["path"])
    return np.concatenate(chunks, axis=0), paths


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg_device(cfg, args.device))
    categories = MVTEC_AD2_CATEGORIES if args.categories == "all" else split_csv(args.categories)
    splits = split_csv(args.splits)
    dataset_root = Path(str(cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD2"))).resolve()
    out_dir = Path(args.out)
    assert_export_root_available(out_dir)
    metadata_dir = (
        Path(args.metadata_out)
        if args.metadata_out
        else default_metadata_dir(out_dir)
    )
    assert_path_outside_submission(
        out_dir,
        metadata_dir,
        "Submission metadata",
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for category in categories:
        validation_maps, _validation_paths = heatmaps_for_split(
            cfg,
            category,
            "validation",
            args.checkpoint,
            args.image_size,
            args.batch_size,
            device,
        )
        threshold = float(
            np.mean(validation_maps)
            + args.threshold_sigma * np.std(validation_maps)
        )
        for split in splits:
            maps, paths = heatmaps_for_split(
                cfg,
                category,
                split,
                args.checkpoint,
                args.image_size,
                args.batch_size,
                device,
            )
            for heatmap, source_path in zip(maps, paths):
                source = Path(source_path)
                with Image.open(source) as image:
                    width, height = image.size
                submission_map = prepare_submission_map(
                    heatmap,
                    (width, height),
                    args.submission_resolution,
                )
                relative = source.resolve().relative_to(dataset_root)
                continuous_path = (
                    out_dir
                    / "anomaly_images"
                    / relative
                ).with_suffix(".tiff")
                threshold_path = (
                    out_dir
                    / "anomaly_images_thresholded"
                    / relative
                ).with_suffix(".png")
                continuous_path.parent.mkdir(parents=True, exist_ok=True)
                threshold_path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(
                    continuous_path,
                    submission_map.astype(np.float16),
                )
                binary = (submission_map >= threshold).astype(np.uint8) * 255
                Image.fromarray(binary, mode="L").save(threshold_path)
                output_height, output_width = submission_map.shape
                manifest_rows.append(
                    {
                        "category": category,
                        "split": split,
                        "source": str(source),
                        "continuous": str(continuous_path),
                        "thresholded": str(threshold_path),
                        "threshold": threshold,
                        "source_height": height,
                        "source_width": width,
                        "output_height": output_height,
                        "output_width": output_width,
                    }
                )

    assert_submission_root(out_dir)
    expected_categories = set(MVTEC_AD2_CATEGORIES)
    expected_splits = {"test_private", "test_private_mixed"}
    full_submission = (
        set(categories) == expected_categories and set(splits) == expected_splits
    )
    checker_result: dict[str, Any]
    checker_path = Path(args.official_checker)
    if args.skip_official_check:
        checker_result = {"status": "skipped_by_user"}
    elif not full_submission:
        checker_result = {"status": "not_run_partial_export"}
    elif not checker_path.is_file():
        checker_result = {
            "status": "not_run_checker_missing",
            "checker": str(checker_path),
        }
    else:
        checker_result = run_official_checker(out_dir, checker_path)

    archive_path = None
    archive_size_bytes = None
    archive_sha256 = None
    if args.archive_out:
        if checker_result.get("status") != "passed":
            raise RuntimeError(
                "Refusing to create an upload archive before the official "
                f"checker passes; checker status={checker_result.get('status')}"
            )
        created_archive = create_submission_archive(
            out_dir,
            Path(args.archive_out),
        )
        archive_path = str(created_archive)
        archive_size_bytes = created_archive.stat().st_size
        archive_sha256 = sha256_file(created_archive)

    with (metadata_dir / "submission_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "split",
                "source",
                "continuous",
                "thresholded",
                "threshold",
                "source_height",
                "source_width",
                "output_height",
                "output_width",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    protocol = {
        "dataset": "mvtec_ad2",
        "splits": splits,
        "categories": categories,
        "continuous_format": "single_channel_float16_tiff",
        "thresholded_format": "single_channel_uint8_png_0_255",
        "threshold_source": "validation_normal_mean_plus_sigma_std",
        "threshold_sigma": args.threshold_sigma,
        "submission_resolution": args.submission_resolution,
        "model_output_size": [args.image_size, args.image_size],
        "files": len(manifest_rows),
        "submission_root": str(out_dir.resolve()),
        "metadata_root": str(metadata_dir.resolve()),
        "submission_root_entries": sorted(
            path.name for path in out_dir.iterdir()
        ),
        "full_official_submission": full_submission,
        "official_checker": checker_result,
        "archive": archive_path,
        "archive_size_bytes": archive_size_bytes,
        "archive_sha256": archive_sha256,
    }
    (metadata_dir / "submission_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
