from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.config import (
    cfg_device,
    cfg_first,
    cfg_int,
    cfg_seed,
    dataset_category,
    image_size as cfg_image_size,
    load_config,
    resolve_device,
)
from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.models.feature_prior import build_retrieval_reference_bank, load_feature_prior_components
from seer_ad_v2.utils.io import load_checkpoint, save_json
from seer_ad_v2.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add a portable normal-patch retrieval bank to an existing feature prior.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--category", default=None)
    p.add_argument("--feature-prior-checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-references", type=int, default=2048)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--allow-random-feature-weights", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg_seed(cfg, args.seed)
    seed_everything(seed)
    checkpoint = load_checkpoint(args.feature_prior_checkpoint)
    category = str(args.category or checkpoint.get("category") or dataset_category(cfg))
    image_size = cfg_image_size(cfg, args.image_size, checkpoint.get("image_size"))
    device = resolve_device(cfg_device(cfg, args.device))
    batch_size = int(args.batch_size or cfg_int(cfg, ("feature_prior.batch_size", "training.batch_size"), 8))
    checkpoint, extractor, layers = load_feature_prior_components(
        checkpoint,
        device,
        allow_random_weights=args.allow_random_feature_weights,
    )
    train_dataset = build_dataset(
        cfg_first(cfg, ("dataset.name",), "mvtec"),
        cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD"),
        category,
        "train",
        image_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg_int(cfg, ("dataset.num_workers",), 0),
    )
    retrieval_state = build_retrieval_reference_bank(
        extractor,
        layers,
        loader,
        device,
        max_references=int(args.max_references),
        patch_size=int(args.patch_size),
        seed=seed,
    )
    prior_state = checkpoint.get("prior_state", checkpoint)
    prior_state = dict(prior_state)
    prior_state.update(retrieval_state)
    payload = dict(checkpoint)
    payload["prior_state"] = prior_state
    payload["retrieval_augmented"] = True
    payload["retrieval_max_references"] = int(args.max_references)
    payload["retrieval_patch_size"] = int(args.patch_size)
    payload["retrieval_seed"] = seed
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    metrics = {
        "source_checkpoint": str(args.feature_prior_checkpoint),
        "output_checkpoint": str(out),
        "category": category,
        "image_size": image_size,
        "train_images": len(train_dataset),
        "retrieval_references": int(retrieval_state.get("retrieval_reference_count", 0)),
        "retrieval_patch_size": int(args.patch_size),
        "selection_data": "train_normal_only",
        "uses_real_anomaly_labels": False,
        "uses_real_anomaly_masks": False,
    }
    save_json(metrics, out.with_suffix(".json"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
