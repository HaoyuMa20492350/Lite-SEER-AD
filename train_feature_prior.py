from __future__ import annotations

import argparse

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from seer_ad_v2.config import cfg_device, cfg_first, cfg_int, cfg_seed, dataset_category, image_size as cfg_image_size, load_config, make_run_dir, resolve_device
from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.models.feature_prior import build_extractor, build_feature_prior, parse_layers
from seer_ad_v2.utils.io import save_checkpoint, save_json
from seer_ad_v2.utils.run import save_run_metadata
from seer_ad_v2.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a frozen-feature normality prior for feature-first Lite-SEER-AD.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--category", default=None)
    p.add_argument("--method", choices=["patchcore", "padim"], default=None)
    p.add_argument("--backbone", choices=["wide_resnet50_2", "resnet18"], default=None)
    p.add_argument("--layers", default=None, help="Comma-separated feature layers. Defaults follow the backbone.")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Deterministically sample this many normal training images using --seed.",
    )
    p.add_argument(
        "--train-splits",
        default="train",
        help="Comma-separated normal-only splits, e.g. train,validation for MVTec AD 2.",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--coreset-ratio", type=float, default=None)
    p.add_argument("--max-bank-patches", type=int, default=None)
    p.add_argument("--padim-dim", type=int, default=None)
    p.add_argument("--retrieval-max-references", type=int, default=None)
    p.add_argument("--retrieval-patch-size", type=int, default=None)
    p.add_argument("--run-name", default="default")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--allow-random-weights", action="store_true", help="Only use for plumbing tests if ImageNet weights are unavailable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg_seed(cfg, args.seed)
    seed_everything(seed)
    category = dataset_category(cfg, args.category)
    image_size = cfg_image_size(cfg, args.image_size)
    batch_size = int(args.batch_size or cfg_int(cfg, ("feature_prior.batch_size", "training.batch_size"), 8))
    method = str(args.method or cfg_first(cfg, ("feature_prior.method",), "patchcore"))
    backbone = str(args.backbone or cfg_first(cfg, ("feature_prior.backbone",), "wide_resnet50_2"))
    layers = parse_layers(args.layers or cfg_first(cfg, ("feature_prior.layers",), None), backbone)
    coreset_ratio = float(args.coreset_ratio if args.coreset_ratio is not None else cfg_first(cfg, ("feature_prior.coreset_ratio",), 0.1))
    max_bank_patches = int(args.max_bank_patches if args.max_bank_patches is not None else cfg_first(cfg, ("feature_prior.max_bank_patches",), 20000))
    padim_dim = int(args.padim_dim if args.padim_dim is not None else cfg_first(cfg, ("feature_prior.padim_dim",), 100))
    retrieval_max_references = int(
        args.retrieval_max_references
        if args.retrieval_max_references is not None
        else cfg_first(cfg, ("feature_prior.retrieval.max_references",), 2048)
    )
    retrieval_patch_size = int(
        args.retrieval_patch_size
        if args.retrieval_patch_size is not None
        else cfg_first(cfg, ("feature_prior.retrieval.patch_size",), 64)
    )
    device = resolve_device(cfg_device(cfg, args.device))
    run_dir = make_run_dir(cfg, args.run_name)
    save_run_metadata(run_dir, cfg, args, device, "train_feature_prior")

    train_splits = [part.strip() for part in args.train_splits.split(",") if part.strip()]
    datasets = [
        build_dataset(
            cfg_first(cfg, ("dataset.name",), "mvtec"),
            cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD"),
            category,
            split,
            image_size,
            max_samples=None if len(train_splits) > 1 else (
                args.train_samples if args.train_samples is not None else args.max_samples
            ),
            sample_seed=seed if len(train_splits) == 1 and args.train_samples is not None else None,
        )
        for split in train_splits
    ]
    train_ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if len(train_splits) > 1 and args.train_samples is not None and len(train_ds) > args.train_samples:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_ds), generator=generator)[: args.train_samples].tolist()
        train_ds = Subset(train_ds, sorted(indices))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=cfg_int(cfg, ("dataset.num_workers",), 0))
    extractor = build_extractor(backbone, layers, device, allow_random_weights=args.allow_random_weights)
    prior_state = build_feature_prior(
        extractor,
        layers,
        loader,
        device,
        method=method,
        coreset_ratio=coreset_ratio,
        max_bank_patches=max_bank_patches,
        padim_dim=padim_dim,
        seed=seed,
        retrieval_max_references=retrieval_max_references,
        retrieval_patch_size=retrieval_patch_size,
    )
    save_checkpoint(
        run_dir / "feature_prior.pt",
        prior_state=prior_state,
        cfg=cfg,
        category=category,
        image_size=image_size,
        method=method,
        backbone=backbone,
        layers=layers,
        coreset_ratio=coreset_ratio,
        max_bank_patches=max_bank_patches,
        padim_dim=padim_dim,
        retrieval_max_references=retrieval_max_references,
        retrieval_patch_size=retrieval_patch_size,
        seed=seed,
    )
    size_info = {
        key: int(value.numel()) if isinstance(value, torch.Tensor) else value
        for key, value in prior_state.items()
        if key in {"bank", "selected", "mean", "inv_cov", "retrieval_features", "retrieval_patches"}
    }
    save_json(
        {
            "category": category,
            "method": method,
            "backbone": backbone,
            "layers": layers,
            "image_size": image_size,
            "train_images": len(train_ds),
            "train_sample_seed": seed if args.train_samples is not None else None,
            "train_splits": train_splits,
            "retrieval_max_references": retrieval_max_references,
            "retrieval_patch_size": retrieval_patch_size,
            "size_info": size_info,
        },
        run_dir / "feature_prior_metrics.json",
    )
    print(f"Saved feature prior checkpoint to {run_dir / 'feature_prior.pt'}")


if __name__ == "__main__":
    main()
