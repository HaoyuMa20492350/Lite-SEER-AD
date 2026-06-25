from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from seer_ad_v2.config import (
    cfg_device,
    cfg_first,
    cfg_float,
    cfg_int,
    cfg_seed,
    dataset_category,
    image_size as cfg_image_size,
    load_config,
    make_run_dir,
    patch_size as cfg_patch_size,
    resolve_device,
)
from seer_ad_v2.data.datasets import DTDTextureDataset, build_dataset
from seer_ad_v2.data.defect_synthesis import synthesize_anomaly
from seer_ad_v2.data.hard_negative_mining import ROI, crop_resize
from seer_ad_v2.models.diffusion.reconstruction import residual_heatmap
from seer_ad_v2.models.feature_prior import feature_prior_scores, load_feature_prior_components
from seer_ad_v2.models.region_verifier.hn_sev import HNSEV, binary_focal_bce, build_sev_input
from seer_ad_v2.models.region_verifier.prototype_bank import PrototypeBank
from seer_ad_v2.models.seer_ad_v2 import build_diffusion_components
from seer_ad_v2.utils.io import load_checkpoint, save_checkpoint, save_json
from seer_ad_v2.utils.run import save_run_metadata
from seer_ad_v2.utils.seed import seed_everything


def read_manifest(hard_negative_dir: str | Path) -> list[Path]:
    manifest = Path(hard_negative_dir) / "manifest.csv"
    if not manifest.exists():
        return []
    paths: list[Path] = []
    with manifest.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paths.append(Path(row["path"]))
    return paths


class HNSEVDataset(Dataset):
    def __init__(
        self,
        normal_dataset,
        hard_paths: list[Path],
        patch_size: int,
        texture_root: str | Path | None = None,
        prototype_bank: PrototypeBank | None = None,
        feature_prior: dict | None = None,
        feature_extractor: torch.nn.Module | None = None,
        feature_layers: list[str] | None = None,
        feature_device: str = "cpu",
        include_clean_normal: bool = True,
        include_hard_negative: bool = True,
    ) -> None:
        self.normal_dataset = normal_dataset
        self.hard_paths = hard_paths
        self.patch_size = patch_size
        self.texture_bank = DTDTextureDataset(texture_root) if texture_root else None
        self.prototype_bank = prototype_bank
        self.feature_prior = feature_prior
        self.feature_extractor = feature_extractor
        self.feature_layers = feature_layers or []
        self.feature_device = feature_device
        self.modes = ["synthetic"]
        if include_clean_normal:
            self.modes.append("clean")
        if include_hard_negative and hard_paths:
            self.modes.append("hard")
        self.length = max(len(normal_dataset) * len(self.modes), len(hard_paths), 1)

    def __len__(self) -> int:
        return self.length

    def _normal_patch(self, image: torch.Tensor) -> torch.Tensor:
        _, h, w = image.shape
        ps = min(self.patch_size, h, w)
        y = np.random.randint(0, max(1, h - ps + 1))
        x = np.random.randint(0, max(1, w - ps + 1))
        patch = image[:, y : y + ps, x : x + ps].unsqueeze(0)
        return F.interpolate(patch, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False)[0]

    def _hard_negative(self, path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = np.load(path, allow_pickle=True)
        orig = torch.from_numpy(item["original"]).float()
        rec = torch.from_numpy(item["reconstruction"]).float()
        residual = torch.from_numpy(item["residual"]).float().unsqueeze(0)
        vals = item["roi"].astype(np.float32).tolist()
        roi = ROI(int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]), int(vals[4]), float(vals[5]))
        o = crop_resize(orig, roi, self.patch_size)
        r = crop_resize(rec, roi, self.patch_size)
        h = crop_resize(residual, roi, self.patch_size)
        return o, r, h

    def _prototype_signal(self, patch: torch.Tensor) -> torch.Tensor | None:
        if self.prototype_bank is None:
            return None
        return self.prototype_bank.novelty(patch.unsqueeze(0))

    def _feature_signal(self, patch: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.feature_prior is None or self.feature_extractor is None or not self.feature_layers:
            return None, None
        with torch.no_grad():
            out = feature_prior_scores(
                self.feature_prior,
                self.feature_extractor,
                self.feature_layers,
                patch.unsqueeze(0).to(self.feature_device),
                self.feature_device,
                self.patch_size,
            )
        feature_patch = torch.from_numpy(out.heatmaps[0]).float().unsqueeze(0)
        feature_gap = torch.tensor([float(out.cosine_heatmaps[0].mean())], dtype=torch.float32)
        return feature_patch, feature_gap

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.modes[idx % len(self.modes)]
        if mode == "hard" and self.hard_paths:
            o, r, h = self._hard_negative(self.hard_paths[idx % len(self.hard_paths)])
            feature_patch, feature_gap = self._feature_signal(o)
            x = build_sev_input(
                o.unsqueeze(0),
                r.unsqueeze(0),
                h.unsqueeze(0),
                self._prototype_signal(o),
                None if feature_patch is None else feature_patch.unsqueeze(0),
                feature_gap,
            )[0]
            return x, torch.tensor(0.0)

        image = self.normal_dataset[idx % len(self.normal_dataset)]["image"]
        clean = self._normal_patch(image)
        if mode == "synthetic":
            corrupt, mask = synthesize_anomaly(clean, self.texture_bank)
            residual = mask
            feature_patch, feature_gap = self._feature_signal(corrupt)
            x = build_sev_input(
                corrupt.unsqueeze(0),
                clean.unsqueeze(0),
                residual.unsqueeze(0),
                self._prototype_signal(corrupt),
                None if feature_patch is None else feature_patch.unsqueeze(0),
                feature_gap,
            )[0]
            return x, torch.tensor(1.0)

        residual = torch.zeros(1, self.patch_size, self.patch_size)
        feature_patch, feature_gap = self._feature_signal(clean)
        x = build_sev_input(
            clean.unsqueeze(0),
            clean.unsqueeze(0),
            residual.unsqueeze(0),
            self._prototype_signal(clean),
            None if feature_patch is None else feature_patch.unsqueeze(0),
            feature_gap,
        )[0]
        return x, torch.tensor(0.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hard-negative-aware semantic verifier.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--category", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hard-negative-dir", default=None)
    p.add_argument("--synthetic-only", action="store_true", help="Train SEV without mined normal hard negatives.")
    p.add_argument("--disable-clean-normal", action="store_true", help="Train without explicit clean-normal negative patches.")
    p.add_argument("--disable-prototype", action="store_true", help="Train with the prototype channel fixed to zero.")
    p.add_argument(
        "--input-ablation-label",
        choices=[
            "synthetic_only_sev",
            "clean_normal_sev",
            "hard_negative_sev",
            "with_clean_normal",
            "with_hard_negative",
            "with_prototype",
            "no_prototype",
        ],
        default=None,
        help="Explicit HN-SEV input-ablation label for downstream audit tables.",
    )
    p.add_argument("--feature-prior-checkpoint", default=None, help="Optional feature prior checkpoint for multi-view feature HN-SEV.")
    p.add_argument(
        "--texture-root",
        default=None,
        help="DTD texture directory. Defaults to hn_sev.texture_root in the config.",
    )
    p.add_argument("--allow-random-feature-weights", action="store_true", help="Only use for smoke tests if pretrained feature weights are unavailable.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--run-name", default="default")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def build_prototype_bank(
    normal_dataset,
    hard_paths: list[Path],
    patch_size: int,
    bank_size: int,
) -> tuple[PrototypeBank, int, int]:
    bank = PrototypeBank()
    if bank_size <= 0:
        return bank, 0, 0
    tmp_ds = HNSEVDataset(normal_dataset, [], patch_size=patch_size, texture_root=None, prototype_bank=None)
    hard_budget = min(len(hard_paths), max(0, bank_size // 3))
    normal_budget = max(len(normal_dataset), bank_size - hard_budget)
    normal_patches = []
    for i in range(normal_budget):
        normal_patches.append(tmp_ds._normal_patch(normal_dataset[i % len(normal_dataset)]["image"]))
    if normal_patches:
        bank.fit(torch.stack(normal_patches))

    hard_patches = []
    for path in hard_paths[:hard_budget]:
        hard_patches.append(tmp_ds._hard_negative(path)[0])
    if hard_patches:
        bank.append(torch.stack(hard_patches))
    return bank, len(normal_patches), len(hard_patches)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg_seed(cfg, args.seed))
    ckpt = load_checkpoint(args.checkpoint)
    category = args.category or ckpt.get("category") or dataset_category(cfg)
    image_size = cfg_image_size(cfg, args.image_size, ckpt.get("image_size"))
    patch_size = cfg_patch_size(cfg)
    epochs = args.epochs or cfg_int(cfg, ("training.epochs",), 20)
    batch_size = args.batch_size or cfg_int(cfg, ("training.batch_size",), 8)
    device = resolve_device(cfg_device(cfg, args.device))
    run_dir = make_run_dir(cfg, args.run_name)
    save_run_metadata(run_dir, cfg, args, device, "train_hn_sev")

    normal_dataset = build_dataset(
        cfg_first(cfg, ("dataset.name",), "mvtec"),
        cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD"),
        category,
        "train",
        image_size,
        max_samples=args.max_samples,
    )
    texture_root = Path(
        args.texture_root
        or cfg_first(
            cfg,
            ("hn_sev.texture_root",),
            "SEER-AD-dataset/DTD/Describable-Textures-Dataset-DTD/images",
        )
    )
    texture_bank_size = len(DTDTextureDataset(texture_root).paths)
    if texture_bank_size == 0:
        raise FileNotFoundError(f"No DTD texture images found at {texture_root}")
    hard_paths = [] if args.synthetic_only or not args.hard_negative_dir else read_manifest(args.hard_negative_dir)
    bank = PrototypeBank()
    prototype_normal_patches = 0
    prototype_hard_patches = 0
    if not args.disable_prototype:
        bank, prototype_normal_patches, prototype_hard_patches = build_prototype_bank(
            normal_dataset,
            hard_paths,
            patch_size,
            cfg_int(cfg, ("hn_sev.prototype_bank_size",), 1000),
        )
    feature_prior = None
    feature_extractor = None
    feature_layers: list[str] = []
    if args.feature_prior_checkpoint:
        feature_ckpt = load_checkpoint(args.feature_prior_checkpoint)
        feature_prior, feature_extractor, feature_layers = load_feature_prior_components(
            feature_ckpt,
            device,
            allow_random_weights=args.allow_random_feature_weights,
        )
    input_sources = {
        "synthetic_positive": True,
        "clean_normal": not bool(args.disable_clean_normal),
        "hard_negative": bool(hard_paths),
        "prototype": not bool(args.disable_prototype),
        "feature_prior": bool(args.feature_prior_checkpoint),
    }
    input_ablation_label = args.input_ablation_label
    if input_ablation_label is None:
        if args.synthetic_only:
            input_ablation_label = "synthetic_only_sev"
        elif args.disable_prototype:
            input_ablation_label = "no_prototype"
    ds = HNSEVDataset(
        normal_dataset,
        hard_paths,
        patch_size=patch_size,
        texture_root=texture_root,
        prototype_bank=None if args.disable_prototype else bank,
        feature_prior=feature_prior,
        feature_extractor=feature_extractor,
        feature_layers=feature_layers,
        feature_device=device,
        include_clean_normal=not bool(args.disable_clean_normal),
        include_hard_negative=bool(hard_paths),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    sample_x, _ = ds[0]
    in_channels = int(sample_x.shape[0])
    model = HNSEV(in_channels=in_channels, base_channels=cfg_int(cfg, ("hn_sev.base_channels", "sev.base_channels"), 24)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg_first(cfg, ("training.lr",), 2e-4)), weight_decay=1e-4)
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for x, y in tqdm(loader, desc=f"hn-sev epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = binary_focal_bce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))

    model.eval()
    calibration_scores: list[torch.Tensor] = []
    calibration_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
            logits = model(x.to(device))
            calibration_scores.append(torch.sigmoid(logits).detach().cpu())
            calibration_labels.append(y.detach().cpu().float())
    if calibration_scores:
        score_tensor = torch.cat(calibration_scores)
        label_tensor = torch.cat(calibration_labels)
        negative_scores = score_tensor[label_tensor < 0.5]
        positive_scores = score_tensor[label_tensor >= 0.5]
        negative_q95 = float(torch.quantile(negative_scores, 0.95)) if len(negative_scores) else 0.5
        positive_median = float(torch.median(positive_scores)) if len(positive_scores) else negative_q95
        positive_q75 = float(torch.quantile(positive_scores, 0.75)) if len(positive_scores) else positive_median
        threshold_floor = cfg_float(cfg, ("hn_sev.operating_threshold_floor",), 0.9)
        operating_threshold = max(threshold_floor, negative_q95, positive_q75)
    else:
        threshold_floor = cfg_float(cfg, ("hn_sev.operating_threshold_floor",), 0.9)
        operating_threshold = threshold_floor
        negative_q95 = 0.5
        positive_median = 0.5
        positive_q75 = 0.5

    save_checkpoint(
        run_dir / "hn_sev.pt",
        model_state=model.state_dict(),
        prototype_bank=bank.state_dict() if not args.disable_prototype else {},
        cfg=cfg,
        category=category,
        patch_size=patch_size,
        base_channels=cfg_int(cfg, ("hn_sev.base_channels", "sev.base_channels"), 24),
        in_channels=in_channels,
        synthetic_only=bool(args.synthetic_only),
        clean_normal_enabled=not bool(args.disable_clean_normal),
        prototype_enabled=not bool(args.disable_prototype),
        feature_prior_enabled=bool(args.feature_prior_checkpoint),
        feature_prior_checkpoint=args.feature_prior_checkpoint,
        hn_sev_input_ablation=input_ablation_label,
        input_sources=input_sources,
        texture_root=str(texture_root),
        texture_images=texture_bank_size,
        normal_train_images=len(normal_dataset),
        operating_threshold=operating_threshold,
    )
    save_json(
        {
            "loss": losses,
            "hard_negatives": len(hard_paths),
            "synthetic_only": bool(args.synthetic_only),
            "clean_normal_enabled": not bool(args.disable_clean_normal),
            "prototype_enabled": not bool(args.disable_prototype),
            "hn_sev_input_ablation": input_ablation_label,
            "input_sources": input_sources,
            "prototype_bank_size": int(len(bank.features) if bank.features is not None else 0),
            "prototype_normal_patches": prototype_normal_patches,
            "prototype_hard_negative_patches": prototype_hard_patches,
            "feature_prior_enabled": bool(args.feature_prior_checkpoint),
            "texture_root": str(texture_root),
            "texture_images": texture_bank_size,
            "normal_train_images": len(normal_dataset),
            "in_channels": in_channels,
            "operating_threshold": operating_threshold,
            "operating_threshold_floor": threshold_floor,
            "negative_score_q95": negative_q95,
            "positive_score_median": positive_median,
            "positive_score_q75": positive_q75,
        },
        run_dir / "hn_sev_metrics.json",
    )
    print(f"Saved HN-SEV checkpoint to {run_dir / 'hn_sev.pt'}")


if __name__ == "__main__":
    main()
