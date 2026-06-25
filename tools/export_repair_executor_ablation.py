"""Export clean-target repair executor ablations.

The protocol corrupts normal images with deterministic synthetic defects and
uses the original clean image as the target. It benchmarks non-diffusion repair
executors and records the missing evidence needed before claiming that
diffusion repair is necessary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.models.seer_ad_v2 import build_diffusion_components
from seer_ad_v2.utils.io import load_checkpoint


IMAGE_SIZE = (128, 128)
NON_DIFFUSION_EXECUTORS = [
    "identity_corrupted",
    "mean_fill",
    "neighbor_mean_inpaint",
    "partial_conv_inpaint",
    "nearest_normal_patch",
    "blur_inpaint",
    "light_ae_proxy_downsample",
    "trained_pca_light_ae",
    "light_unet_proxy_multiscale",
    "trained_linear_partial_conv",
]

DIFFUSION_EXECUTOR_NAME = "same_protocol_diffusion"


def stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def collect_normal_images(root: Path, images_per_category: int) -> list[dict[str, Any]]:
    patterns = [
        ("mvtec15", "SEER-AD-dataset/MVTec-AD/*/train/good/*"),
        ("visa", "SEER-AD-dataset/VisA/*/Data/Images/Normal/*"),
        ("mpdd", "SEER-AD-dataset/MPDD/official/MPDD/MPDD/*/train/good/*"),
    ]
    rows: list[dict[str, Any]] = []
    for dataset, pattern in patterns:
        by_category: dict[str, list[Path]] = {}
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            if dataset == "mvtec15":
                category = path.parts[-4]
            elif dataset == "visa":
                category = path.parts[-5]
            else:
                category = path.parts[-4]
            by_category.setdefault(category, []).append(path)
        for category, paths in sorted(by_category.items()):
            for path in paths[:images_per_category]:
                rows.append(
                    {
                        "dataset": dataset,
                        "category": category,
                        "path": path,
                    }
                )
    return rows


def image_to_array(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0


def synthetic_defect(clean: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray]:
    h, w = clean.shape[:2]
    rng = np.random.default_rng(stable_int(key))
    mask = np.zeros((h, w), dtype=bool)
    defect_kind = stable_int(key + ":kind") % 3
    if defect_kind == 0:
        bw = int(rng.integers(max(8, w // 12), max(12, w // 5)))
        bh = int(rng.integers(max(8, h // 12), max(12, h // 5)))
        x1 = int(rng.integers(0, max(1, w - bw)))
        y1 = int(rng.integers(0, max(1, h - bh)))
        mask[y1 : y1 + bh, x1 : x1 + bw] = True
    elif defect_kind == 1:
        y = int(rng.integers(h // 5, 4 * h // 5))
        thickness = int(rng.integers(3, 7))
        x0 = int(rng.integers(0, w // 3))
        x1 = int(rng.integers(2 * w // 3, w))
        for offset in range(-thickness, thickness + 1):
            yy = max(0, min(h - 1, y + offset))
            mask[yy, x0:x1] = True
    else:
        cx = int(rng.integers(w // 4, 3 * w // 4))
        cy = int(rng.integers(h // 4, 3 * h // 4))
        rx = int(rng.integers(max(6, w // 16), max(8, w // 8)))
        ry = int(rng.integers(max(6, h // 16), max(8, h // 8)))
        yy, xx = np.ogrid[:h, :w]
        mask[((xx - cx) / max(1, rx)) ** 2 + ((yy - cy) / max(1, ry)) ** 2 <= 1.0] = True

    corrupted = clean.copy()
    noise_color = rng.uniform(0.0, 1.0, size=(3,)).astype(np.float32)
    inverted = 1.0 - clean[mask]
    corrupted[mask] = 0.65 * inverted + 0.35 * noise_color
    return corrupted, mask


def dilate_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image = image.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    return np.asarray(image, dtype=np.uint8) > 0


def mean_fill(corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = corrupted.copy()
    bg = ~mask
    fill = corrupted[bg].mean(axis=0) if bg.any() else np.zeros(3, dtype=np.float32)
    out[mask] = fill
    return out


def neighbor_mean_inpaint(corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = corrupted.copy()
    ring = dilate_mask(mask, radius=3) & ~mask
    fill = corrupted[ring].mean(axis=0) if ring.any() else corrupted[~mask].mean(axis=0)
    out[mask] = fill
    return out


def nearest_normal_patch(corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = corrupted.copy()
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return out
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    ph, pw = y2 - y1, x2 - x1
    h, w = mask.shape
    ring = dilate_mask(mask, radius=2) & ~mask
    target_color = corrupted[ring].mean(axis=0) if ring.any() else corrupted[~mask].mean(axis=0)
    best_score = float("inf")
    best_patch = None
    step = max(4, min(ph, pw) // 2)
    for yy in range(0, max(1, h - ph + 1), step):
        for xx in range(0, max(1, w - pw + 1), step):
            candidate_mask = mask[yy : yy + ph, xx : xx + pw]
            if candidate_mask.any():
                continue
            patch = corrupted[yy : yy + ph, xx : xx + pw]
            score = float(np.mean((patch.mean(axis=(0, 1)) - target_color) ** 2))
            if score < best_score:
                best_score = score
                best_patch = patch
    if best_patch is not None:
        local_mask = mask[y1:y2, x1:x2]
        out[y1:y2, x1:x2][local_mask] = best_patch[local_mask]
    else:
        out = neighbor_mean_inpaint(corrupted, mask)
    return out


def partial_conv_inpaint(corrupted: np.ndarray, mask: np.ndarray, iterations: int = 18) -> np.ndarray:
    out = corrupted.copy()
    unknown = mask.copy()
    for _ in range(iterations):
        if not unknown.any():
            break
        padded_values = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_known = np.pad((~unknown).astype(np.float32), ((1, 1), (1, 1)), mode="constant")
        value_sum = np.zeros_like(out)
        weight_sum = np.zeros(mask.shape, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                known = padded_known[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
                value_sum += padded_values[dy : dy + mask.shape[0], dx : dx + mask.shape[1]] * known[..., None]
                weight_sum += known
        fillable = unknown & (weight_sum > 0)
        out[fillable] = value_sum[fillable] / weight_sum[fillable][:, None]
        unknown[fillable] = False
    if unknown.any():
        out[unknown] = neighbor_mean_inpaint(corrupted, mask)[unknown]
    return out


def blur_inpaint(corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.clip(corrupted * 255.0, 0, 255).astype(np.uint8))
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=5)), dtype=np.float32) / 255.0
    out = corrupted.copy()
    out[mask] = blurred[mask]
    return out


def light_ae_proxy_downsample(corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.clip(corrupted * 255.0, 0, 255).astype(np.uint8))
    small = image.resize((32, 32), Image.Resampling.BILINEAR)
    recon = np.asarray(small.resize(IMAGE_SIZE, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    out = corrupted.copy()
    out[mask] = recon[mask]
    return out


def light_unet_proxy_multiscale(corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.clip(corrupted * 255.0, 0, 255).astype(np.uint8))
    coarse = np.asarray(
        image.resize((32, 32), Image.Resampling.BILINEAR).resize(IMAGE_SIZE, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    mid = np.asarray(
        image.resize((64, 64), Image.Resampling.BILINEAR).resize(IMAGE_SIZE, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    local = partial_conv_inpaint(corrupted, mask, iterations=8)
    proxy = np.clip(0.25 * coarse + 0.35 * mid + 0.40 * local, 0.0, 1.0)
    out = corrupted.copy()
    out[mask] = proxy[mask]
    return out


def identity_corrupted(corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return corrupted.copy()


class TrainedPCALightAE:
    def __init__(self, mean: np.ndarray, components: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.components = components.astype(np.float32)

    def repair(self, corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
        flat = corrupted.reshape(1, -1).astype(np.float32)
        centered = flat - self.mean
        if self.components.size:
            code = centered @ self.components.T
            recon = self.mean + code @ self.components
        else:
            recon = self.mean
        recon_image = np.clip(recon.reshape(corrupted.shape), 0.0, 1.0)
        out = corrupted.copy()
        out[mask] = recon_image[mask]
        return out


def train_pca_light_ae(image_rows: list[dict[str, Any]], max_components: int = 8) -> TrainedPCALightAE:
    if not image_rows:
        mean = np.zeros((1, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3), dtype=np.float32)
        return TrainedPCALightAE(mean, np.zeros((0, mean.shape[1]), dtype=np.float32))
    data = np.stack(
        [image_to_array(Path(row["path"])).reshape(-1) for row in image_rows],
        axis=0,
    ).astype(np.float32)
    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean
    if data.shape[0] < 2 or not np.any(centered):
        return TrainedPCALightAE(mean, np.zeros((0, data.shape[1]), dtype=np.float32))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: max(1, min(max_components, data.shape[0] - 1))]
    return TrainedPCALightAE(mean, components)


def partial_conv_features(image: np.ndarray, known: np.ndarray, y: int, x: int) -> np.ndarray:
    values: list[float] = []
    flags: list[float] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy = min(max(y + dy, 0), image.shape[0] - 1)
            xx = min(max(x + dx, 0), image.shape[1] - 1)
            valid = bool(known[yy, xx])
            flags.append(1.0 if valid else 0.0)
            values.extend(image[yy, xx].tolist() if valid else [0.0, 0.0, 0.0])
    values.extend(flags)
    values.append(1.0)
    return np.asarray(values, dtype=np.float32)


class TrainedLinearPartialConv:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights.astype(np.float32)

    def repair(self, corrupted: np.ndarray, mask: np.ndarray, iterations: int = 18) -> np.ndarray:
        out = corrupted.copy()
        unknown = mask.copy()
        for _ in range(iterations):
            if not unknown.any():
                break
            known = ~unknown
            fillable = unknown & dilate_mask(known, radius=1)
            ys, xs = np.where(fillable)
            if len(xs) == 0:
                break
            for y, x in zip(ys, xs):
                features = partial_conv_features(out, known, int(y), int(x))
                out[y, x] = np.clip(features @ self.weights, 0.0, 1.0)
            unknown[fillable] = False
        if unknown.any():
            out[unknown] = partial_conv_inpaint(corrupted, mask)[unknown]
        return out


def train_linear_partial_conv(
    image_rows: list[dict[str, Any]],
    samples_per_image: int = 192,
    ridge: float = 1e-3,
) -> TrainedLinearPartialConv:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for row in image_rows:
        path = Path(row["path"])
        clean = image_to_array(path)
        _, mask = synthetic_defect(clean, path.as_posix() + ":partial-conv-train")
        known = ~mask
        candidates = np.argwhere(mask & dilate_mask(known, radius=1))
        if len(candidates) == 0:
            continue
        rng = np.random.default_rng(stable_int(path.as_posix() + ":partial-conv-samples"))
        take = min(samples_per_image, len(candidates))
        indices = rng.choice(len(candidates), size=take, replace=False)
        for idx in indices:
            y, x = candidates[int(idx)]
            features.append(partial_conv_features(clean, known, int(y), int(x)))
            targets.append(clean[int(y), int(x)].astype(np.float32))
    if not features:
        return TrainedLinearPartialConv(np.zeros((33, 3), dtype=np.float32))
    x = np.stack(features, axis=0).astype(np.float32)
    y = np.stack(targets, axis=0).astype(np.float32)
    xtx = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float32)
    xty = x.T @ y
    weights = np.linalg.solve(xtx, xty).astype(np.float32)
    return TrainedLinearPartialConv(weights)


EXECUTORS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "identity_corrupted": identity_corrupted,
    "mean_fill": mean_fill,
    "neighbor_mean_inpaint": neighbor_mean_inpaint,
    "partial_conv_inpaint": partial_conv_inpaint,
    "nearest_normal_patch": nearest_normal_patch,
    "blur_inpaint": blur_inpaint,
    "light_ae_proxy_downsample": light_ae_proxy_downsample,
    "light_unet_proxy_multiscale": light_unet_proxy_multiscale,
}


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


class DiffusionCheckpointExecutor:
    def __init__(self, checkpoint_path: Path, device: str = "cpu", steps: int = 10) -> None:
        import torch

        self.torch = torch
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = load_checkpoint(self.checkpoint_path, map_location=device)
        base_channels = int(checkpoint.get("base_channels", 32))
        timesteps = int(checkpoint.get("timesteps", 100))
        self.model, self.diffusion = build_diffusion_components(
            base_channels=base_channels,
            timesteps=timesteps,
            device=device,
        )
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()
        self.device = device
        self.steps = int(steps)

    def _pad_for_unet(self, tensor: Any) -> tuple[Any, int, int]:
        height, width = tensor.shape[-2:]
        target_h = max(4, int(height))
        target_w = max(4, int(width))
        pad_h = target_h - int(height)
        pad_w = target_w - int(width)
        if pad_h or pad_w:
            tensor = self.torch.nn.functional.pad(
                tensor,
                (0, pad_w, 0, pad_h),
                mode="replicate",
            )
        return tensor, int(height), int(width)

    def repair(self, corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray:
        box = mask_bbox(mask)
        if box is None:
            return corrupted.copy()
        y1, y2, x1, x2 = box
        crop = corrupted[y1:y2, x1:x2]
        tensor = self.torch.from_numpy(crop.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor * 2.0 - 1.0).to(self.device)
        tensor, crop_h, crop_w = self._pad_for_unet(tensor)
        with self.torch.no_grad():
            repaired = self.diffusion.reconstruct(self.model, tensor, steps=self.steps)
        repaired_np = (
            ((repaired.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 0.5)
            .clip(0.0, 1.0)
            .astype(np.float32)
        )
        repaired_np = repaired_np[:crop_h, :crop_w]
        out = corrupted.copy()
        local_mask = mask[y1:y2, x1:x2]
        out[y1:y2, x1:x2][local_mask] = repaired_np[local_mask]
        return out


class DiffusionCheckpointRegistry:
    def __init__(
        self,
        checkpoint: Path | None = None,
        checkpoint_root: Path | None = None,
        device: str = "cpu",
        steps: int = 10,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.checkpoint_root = Path(checkpoint_root) if checkpoint_root is not None else None
        self.device = device
        self.steps = int(steps)
        self._cache: dict[Path, DiffusionCheckpointExecutor] = {}

    def checkpoint_for(self, image_info: dict[str, Any]) -> Path | None:
        if self.checkpoint is not None:
            return self.checkpoint if self.checkpoint.is_file() else None
        if self.checkpoint_root is None:
            return None
        dataset = str(image_info["dataset"])
        category = str(image_info["category"])
        candidates = [
            self.checkpoint_root / dataset / category / "diffusion.pt",
            self.checkpoint_root / dataset / category / "default" / "diffusion.pt",
            self.checkpoint_root / category / "diffusion.pt",
            self.checkpoint_root / category / "default" / "diffusion.pt",
            self.checkpoint_root / f"{dataset}_{category}" / "diffusion.pt",
            self.checkpoint_root / f"feature_{dataset}_{category}_models" / "diffusion.pt",
            self.checkpoint_root / f"fulltest_{dataset}_{category}_models" / "diffusion.pt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def complete_category_coverage(self, image_rows: list[dict[str, Any]]) -> bool:
        if self.checkpoint_root is None or not image_rows:
            return False
        return all(self.checkpoint_for(row) is not None for row in image_rows)

    def __call__(self, image_info: dict[str, Any], corrupted: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        checkpoint_path = self.checkpoint_for(image_info)
        if checkpoint_path is None:
            return None
        resolved = checkpoint_path.resolve()
        executor = self._cache.get(resolved)
        if executor is None:
            executor = DiffusionCheckpointExecutor(resolved, device=self.device, steps=self.steps)
            self._cache[resolved] = executor
        return executor.repair(corrupted, mask)


class LpipsScorer:
    def __init__(self) -> None:
        import lpips
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = lpips.LPIPS(net="alex").to(self.device).eval()

    def __call__(self, clean: np.ndarray, repaired: np.ndarray) -> float:
        clean_t = self._tensor(clean)
        repaired_t = self._tensor(repaired)
        with self.torch.no_grad():
            value = self.model(clean_t, repaired_t)
        return float(value.detach().cpu().item())

    def _tensor(self, image: np.ndarray) -> Any:
        arr = np.clip(image, 0.0, 1.0).astype(np.float32)
        tensor = self.torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return (tensor * 2.0 - 1.0).to(self.device)


def make_lpips_scorer(enable_lpips: bool) -> Callable[[np.ndarray, np.ndarray], float] | None:
    if not enable_lpips:
        return None
    try:
        return LpipsScorer()
    except Exception:
        return None


def repair_metrics(
    clean: np.ndarray,
    repaired: np.ndarray,
    mask: np.ndarray,
    latency_ms: float,
    lpips_value: float | None = None,
) -> dict[str, float]:
    mask3 = mask[..., None]
    bg = ~mask
    fg_mae = float(np.mean(np.abs(clean[mask] - repaired[mask]))) if mask.any() else 0.0
    bg_mae = float(np.mean(np.abs(clean[bg] - repaired[bg]))) if bg.any() else 0.0
    corrupted_area = float(mask.mean())
    psnr = float(peak_signal_noise_ratio(clean, repaired, data_range=1.0))
    ssim = float(structural_similarity(clean, repaired, channel_axis=-1, data_range=1.0))
    local_l1 = float(np.mean(np.abs(clean * mask3 - repaired * mask3)))
    return {
        "psnr": psnr,
        "ssim": ssim,
        "foreground_mae": fg_mae,
        "background_mae": bg_mae,
        "lpips_available": lpips_value is not None,
        "lpips": lpips_value,
        "lpips_proxy_l1": local_l1,
        "latency_ms": latency_ms,
        "corrupted_area_ratio": corrupted_area,
    }


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, q))


def finite_mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not numeric:
        return None
    return float(np.mean(numeric))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["executor"]), []).append(row)
    summaries = []
    for method, method_rows in sorted(by_method.items()):
        latencies = [float(row["latency_ms"]) for row in method_rows]
        summaries.append(
            {
                "executor": method,
                "images": len(method_rows),
                "psnr_mean": finite_mean([row["psnr"] for row in method_rows]),
                "ssim_mean": finite_mean([row["ssim"] for row in method_rows]),
                "foreground_mae_mean": finite_mean([row["foreground_mae"] for row in method_rows]),
                "background_mae_mean": finite_mean([row["background_mae"] for row in method_rows]),
                "lpips_available": all(bool(row["lpips_available"]) for row in method_rows),
                "lpips_mean": finite_mean([row["lpips"] for row in method_rows]),
                "lpips_proxy_l1_mean": finite_mean([row["lpips_proxy_l1"] for row in method_rows]),
                "latency_mean_ms": finite_mean(latencies),
                "latency_p95_ms": quantile(latencies, 0.95),
                "corrupted_area_ratio_mean": finite_mean([row["corrupted_area_ratio"] for row in method_rows]),
            }
        )
    return summaries


def pareto_decision(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    diffusion = [row for row in summary_rows if row.get("executor") == DIFFUSION_EXECUTOR_NAME]
    non_diffusion = [row for row in summary_rows if row.get("executor") != DIFFUSION_EXECUTOR_NAME]
    if not diffusion:
        return {
            "ready": False,
            "decision": "missing_diffusion_executor",
            "best_non_diffusion_executor": None,
        }
    diffusion_row = diffusion[0]
    best_non_diffusion = max(
        non_diffusion,
        key=lambda row: (
            float(row["ssim_mean"] or 0.0),
            float(row["psnr_mean"] or 0.0),
            -float(row["latency_mean_ms"] or 0.0),
        ),
        default={},
    )
    lpips_ready = diffusion_row.get("lpips_mean") is not None and best_non_diffusion.get("lpips_mean") is not None
    if lpips_ready:
        quality_better = float(diffusion_row["lpips_mean"]) < float(best_non_diffusion["lpips_mean"])
    else:
        quality_better = float(diffusion_row["ssim_mean"] or 0.0) > float(best_non_diffusion.get("ssim_mean") or 0.0)
    latency_better = float(diffusion_row["latency_mean_ms"] or 0.0) <= float(best_non_diffusion.get("latency_mean_ms") or 0.0)
    if quality_better and latency_better:
        decision = "diffusion_pareto_better"
    elif quality_better:
        decision = "diffusion_quality_better_but_slower"
    else:
        decision = "diffusion_not_necessary"
    return {
        "ready": True,
        "decision": decision,
        "best_non_diffusion_executor": best_non_diffusion.get("executor"),
        "diffusion_lpips_mean": diffusion_row.get("lpips_mean"),
        "best_non_diffusion_lpips_mean": best_non_diffusion.get("lpips_mean"),
        "diffusion_ssim_mean": diffusion_row.get("ssim_mean"),
        "best_non_diffusion_ssim_mean": best_non_diffusion.get("ssim_mean"),
        "diffusion_latency_mean_ms": diffusion_row.get("latency_mean_ms"),
        "best_non_diffusion_latency_mean_ms": best_non_diffusion.get("latency_mean_ms"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_case(clean: np.ndarray, corrupted: np.ndarray, mask: np.ndarray, repairs: dict[str, np.ndarray], out_path: Path) -> None:
    methods = [
        "clean",
        "corrupted",
        "neighbor_mean_inpaint",
        "partial_conv_inpaint",
        "nearest_normal_patch",
        "light_unet_proxy_multiscale",
    ]
    tiles = []
    for method in methods:
        if method == "clean":
            arr = clean
        elif method == "corrupted":
            arr = corrupted
        else:
            arr = repairs[method]
        image = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).resize((96, 96))
        draw = ImageDraw.Draw(image)
        draw.text((3, 3), method, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
        tiles.append(image)
    panel = Image.new("RGB", (96 * len(tiles), 96), (0, 0, 0))
    for idx, tile in enumerate(tiles):
        panel.paste(tile, (idx * 96, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def build_ablation(
    root: Path,
    out_dir: Path | None = None,
    images_per_category: int = 1,
    enable_lpips: bool = False,
    lpips_scorer: Callable[[np.ndarray, np.ndarray], float] | None = None,
    diffusion_executor: Callable[[dict[str, Any], np.ndarray, np.ndarray], np.ndarray | None] | None = None,
    diffusion_coverage_scope: str = "none",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    image_rows = collect_normal_images(root, images_per_category)
    result_rows: list[dict[str, Any]] = []
    case_count = 0
    scorer = lpips_scorer if lpips_scorer is not None else make_lpips_scorer(enable_lpips)
    pca_light_ae = train_pca_light_ae(image_rows)
    linear_partial_conv = train_linear_partial_conv(image_rows)
    executors = dict(EXECUTORS)
    executors["trained_pca_light_ae"] = pca_light_ae.repair
    executors["trained_linear_partial_conv"] = linear_partial_conv.repair
    for image_info in image_rows:
        path = Path(image_info["path"])
        clean = image_to_array(path)
        corrupted, mask = synthetic_defect(clean, path.as_posix())
        repairs: dict[str, np.ndarray] = {}
        for name, executor in executors.items():
            start = time.perf_counter()
            repaired = executor(corrupted, mask)
            latency_ms = (time.perf_counter() - start) * 1000.0
            repairs[name] = repaired
            lpips_value = scorer(clean, repaired) if scorer is not None else None
            result_rows.append(
                {
                    "dataset": image_info["dataset"],
                    "category": image_info["category"],
                    "image_path": path.as_posix(),
                    "executor": name,
                    **repair_metrics(clean, repaired, mask, latency_ms, lpips_value),
                }
            )
        if diffusion_executor is not None:
            start = time.perf_counter()
            repaired = diffusion_executor(image_info, corrupted, mask)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if repaired is not None:
                repairs[DIFFUSION_EXECUTOR_NAME] = repaired
                lpips_value = scorer(clean, repaired) if scorer is not None else None
                result_rows.append(
                    {
                        "dataset": image_info["dataset"],
                        "category": image_info["category"],
                        "image_path": path.as_posix(),
                        "executor": DIFFUSION_EXECUTOR_NAME,
                        **repair_metrics(clean, repaired, mask, latency_ms, lpips_value),
                    }
                )
        if out_dir is not None and case_count < 6:
            render_case(
                clean,
                corrupted,
                mask,
                repairs,
                out_dir / "case_panels" / f"repair_executor_case_{case_count:02d}.png",
            )
            case_count += 1
    summary_rows = summarize(result_rows)
    lpips_ready = bool(result_rows) and all(bool(row["lpips_available"]) for row in result_rows)
    if lpips_scorer is not None and lpips_ready:
        lpips_protocol = "external_lpips_scorer"
    elif scorer is not None and lpips_ready:
        lpips_protocol = "lpips_alex_v0.1"
    else:
        lpips_protocol = "lpips_proxy_l1_only"
    diffusion_rows = [row for row in result_rows if row.get("executor") == DIFFUSION_EXECUTOR_NAME]
    diffusion_evaluated_all = bool(image_rows) and len(diffusion_rows) == len(image_rows)
    same_protocol_diffusion_ready = (
        diffusion_evaluated_all
        and diffusion_coverage_scope in {"complete_category_checkpoints", "injected_complete"}
    )
    pareto = pareto_decision(summary_rows)
    missing_release_evidence = []
    if not same_protocol_diffusion_ready:
        missing_release_evidence.append("same-protocol diffusion")
    if not (same_protocol_diffusion_ready and pareto["ready"]):
        missing_release_evidence.append("Pareto decision")
    if not lpips_ready:
        missing_release_evidence.append("LPIPS")
    best_non_diffusion = max(
        [row for row in summary_rows if row.get("executor") != DIFFUSION_EXECUTOR_NAME],
        key=lambda row: (
            float(row["ssim_mean"] or 0.0),
            float(row["psnr_mean"] or 0.0),
            -float(row["latency_mean_ms"] or 0.0),
        ),
        default={},
    )
    release_gate_passed = same_protocol_diffusion_ready and pareto["ready"] and lpips_ready
    summary = {
        "schema": "lite-seer-ad-repair-executor-ablation-v1",
        "evidence_level": "clean_target_non_diffusion_executor_ablation_v1",
        "release_gate_passed": release_gate_passed,
        "release_gate_reason": (
            "Diffusion necessity is answered under the same clean-target protocol."
            if release_gate_passed
            else (
                "Non-diffusion clean-target baselines are present, but "
                + ", ".join(missing_release_evidence)
                + " evidence is still missing."
            )
        ),
        "datasets": sorted({row["dataset"] for row in image_rows}),
        "categories": len({(row["dataset"], row["category"]) for row in image_rows}),
        "images": len(image_rows),
        "executors": NON_DIFFUSION_EXECUTORS
        + ([DIFFUSION_EXECUTOR_NAME] if diffusion_rows else []),
        "best_non_diffusion_executor": best_non_diffusion.get("executor"),
        "diffusion_evaluated_images": len(diffusion_rows),
        "diffusion_coverage_scope": diffusion_coverage_scope,
        "diffusion_pareto_decision": pareto,
        "executor_family_coverage": {
            "simple_inpainting_ready": True,
            "partial_conv_proxy_ready": True,
            "nearest_normal_patch_ready": True,
            "light_ae_proxy_ready": True,
            "light_unet_proxy_ready": True,
            "trained_light_ae_ready": True,
            "trained_light_unet_or_partial_conv_ready": True,
            "same_protocol_diffusion_ready": same_protocol_diffusion_ready,
            "lpips_metric_ready": lpips_ready,
        },
        "diffusion_executor_ready": same_protocol_diffusion_ready,
        "learned_light_ae_ready": True,
        "learned_light_unet_ready": False,
        "lpips_available": lpips_ready,
        "lpips_metric_protocol": lpips_protocol,
        "perceptual_proxy_metric": "lpips_proxy_l1",
        "before_after_visualizations": case_count,
        "required_for_release": [
            *(
                []
                if same_protocol_diffusion_ready
                else ["same-protocol diffusion repair executor on clean-target synthetic defects"]
            ),
            *([] if lpips_ready else ["LPIPS metric or documented replacement if LPIPS is unavailable"]),
            *(
                []
                if same_protocol_diffusion_ready and pareto["ready"]
                else ["Pareto decision: keep diffusion only if it is better than non-diffusion alternatives"]
            ),
        ],
    }
    return result_rows, summary_rows, summary


def executor_coverage_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    coverage = summary.get("executor_family_coverage", {}) or {}
    rows = [
        {
            "family": "simple_inpainting",
            "status": "pass" if coverage.get("simple_inpainting_ready") else "missing",
            "evidence": "mean_fill; neighbor_mean_inpaint; blur_inpaint",
            "required_for_100": True,
        },
        {
            "family": "partial_conv_inpainting",
            "status": "pass"
            if coverage.get("trained_light_unet_or_partial_conv_ready")
            else ("proxy" if coverage.get("partial_conv_proxy_ready") else "missing"),
            "evidence": "partial_conv_inpaint; trained_linear_partial_conv"
            if coverage.get("trained_light_unet_or_partial_conv_ready")
            else "partial_conv_inpaint",
            "required_for_100": True,
        },
        {
            "family": "nearest_normal_patch",
            "status": "pass" if coverage.get("nearest_normal_patch_ready") else "missing",
            "evidence": "nearest_normal_patch",
            "required_for_100": True,
        },
        {
            "family": "light_ae",
            "status": "pass"
            if coverage.get("trained_light_ae_ready")
            else ("proxy" if coverage.get("light_ae_proxy_ready") else "missing"),
            "evidence": "light_ae_proxy_downsample; trained_pca_light_ae"
            if coverage.get("trained_light_ae_ready")
            else "light_ae_proxy_downsample",
            "required_for_100": True,
        },
        {
            "family": "light_unet",
            "status": "proxy" if coverage.get("light_unet_proxy_ready") else "missing",
            "evidence": "light_unet_proxy_multiscale",
            "required_for_100": True,
        },
        {
            "family": "same_protocol_diffusion",
            "status": "pass" if coverage.get("same_protocol_diffusion_ready") else "missing",
            "evidence": "same_protocol_diffusion rows in table_repair_executor_images.csv"
            if coverage.get("same_protocol_diffusion_ready")
            else "",
            "required_for_100": True,
        },
        {
            "family": "lpips",
            "status": "pass" if coverage.get("lpips_metric_ready") else "proxy_only",
            "evidence": "lpips; lpips_proxy_l1" if coverage.get("lpips_metric_ready") else "lpips_proxy_l1",
            "required_for_100": True,
        },
    ]
    return rows


def write_outputs(
    root: Path,
    out_dir: Path,
    images_per_category: int = 1,
    enable_lpips: bool = False,
    diffusion_checkpoint: Path | None = None,
    diffusion_checkpoint_root: Path | None = None,
    diffusion_device: str = "cpu",
    diffusion_steps: int = 10,
) -> dict[str, Any]:
    diffusion_executor = None
    diffusion_coverage_scope = "none"
    if diffusion_checkpoint is not None or diffusion_checkpoint_root is not None:
        registry = DiffusionCheckpointRegistry(
            checkpoint=diffusion_checkpoint,
            checkpoint_root=diffusion_checkpoint_root,
            device=diffusion_device,
            steps=diffusion_steps,
        )
        diffusion_executor = registry
        image_rows = collect_normal_images(root, images_per_category)
        if diffusion_checkpoint_root is not None and registry.complete_category_coverage(image_rows):
            diffusion_coverage_scope = "complete_category_checkpoints"
        elif diffusion_checkpoint is not None:
            diffusion_coverage_scope = "single_checkpoint_smoke"
        else:
            diffusion_coverage_scope = "incomplete_category_checkpoints"
    result_rows, summary_rows, summary = build_ablation(
        root,
        out_dir,
        images_per_category,
        enable_lpips=enable_lpips,
        diffusion_executor=diffusion_executor,
        diffusion_coverage_scope=diffusion_coverage_scope,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_repair_executor_images.csv", result_rows)
    write_csv(out_dir / "table_repair_executor_summary.csv", summary_rows)
    write_csv(out_dir / "table_executor_family_coverage.csv", executor_coverage_rows(summary))
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tables/repair_executor_ablation"),
    )
    parser.add_argument("--images-per-category", type=int, default=1)
    parser.add_argument("--enable-lpips", action="store_true")
    parser.add_argument(
        "--diffusion-checkpoint",
        type=Path,
        default=None,
        help="Evaluate one trained Lite-SEER diffusion.pt checkpoint as a smoke executor.",
    )
    parser.add_argument(
        "--diffusion-checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Evaluate category checkpoints discovered as root/dataset/category/diffusion.pt "
            "or root/category/diffusion.pt. Complete coverage is required for release readiness."
        ),
    )
    parser.add_argument("--diffusion-device", default="cpu")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    args = parser.parse_args()
    if args.diffusion_checkpoint is not None and args.diffusion_checkpoint_root is not None:
        raise SystemExit("Use either --diffusion-checkpoint or --diffusion-checkpoint-root, not both.")
    summary = write_outputs(
        args.root,
        args.out_dir,
        args.images_per_category,
        enable_lpips=args.enable_lpips,
        diffusion_checkpoint=args.diffusion_checkpoint,
        diffusion_checkpoint_root=args.diffusion_checkpoint_root,
        diffusion_device=args.diffusion_device,
        diffusion_steps=args.diffusion_steps,
    )
    print(
        f"Wrote repair executor ablation for {summary['images']} images "
        f"to {args.out_dir} (release_gate_passed={summary['release_gate_passed']})"
    )


if __name__ == "__main__":
    main()
