from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


FEATURE_MAP_SOURCES = {
    "feature",
    "feature_raw",
    "feature_raw_distance",
    "feature_raw_cosine",
}
MULTISCALE_STATS_NAME = "multiscale_normal_stats.npz"


def feature_output_map(output: Any, source: str) -> np.ndarray:
    if source == "feature":
        return output.heatmaps
    if source == "feature_raw":
        return output.raw_heatmaps
    if source == "feature_raw_distance":
        return output.raw_distance_heatmaps
    if source == "feature_raw_cosine":
        return output.raw_cosine_heatmaps
    raise ValueError(f"Unsupported feature map source: {source}")


def calibrate_maps(maps: np.ndarray, stats: dict[str, np.ndarray], mode: str) -> np.ndarray:
    eps = 1e-6
    if mode == "raw":
        out = maps
    elif mode == "center":
        out = maps - stats["mean"]
    elif mode == "relu_center":
        out = np.maximum(maps - stats["mean"], 0.0)
    elif mode == "zscore":
        out = (maps - stats["mean"]) / (stats["std"] + eps)
    elif mode == "relu_zscore":
        out = np.maximum((maps - stats["mean"]) / (stats["std"] + eps), 0.0)
    elif mode == "robust_zscore":
        out = (maps - stats["median"]) / (stats["iqr"] + eps)
    elif mode == "relu_robust_zscore":
        out = np.maximum((maps - stats["median"]) / (stats["iqr"] + eps), 0.0)
    elif mode == "ratio":
        out = maps / (stats["mean"] + eps)
    elif mode == "log_ratio":
        out = np.log1p(np.maximum(maps, 0.0)) - np.log1p(np.maximum(stats["mean"], 0.0))
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    k = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_postprocess(maps: np.ndarray, mode: str) -> np.ndarray:
    def apply_one(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        if mode in {"", "none", "raw"}:
            out = img
        elif mode.startswith("gaussian:"):
            out = _gaussian(img, float(mode.split(":", 1)[1]))
        elif mode.startswith("highpass:"):
            sigma = float(mode.split(":", 1)[1])
            out = np.maximum(img - _gaussian(img, sigma), 0.0)
        elif mode.startswith("tophat:"):
            size = int(float(mode.split(":", 1)[1]))
            out = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, _kernel(size))
        elif mode.startswith("closing:"):
            size = int(float(mode.split(":", 1)[1]))
            out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, _kernel(size))
        else:
            raise ValueError(f"Unknown postprocess mode: {mode}")
        return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    return np.stack([apply_one(heatmap) for heatmap in maps]).astype(np.float32)


def load_calibration_stats(run_dir: Path) -> dict[str, np.ndarray]:
    path = run_dir / "normal_pixel_stats.npz"
    if not path.exists():
        raise FileNotFoundError(f"Fixed pixel policy requires {path}")
    data = np.load(path)
    return {name: np.asarray(data[name], dtype=np.float32) for name in ("mean", "std", "median", "iqr")}


def multiscale_candidate_maps(
    primary_output: Any,
    secondary_output: Any,
    run_args: dict[str, Any],
    run_dir: Path,
) -> np.ndarray:
    stats_path = run_dir / MULTISCALE_STATS_NAME
    if not stats_path.exists():
        raise FileNotFoundError(f"Multiscale policy requires {stats_path}")
    saved = np.load(stats_path)
    primary_stats = {
        name: np.asarray(saved[f"primary_{name}"], dtype=np.float32)
        for name in ("mean", "std", "median", "iqr")
    }
    secondary_stats = {
        name: np.asarray(saved[f"secondary_{name}"], dtype=np.float32)
        for name in ("mean", "std", "median", "iqr")
    }
    primary_source = str(run_args.get("multiscale_primary_source") or "feature_raw")
    secondary_source = str(run_args.get("multiscale_secondary_source") or "feature_raw")
    calibration = str(run_args.get("multiscale_calibration_mode") or "relu_zscore")
    primary = calibrate_maps(feature_output_map(primary_output, primary_source), primary_stats, calibration)
    secondary = calibrate_maps(feature_output_map(secondary_output, secondary_source), secondary_stats, calibration)
    primary = apply_postprocess(primary, str(run_args.get("multiscale_primary_postprocess") or "raw"))
    secondary = apply_postprocess(secondary, str(run_args.get("multiscale_secondary_postprocess") or "raw"))
    weight = float(np.clip(float(run_args.get("multiscale_primary_weight", 0.5)), 0.0, 1.0))
    fusion = str(run_args.get("multiscale_fusion_mode") or "weighted")
    if fusion == "weighted":
        fused = weight * primary + (1.0 - weight) * secondary
    elif fusion == "max":
        fused = np.maximum(primary, secondary)
    else:
        raise ValueError(f"Unknown multiscale fusion mode: {fusion}")
    return apply_postprocess(fused, str(run_args.get("multiscale_postprocess") or "gaussian:1"))


def candidate_pixel_maps(output: Any, run_args: dict[str, Any], run_dir: Path) -> np.ndarray:
    source = str(run_args.get("pixel_heatmap_source") or "feature_raw")
    if source == "multiscale_fusion":
        raise ValueError("multiscale_fusion requires multiscale_candidate_maps with two feature outputs")
    calibration_mode = run_args.get("pixel_policy_calibration_mode")
    if calibration_mode:
        base_source = str(run_args.get("pixel_policy_normal_source_map") or "feature_raw")
        maps = feature_output_map(output, base_source)
        maps = calibrate_maps(maps, load_calibration_stats(run_dir), str(calibration_mode))
        return apply_postprocess(maps, str(run_args.get("pixel_policy_postprocess_mode") or "raw"))

    if source.startswith("postprocess_"):
        base_source = str(run_args.get("pixel_policy_base_source") or "feature_raw")
        return apply_postprocess(feature_output_map(output, base_source), source[len("postprocess_") :])

    if source.startswith("fixed_train_normal_"):
        raise ValueError(
            "Fixed train-normal policy metadata is incomplete. "
            "Expected pixel_policy_calibration_mode and pixel_policy_postprocess_mode."
        )
    return feature_output_map(output, source)


def candidate_score_maps(output: Any, run_args: dict[str, Any]) -> np.ndarray:
    source = str(run_args.get("image_score_source") or "feature_raw_cosine")
    if source == "multiscale_fusion":
        raise ValueError("multiscale_fusion score maps require two feature outputs")
    return feature_output_map(output, source)
