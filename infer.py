from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seer_ad_v2.config import (
    cfg_device,
    cfg_first,
    cfg_float,
    cfg_int,
    cfg_seed,
    dataset_category,
    diffusion_base_channels,
    diffusion_timesteps,
    image_size as cfg_image_size,
    latency_budget_ms,
    load_config,
    make_run_dir,
    max_regions,
    patch_size as cfg_patch_size,
    reconstruction_steps,
    resolve_device,
)
from seer_ad_v2.data.datasets import build_dataset
from seer_ad_v2.data.hard_negative_mining import ROI, crop_resize, heatmap_to_rois
from seer_ad_v2.evaluation.metrics_detection import detection_metrics
from seer_ad_v2.evaluation.metrics_efficiency import timed_cuda
from seer_ad_v2.evaluation.metrics_plan import efficiency_summary, plan_metric_summary
from seer_ad_v2.evaluation.pareto import write_pareto
from seer_ad_v2.evaluation.pixel_threshold_policy import (
    load_pixel_threshold_policy,
)
from seer_ad_v2.evaluation.prediction_schema import prediction_heatmap_payload
from seer_ad_v2.evaluation.score_aggregation import IMAGE_SCORE_MODES, image_score_from_heatmap
from seer_ad_v2.models.counterfactual.repair_verification import apply_crv_to_heatmap, apply_verifier_to_heatmap, roi_score, score_drop
from seer_ad_v2.models.diffusion.local_refiner import local_repair
from seer_ad_v2.models.diffusion.reconstruction import fused_residual_heatmap, residual_heatmap
from seer_ad_v2.models.feature_prior import feature_prior_scores, load_feature_prior_components, retrieve_normal_reference
from seer_ad_v2.models.region_verifier.hn_sev import HNSEV, build_sev_input, sev_probability
from seer_ad_v2.models.region_verifier.prototype_bank import PrototypeBank
from seer_ad_v2.models.scheduler.lc_rds import (
    ExpectedUtilityScheduler,
    LCRDS,
    RuleScheduler,
    SchedulerAction,
    action_from_name,
    action_from_steps,
    choose_action_with_model,
    production_budget_guard_latency_estimates,
    roi_features,
)
from seer_ad_v2.models.seer_ad_v2 import build_diffusion_components
from seer_ad_v2.utils.image import heatmap_to_uint8, save_image, tensor_to_uint8
from seer_ad_v2.utils.io import load_checkpoint, save_json
from seer_ad_v2.utils.run import save_run_metadata
from seer_ad_v2.utils.seed import seed_everything


ABLATIONS = [
    "full",
    "residual_only",
    "no_sev",
    "synthetic_only_sev",
    "no_prototype",
    "no_crv",
    "repair_visualization_only",
    "fixed10",
    "fixed25",
    "rule_brds",
    "learned_lc_rds",
    "feature_only",
    "feature_hn_sev",
    "feature_hn_sev_crv",
    "feature_first",
    "feature_first_no_retrieval",
    "feature_fixed10",
    "feature_fixed25",
    "feature_rule_brds",
    "utility_lc_rds",
]
FEATURE_ABLATIONS = {
    "feature_only",
    "feature_hn_sev",
    "feature_hn_sev_crv",
    "feature_first",
    "feature_first_no_retrieval",
    "feature_fixed10",
    "feature_fixed25",
    "feature_rule_brds",
    "utility_lc_rds",
}
IMAGE_SCORE_SOURCES = [
    "final",
    "base",
    "feature",
    "feature_raw",
    "feature_raw_distance",
    "feature_raw_cosine",
]
PIXEL_HEATMAP_SOURCES = [
    "final",
    "base",
    "residual",
    "feature",
    "feature_raw",
    "feature_raw_distance",
    "feature_raw_cosine",
    "score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Lite-SEER-AD v2 inference.")
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--category", default=None)
    p.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    p.add_argument("--sev-checkpoint", default=None)
    p.add_argument("--scheduler-checkpoint", default=None)
    p.add_argument("--feature-prior-checkpoint", default=None)
    p.add_argument("--feature-prior-weight", type=float, default=1.0)
    p.add_argument("--enable-retrieval-repair", action="store_true")
    p.add_argument("--disable-retrieval-repair", action="store_true")
    p.add_argument("--retrieval-reference-weight", type=float, default=None)
    p.add_argument("--retrieval-min-similarity", type=float, default=None)
    p.add_argument("--retrieval-spatial-weight", type=float, default=None)
    p.add_argument("--retrieval-reference-mode", choices=["raw", "texture"], default=None)
    p.add_argument("--allow-random-feature-weights", action="store_true", help="Only use for smoke tests if pretrained feature weights are unavailable.")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--run-name", default="default")
    p.add_argument("--ablation", choices=ABLATIONS, default="full")
    p.add_argument("--crv-weight", type=float, default=0.35)
    p.add_argument("--image-score-mode", choices=IMAGE_SCORE_MODES, default=None)
    p.add_argument("--image-score-source", choices=IMAGE_SCORE_SOURCES, default=None)
    p.add_argument("--pixel-heatmap-source", choices=PIXEL_HEATMAP_SOURCES, default=None)
    p.add_argument("--reconstruction-steps", type=int, default=None)
    p.add_argument(
        "--latency-budget-ms",
        type=float,
        default=None,
        help="Override lc_rds.latency_budget_ms for production budget-sweep runs.",
    )
    p.add_argument("--pixel-threshold-policy", default=None)
    p.add_argument("--require-fixed-threshold", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _timer_start(device: str) -> float:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _timer_stop(start: float, device: str) -> float:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _select_heatmap_source(
    source: str,
    *,
    final: np.ndarray,
    base: np.ndarray,
    residual: np.ndarray,
    feature: np.ndarray | None,
    feature_raw: np.ndarray | None,
    feature_raw_distance: np.ndarray | None,
    feature_raw_cosine: np.ndarray | None,
    score: np.ndarray | None = None,
) -> np.ndarray:
    choices = {
        "final": final,
        "base": base,
        "residual": residual,
        "feature": feature,
        "feature_raw": feature_raw,
        "feature_raw_distance": feature_raw_distance,
        "feature_raw_cosine": feature_raw_cosine,
        "score": score,
    }
    heatmap = choices.get(source)
    if heatmap is None:
        raise ValueError(f"Heatmap source '{source}' is unavailable for this run.")
    return heatmap


def _load_sev(path: str | None, device: str) -> tuple[HNSEV | None, PrototypeBank | None, int, float, int]:
    if not path:
        return None, None, 64, 0.5, 8
    ckpt = load_checkpoint(path)
    patch_size = int(ckpt.get("patch_size", 64))
    in_channels = int(ckpt.get("in_channels", 8))
    model = HNSEV(in_channels=in_channels, base_channels=int(ckpt.get("base_channels", 24))).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    bank = PrototypeBank()
    bank.load_state_dict(ckpt.get("prototype_bank", {}))
    return model, bank, patch_size, float(ckpt.get("operating_threshold", 0.5)), in_channels


def _load_scheduler(path: str | None, device: str) -> LCRDS | None:
    if not path:
        return None
    ckpt = load_checkpoint(path)
    action_names = ckpt.get("action_names")
    if isinstance(action_names, list) and action_names:
        action_names = [str(name) for name in action_names]
    else:
        action_names = None
    if action_names:
        model = LCRDS(num_actions=len(action_names), action_names=action_names).to(device)
    else:
        model = LCRDS().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _load_feature_prior(path: str | None, device: str, allow_random_weights: bool) -> tuple[dict | None, torch.nn.Module | None, list[str]]:
    if not path:
        return None, None, []
    ckpt = load_checkpoint(path)
    feature_prior, extractor, layers = load_feature_prior_components(ckpt, device, allow_random_weights=allow_random_weights)
    return feature_prior, extractor, layers


def _crop_heatmap_patch(heatmap: np.ndarray | None, roi: ROI, patch_size: int) -> torch.Tensor | None:
    if heatmap is None:
        return None
    patch = heatmap[roi.y1 : roi.y2, roi.x1 : roi.x2]
    if patch.size == 0:
        patch = np.zeros((patch_size, patch_size), dtype=np.float32)
    else:
        patch = cv2.resize(patch.astype(np.float32), (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(patch).float().unsqueeze(0)


def _draw_rois(shape: tuple[int, int], rois: list[ROI], color: int = 255) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    for roi in rois:
        cv2.rectangle(canvas, (roi.x1, roi.y1), (max(roi.x1, roi.x2 - 1), max(roi.y1, roi.y2 - 1)), color, 1)
    return canvas


def _save_metrics_csv(metrics: dict[str, float | int | None], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


def _scheduler_action(
    ablation: str,
    scheduler: LCRDS | None,
    features: np.ndarray,
    device: str,
    budget_ms: float,
    spent_ms: float,
    expected_gain: float | None = None,
    utility_scheduler: ExpectedUtilityScheduler | None = None,
) -> SchedulerAction:
    if ablation in {"residual_only", "feature_only", "feature_hn_sev"}:
        return action_from_name("skip")
    if ablation == "fixed10":
        return action_from_steps(10)
    if ablation == "fixed25":
        return action_from_steps(25)
    if ablation == "feature_fixed10":
        return action_from_steps(10)
    if ablation == "feature_fixed25":
        return action_from_steps(25)
    if ablation == "rule_brds":
        return RuleScheduler(latency_budget_ms=budget_ms).choose(features, spent_ms=spent_ms)
    if ablation == "feature_rule_brds":
        return RuleScheduler(latency_budget_ms=budget_ms).choose(features, spent_ms=spent_ms)
    if ablation in {"utility_lc_rds", "feature_first", "feature_first_no_retrieval", "feature_hn_sev_crv"}:
        active_scheduler = utility_scheduler or ExpectedUtilityScheduler(latency_budget_ms=budget_ms)
        return active_scheduler.choose(features, spent_ms=spent_ms, expected_gain=expected_gain)
    if ablation == "learned_lc_rds":
        return choose_action_with_model(scheduler, features, device=device) if scheduler is not None else RuleScheduler(budget_ms).choose(features, spent_ms)
    if scheduler is not None:
        return choose_action_with_model(scheduler, features, device=device)
    return RuleScheduler(latency_budget_ms=budget_ms).choose(features, spent_ms=spent_ms)


def _roi_log_row(
    idx: int,
    source_path: str,
    roi_idx: int,
    roi: ROI,
    image_shape: tuple[int, int],
    residual_score: float,
    sev_score: float,
    sev_threshold: float,
    proto_dist: float,
    proto_novelty: float,
    action: SchedulerAction,
    before: float,
    after: float,
    drop: float,
    feature_score: float = 0.0,
    feature_cosine_gap: float = 0.0,
    pixel_sdr: float = 0.0,
    feature_sdr: float = 0.0,
    prototype_sdr: float = 0.0,
    expected_gain: float = 0.0,
    expected_latency_ms: float = 0.0,
    action_latency_ms: float = 0.0,
    cumulative_spent_ms: float = 0.0,
    retrieval_reference_index: int = -1,
    retrieval_similarity: float = 0.0,
    retrieval_weight: float = 0.0,
) -> dict[str, float | int | str | list[int]]:
    h, w = image_shape
    return {
        "image_index": idx,
        "source_path": source_path,
        "roi_id": roi_idx,
        "bbox": [roi.x1, roi.y1, roi.x2, roi.y2],
        "area_ratio": float((roi.x2 - roi.x1) * (roi.y2 - roi.y1) / max(1, h * w)),
        "residual_score": residual_score,
        "hn_sev_confidence": sev_score,
        "hn_sev_threshold": sev_threshold,
        "hn_sev_positive": int(sev_score >= sev_threshold),
        "prototype_distance": proto_dist,
        "prototype_novelty": proto_novelty,
        "feature_score": feature_score,
        "feature_cosine_gap": feature_cosine_gap,
        "scheduler_action": action.name,
        "nfe": action.steps,
        "score_before": before,
        "score_after": after,
        "sdr": drop,
        "pixel_sdr": pixel_sdr,
        "feature_sdr": feature_sdr,
        "prototype_sdr": prototype_sdr,
        "repair_gain": drop,
        "expected_gain": expected_gain,
        "expected_latency_ms": expected_latency_ms,
        "action_latency_ms": action_latency_ms,
        "cumulative_spent_ms": cumulative_spent_ms,
        "realized_gain_per_ms": float(drop / max(1.0, action_latency_ms)),
        "retrieval_reference_index": retrieval_reference_index,
        "retrieval_similarity": retrieval_similarity,
        "retrieval_weight": retrieval_weight,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    args.image_score_mode = args.image_score_mode or str(cfg_first(cfg, ("evaluation.image_score_mode",), "max_mean"))
    if args.image_score_mode not in IMAGE_SCORE_MODES:
        raise ValueError(f"Unknown image score mode: {args.image_score_mode}")
    args.image_score_source = args.image_score_source or str(cfg_first(cfg, ("evaluation.image_score_source",), "final"))
    if args.image_score_source not in IMAGE_SCORE_SOURCES:
        raise ValueError(f"Unknown image score source: {args.image_score_source}")
    args.pixel_heatmap_source = args.pixel_heatmap_source or str(cfg_first(cfg, ("evaluation.pixel_heatmap_source",), "final"))
    if args.pixel_heatmap_source not in PIXEL_HEATMAP_SOURCES:
        raise ValueError(f"Unknown pixel heatmap source: {args.pixel_heatmap_source}")
    seed_everything(cfg_seed(cfg, args.seed))
    ckpt = load_checkpoint(args.checkpoint)
    category = args.category or ckpt.get("category") or dataset_category(cfg)
    image_size = cfg_image_size(cfg, args.image_size, ckpt.get("image_size"))
    device = resolve_device(cfg_device(cfg, args.device))
    run_dir = make_run_dir(cfg, args.run_name)
    save_run_metadata(run_dir, cfg, args, device, "infer")
    for sub in ["heatmaps", "masks", "repairs", "images"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        cfg_first(cfg, ("dataset.name",), "mvtec"),
        cfg_first(cfg, ("dataset.root",), "SEER-AD-dataset/MVTec-AD"),
        category,
        "test",
        image_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model, diffusion = build_diffusion_components(
        diffusion_base_channels(cfg, ckpt.get("base_channels")),
        diffusion_timesteps(cfg, ckpt.get("timesteps")),
        device,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    sev_model, proto_bank, sev_patch_size, sev_threshold, sev_in_channels = _load_sev(args.sev_checkpoint, device)
    patch_size = sev_patch_size or cfg_patch_size(cfg)
    scheduler = _load_scheduler(args.scheduler_checkpoint, device)
    feature_prior, feature_extractor, feature_layers = _load_feature_prior(args.feature_prior_checkpoint, device, args.allow_random_feature_weights)
    feature_mode = args.ablation in FEATURE_ABLATIONS
    needs_feature_prior = feature_prior is not None and (feature_mode or sev_in_channels > 8)
    if feature_mode and feature_prior is None:
        raise ValueError(f"A feature prior checkpoint is required for ablation={args.ablation}.")
    if args.ablation in {"residual_only", "no_sev", "feature_only"}:
        sev_model, proto_bank = None, None
        sev_threshold = 0.5
    if args.ablation == "no_prototype":
        proto_bank = None
    if args.ablation == "rule_brds":
        scheduler = None
    recon_steps = int(args.reconstruction_steps or reconstruction_steps(cfg))
    retrieval_reference_weight = float(
        args.retrieval_reference_weight
        if args.retrieval_reference_weight is not None
        else cfg_float(cfg, ("feature_prior.retrieval.reference_weight",), 0.25)
    )
    retrieval_min_similarity = float(
        args.retrieval_min_similarity
        if args.retrieval_min_similarity is not None
        else cfg_float(cfg, ("feature_prior.retrieval.min_similarity",), 0.65)
    )
    retrieval_spatial_weight = float(
        args.retrieval_spatial_weight
        if args.retrieval_spatial_weight is not None
        else cfg_float(cfg, ("feature_prior.retrieval.spatial_weight",), 0.25)
    )
    retrieval_reference_mode = str(
        args.retrieval_reference_mode
        or cfg_first(cfg, ("feature_prior.retrieval.reference_mode",), "texture")
    )
    prior_state = feature_prior.get("prior_state", feature_prior) if feature_prior is not None else {}
    retrieval_requested = bool(
        args.enable_retrieval_repair
        or cfg_first(cfg, ("feature_prior.retrieval.enabled",), False)
    )
    retrieval_enabled = bool(
        feature_mode
        and feature_prior is not None
        and feature_extractor is not None
        and "retrieval_features_norm" in prior_state
        and "retrieval_patches" in prior_state
        and retrieval_requested
        and not args.disable_retrieval_repair
        and args.ablation != "feature_first_no_retrieval"
    )

    labels: list[int] = []
    image_scores: list[float] = []
    masks: list[np.ndarray] = []
    heatmaps: list[np.ndarray] = []
    final_heatmaps: list[np.ndarray] = []
    score_heatmaps: list[np.ndarray] = []
    score_rows: list[dict[str, str | int | float]] = []
    budget_rows: list[dict[str, float | int | str | list[int]]] = []
    pareto_rows: list[dict[str, str | int | float]] = []
    crv_drops: list[float] = []
    per_image_efficiency: list[dict[str, float | int]] = []
    component_latency_rows: list[dict[str, float | int]] = []
    budget_ms = float(args.latency_budget_ms if args.latency_budget_ms is not None else latency_budget_ms(cfg))
    production_budget_guard = (
        args.latency_budget_ms is not None
        and args.ablation in {"utility_lc_rds", "feature_first", "feature_first_no_retrieval", "feature_hn_sev_crv"}
    )
    utility_scheduler = ExpectedUtilityScheduler(
        latency_budget_ms=budget_ms,
        latency_estimates=production_budget_guard_latency_estimates() if production_budget_guard else None,
    )
    retrieval_similarities: list[float] = []
    retrieval_weights: list[float] = []

    for idx, batch in enumerate(tqdm(loader, desc=f"infer:{args.ablation}", leave=False)):
        x = batch["image"].to(device)
        source_path = batch["path"][0]
        image_dir = run_dir / "images" / f"{idx:05d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        with timed_cuda(device) as timer:
            detector_start = _timer_start(device)
            recon = diffusion.reconstruct(model, x, steps=recon_steps)
            pixel_heatmap = fused_residual_heatmap(x, recon)
            feature_heatmap = None
            feature_distance_heatmap = None
            feature_cosine_heatmap = None
            feature_raw_heatmap = None
            feature_raw_distance_heatmap = None
            feature_raw_cosine_heatmap = None
            if needs_feature_prior and feature_prior is not None and feature_extractor is not None:
                feature_out = feature_prior_scores(feature_prior, feature_extractor, feature_layers, x, device, image_size)
                feature_heatmap = feature_out.heatmaps[0]
                feature_distance_heatmap = feature_out.distance_heatmaps[0]
                feature_cosine_heatmap = feature_out.cosine_heatmaps[0]
                feature_raw_heatmap = feature_out.raw_heatmaps[0]
                feature_raw_distance_heatmap = feature_out.raw_distance_heatmaps[0]
                feature_raw_cosine_heatmap = feature_out.raw_cosine_heatmaps[0]
            base_heatmap = feature_heatmap if feature_mode and feature_heatmap is not None else pixel_heatmap
            rois = heatmap_to_rois(
                base_heatmap,
                threshold_quantile=cfg_float(cfg, ("roi.threshold_quantile",), 0.985),
                min_area=cfg_int(cfg, ("roi.min_area",), 16),
                max_rois=max_regions(cfg),
                pad=cfg_int(cfg, ("roi.pad",), 8),
            )
            detector_latency_ms = _timer_stop(detector_start, device)
            verifier_latency_ms = 0.0
            repair_latency_ms = 0.0
            repaired = x.clone()
            selected_rois: list[ROI] = []
            verified_rois: list[ROI] = []
            drops: list[float] = []
            verifier_scores: list[float] = []
            image_roi_rows: list[dict[str, float | int | str | list[int]]] = []
            nfe = recon_steps
            spent_ms = 0.0
            repaired_area = 0

            for ridx, roi in enumerate(rois):
                orig_patch = crop_resize(x[0], roi, patch_size).unsqueeze(0).to(device)
                rec_patch = crop_resize(recon[0], roi, patch_size).unsqueeze(0).to(device)
                res_patch = residual_heatmap(orig_patch, rec_patch)
                verifier_start = _timer_start(device)
                proto_dist = 0.0
                proto_novelty = 0.0
                feature_score = roi_score(feature_heatmap, roi) if feature_heatmap is not None else 0.0
                feature_gap_score = roi_score(feature_cosine_heatmap, roi) if feature_cosine_heatmap is not None else 0.0
                feature_patch = _crop_heatmap_patch(feature_heatmap, roi, patch_size)
                if sev_in_channels > 8 and feature_patch is None:
                    feature_patch = torch.zeros(1, patch_size, patch_size)
                if proto_bank is not None:
                    proto_dist = float(proto_bank.distance(orig_patch).detach().cpu()[0])
                    proto_novelty = float(proto_bank.novelty(orig_patch).detach().cpu()[0])
                if sev_model is not None:
                    sev_in = build_sev_input(
                        orig_patch,
                        rec_patch,
                        res_patch,
                        torch.tensor([proto_novelty], device=device),
                        None if feature_patch is None or sev_in_channels <= 8 else feature_patch.unsqueeze(0).to(device),
                        None if sev_in_channels <= 8 else torch.tensor([feature_gap_score], device=device),
                    )
                    sev_score = float(sev_probability(sev_model, sev_in).detach().cpu()[0])
                else:
                    sev_score = float(feature_score if feature_mode else roi.peak)
                verifier_latency_ms += _timer_stop(verifier_start, device)
                if sev_score >= sev_threshold or args.ablation in {"residual_only", "no_sev"}:
                    verified_rois.append(roi)
                expected_gain = max(float(sev_score), float(proto_novelty), float(feature_score), float(feature_gap_score))
                feats = roi_features(roi, sev_score, max(proto_novelty, feature_gap_score), base_heatmap.shape, len(rois))
                action = _scheduler_action(
                    args.ablation,
                    scheduler,
                    feats,
                    device,
                    budget_ms,
                    spent_ms,
                    expected_gain=expected_gain,
                    utility_scheduler=utility_scheduler,
                )
                if spent_ms > budget_ms and args.ablation not in {"fixed10", "fixed25", "feature_fixed10", "feature_fixed25"}:
                    action = action_from_name("skip")
                expected_latency_ms = float(utility_scheduler.expected_latency.get(action.name, 0.0))
                before = roi_score(base_heatmap, roi)
                after = before
                pixel_drop = 0.0
                feature_drop = 0.0
                prototype_drop = 0.0
                action_latency_ms = 0.0
                retrieval_reference_index = -1
                retrieval_similarity = 0.0
                retrieval_weight = 0.0
                if action.steps > 0:
                    if str(device).startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    action_start = time.perf_counter()
                    reference_patch = None
                    if retrieval_enabled and feature_prior is not None and feature_extractor is not None:
                        reference_patch, retrieval_similarity, retrieval_reference_index = retrieve_normal_reference(
                            feature_prior,
                            feature_extractor,
                            feature_layers,
                            x,
                            roi,
                            device,
                            output_size=(max(1, roi.y2 - roi.y1), max(1, roi.x2 - roi.x1)),
                            spatial_weight=retrieval_spatial_weight,
                        )
                        confidence = float(
                            np.clip(
                                (retrieval_similarity - retrieval_min_similarity)
                                / max(1e-6, 1.0 - retrieval_min_similarity),
                                0.0,
                                1.0,
                            )
                        )
                        retrieval_weight = retrieval_reference_weight * confidence
                        retrieval_similarities.append(retrieval_similarity)
                        retrieval_weights.append(retrieval_weight)
                    repaired = local_repair(
                        repaired,
                        model,
                        diffusion,
                        roi,
                        steps=action.steps,
                        native_size=image_size if action.native_refine else None,
                        reference_patch=reference_patch,
                        reference_weight=retrieval_weight,
                        reference_mode=retrieval_reference_mode,
                    )
                    nfe += action.steps
                    repaired_area += max(0, roi.x2 - roi.x1) * max(0, roi.y2 - roi.y1)
                    after_pixel_map = fused_residual_heatmap(repaired, recon)
                    pixel_drop = score_drop(pixel_heatmap, after_pixel_map, roi)
                    if feature_mode and feature_prior is not None and feature_extractor is not None:
                        after_feature_out = feature_prior_scores(feature_prior, feature_extractor, feature_layers, repaired, device, image_size)
                        after_feature_heatmap = after_feature_out.heatmaps[0]
                        after_distance_heatmap = after_feature_out.distance_heatmaps[0]
                        feature_drop = score_drop(feature_heatmap, after_feature_heatmap, roi) if feature_heatmap is not None else 0.0
                        prototype_drop = score_drop(feature_distance_heatmap, after_distance_heatmap, roi) if feature_distance_heatmap is not None else 0.0
                        after = roi_score(after_feature_heatmap, roi)
                    else:
                        after = roi_score(after_pixel_map, roi)
                drop = max(0.0, 0.34 * pixel_drop + 0.43 * feature_drop + 0.23 * prototype_drop) if feature_mode else (
                    score_drop(base_heatmap, fused_residual_heatmap(repaired, recon), roi) if action.steps > 0 else 0.0
                )
                if action.steps > 0:
                    if str(device).startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    action_latency_ms = (time.perf_counter() - action_start) * 1000.0
                    spent_ms += action_latency_ms
                    repair_latency_ms += action_latency_ms
                    utility_scheduler.observe(action, action_latency_ms, drop, predicted_gain=expected_gain)
                selected_rois.append(roi)
                drops.append(drop)
                verifier_scores.append(float(sev_score if sev_model is not None else max(proto_novelty, feature_score)))
                crv_drops.append(drop)
                row = _roi_log_row(
                    idx,
                    source_path,
                    ridx,
                    roi,
                    base_heatmap.shape,
                    before,
                    sev_score,
                    sev_threshold,
                    proto_dist,
                    proto_novelty,
                    action,
                    before,
                    after,
                    drop,
                    feature_score=feature_score,
                    feature_cosine_gap=feature_gap_score,
                    pixel_sdr=pixel_drop,
                    feature_sdr=feature_drop,
                    prototype_sdr=prototype_drop,
                    expected_gain=expected_gain,
                    expected_latency_ms=expected_latency_ms,
                    action_latency_ms=action_latency_ms,
                    cumulative_spent_ms=spent_ms,
                    retrieval_reference_index=retrieval_reference_index,
                    retrieval_similarity=retrieval_similarity,
                    retrieval_weight=retrieval_weight,
                )
                budget_rows.append(row)
                image_roi_rows.append(row)

            if args.ablation in {"no_crv", "repair_visualization_only", "residual_only", "feature_only", "feature_hn_sev"}:
                final_heatmap = base_heatmap
            else:
                final_heatmap = apply_crv_to_heatmap(base_heatmap, selected_rois, drops, weight=args.crv_weight)
            verifier_weight = cfg_float(cfg, ("hn_sev.prototype_heatmap_weight",), 0.5)
            if proto_bank is not None and args.ablation not in {"residual_only", "no_sev", "no_prototype", "feature_only"}:
                final_heatmap = apply_verifier_to_heatmap(final_heatmap, selected_rois, verifier_scores, weight=verifier_weight)
        latency_ms = timer.elapsed_ms
        component_latency_rows.append(
            {
                "index": idx,
                "detector_latency_ms": detector_latency_ms,
                "hn_sev_latency_ms": verifier_latency_ms,
                "repair_latency_ms": repair_latency_ms,
                "end_to_end_latency_ms": latency_ms,
            }
        )
        score_heatmap = _select_heatmap_source(
            args.image_score_source,
            final=final_heatmap,
            base=base_heatmap,
            residual=pixel_heatmap,
            feature=feature_heatmap,
            feature_raw=feature_raw_heatmap,
            feature_raw_distance=feature_raw_distance_heatmap,
            feature_raw_cosine=feature_raw_cosine_heatmap,
        )
        pixel_eval_heatmap = _select_heatmap_source(
            args.pixel_heatmap_source,
            final=final_heatmap,
            base=base_heatmap,
            residual=pixel_heatmap,
            feature=feature_heatmap,
            feature_raw=feature_raw_heatmap,
            feature_raw_distance=feature_raw_distance_heatmap,
            feature_raw_cosine=feature_raw_cosine_heatmap,
            score=score_heatmap,
        )
        image_score = image_score_from_heatmap(score_heatmap, mode=args.image_score_mode)
        mask_np = batch["mask"][0, 0].numpy().astype(np.uint8)

        labels.append(int(batch["label"][0]))
        image_scores.append(image_score)
        masks.append(mask_np)
        heatmaps.append(pixel_eval_heatmap.astype(np.float32))
        final_heatmaps.append(final_heatmap.astype(np.float32))
        score_heatmaps.append(score_heatmap.astype(np.float32))
        stem = f"{idx:05d}"
        save_image(run_dir / "heatmaps" / f"{stem}.png", heatmap_to_uint8(pixel_eval_heatmap))
        save_image(run_dir / "masks" / f"{stem}.png", (mask_np * 255).astype(np.uint8))
        save_image(run_dir / "repairs" / f"{stem}.png", tensor_to_uint8(repaired[0]))
        save_image(image_dir / "input.png", tensor_to_uint8(x[0]))
        save_image(image_dir / "reconstruction.png", tensor_to_uint8(recon[0]))
        save_image(image_dir / "residual.png", heatmap_to_uint8(pixel_heatmap))
        if feature_heatmap is not None:
            save_image(image_dir / "feature_prior.png", heatmap_to_uint8(feature_heatmap))
        save_image(image_dir / "candidate_roi.png", _draw_rois(base_heatmap.shape, rois))
        save_image(image_dir / "verified_roi.png", _draw_rois(base_heatmap.shape, verified_rois))
        save_image(image_dir / "mask.png", (mask_np * 255).astype(np.uint8))
        save_image(image_dir / "ground_truth.png", (mask_np * 255).astype(np.uint8))
        save_image(image_dir / "final_heatmap.png", heatmap_to_uint8(final_heatmap))
        save_image(image_dir / "pixel_eval_heatmap.png", heatmap_to_uint8(pixel_eval_heatmap))
        save_image(image_dir / "repair.png", tensor_to_uint8(repaired[0]))
        np.savez_compressed(
            image_dir / "residual_heatmap.npz",
            residual=pixel_heatmap.astype(np.float32),
            feature=np.zeros_like(pixel_heatmap, dtype=np.float32) if feature_heatmap is None else feature_heatmap.astype(np.float32),
            feature_raw=np.zeros_like(pixel_heatmap, dtype=np.float32) if feature_raw_heatmap is None else feature_raw_heatmap.astype(np.float32),
            feature_raw_distance=np.zeros_like(pixel_heatmap, dtype=np.float32)
            if feature_raw_distance_heatmap is None
            else feature_raw_distance_heatmap.astype(np.float32),
            feature_raw_cosine=np.zeros_like(pixel_heatmap, dtype=np.float32) if feature_raw_cosine_heatmap is None else feature_raw_cosine_heatmap.astype(np.float32),
            final=final_heatmap.astype(np.float32),
            score=score_heatmap.astype(np.float32),
            pixel_eval=pixel_eval_heatmap.astype(np.float32),
        )
        with (image_dir / "roi_log.jsonl").open("w", encoding="utf-8") as f:
            for row in image_roi_rows:
                f.write(json.dumps(row) + "\n")

        local_region_ratio = float(sum((r.x2 - r.x1) * (r.y2 - r.y1) for r in selected_rois) / max(1, base_heatmap.size))
        repaired_area_ratio = float(repaired_area / max(1, base_heatmap.size))
        score_rows.append(
            {
                "index": idx,
                "path": source_path,
                "label": int(batch["label"][0]),
                "image_score": image_score,
                "latency_ms": latency_ms,
                "nfe": nfe,
                "ablation": args.ablation,
            }
        )
        pareto_rows.append({"index": idx, "latency_ms": latency_ms, "nfe": nfe, "image_score": image_score, "ablation": args.ablation})
        per_image_efficiency.append(
            {
                "index": idx,
                "latency_ms": latency_ms,
                "nfe": nfe,
                "repaired_area_ratio": repaired_area_ratio,
                "local_region_ratio": local_region_ratio,
            }
        )

    labels_np = np.asarray(labels, dtype=np.uint8)
    scores_np = np.asarray(image_scores, dtype=np.float32)
    masks_np = np.stack(masks).astype(np.uint8)
    heatmaps_np = np.stack(heatmaps).astype(np.float32)
    final_heatmaps_np = np.stack(final_heatmaps).astype(np.float32)
    score_heatmaps_np = np.stack(score_heatmaps).astype(np.float32)
    policy_path = (
        Path(args.pixel_threshold_policy)
        if args.pixel_threshold_policy
        else run_dir / "pixel_threshold_policy.json"
    )
    threshold_policy = (
        load_pixel_threshold_policy(policy_path) if policy_path.exists() else None
    )
    if args.require_fixed_threshold and threshold_policy is None:
        raise FileNotFoundError(
            f"Fixed pixel threshold policy is required but missing: {policy_path}"
        )
    metrics = detection_metrics(
        labels_np,
        scores_np,
        masks_np,
        heatmaps_np,
        pixel_threshold=(
            float(threshold_policy["threshold"]) if threshold_policy else None
        ),
        threshold_protocol=(
            str(threshold_policy.get("protocol", "fixed_external"))
            if threshold_policy
            else None
        ),
    )
    plan_metrics = plan_metric_summary(masks_np, heatmaps_np, budget_rows, crv_drops, pareto_rows, labels_np)
    eff_metrics = efficiency_summary(per_image_efficiency)
    metrics.update(plan_metrics)
    metrics.update({f"eff_{k}": v for k, v in eff_metrics.items()})
    metrics["crv_weight"] = float(args.crv_weight)
    metrics["image_score_mode"] = args.image_score_mode
    metrics["image_score_source"] = args.image_score_source
    metrics["pixel_heatmap_source"] = args.pixel_heatmap_source
    metrics["reconstruction_steps"] = recon_steps
    metrics["feature_prior_enabled"] = int(feature_prior is not None)
    metrics["feature_mode"] = int(feature_mode)
    metrics["retrieval_repair_enabled"] = int(retrieval_enabled)
    metrics["retrieval_repair_requested"] = int(retrieval_requested)
    metrics["retrieval_reference_weight"] = retrieval_reference_weight
    metrics["retrieval_min_similarity"] = retrieval_min_similarity
    metrics["retrieval_spatial_weight"] = retrieval_spatial_weight
    metrics["retrieval_reference_mode"] = retrieval_reference_mode
    metrics["retrieval_match_count"] = len(retrieval_similarities)
    metrics["retrieval_similarity_mean"] = float(np.mean(retrieval_similarities)) if retrieval_similarities else None
    metrics["retrieval_effective_weight_mean"] = float(np.mean(retrieval_weights)) if retrieval_weights else None
    metrics["latency_budget_ms"] = budget_ms
    metrics["scheduler_expected_latency_ms"] = utility_scheduler.expected_latency
    metrics["scheduler_gain_multiplier"] = utility_scheduler.gain_multiplier
    metrics["prototype_heatmap_weight"] = float(cfg_float(cfg, ("hn_sev.prototype_heatmap_weight",), 0.5))
    for key in (
        "detector_latency_ms",
        "hn_sev_latency_ms",
        "repair_latency_ms",
        "end_to_end_latency_ms",
    ):
        values = [float(row[key]) for row in component_latency_rows]
        metrics[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
    metrics["component_latency_protocol"] = (
        "per_image_synchronized_component_breakdown_v1"
    )
    if threshold_policy:
        metrics["pixel_threshold_policy_path"] = str(policy_path)
        metrics["max_normal_pixel_fpr"] = threshold_policy.get(
            "max_normal_pixel_fpr"
        )
        metrics["threshold_uses_real_anomaly_labels"] = threshold_policy.get(
            "uses_real_anomaly_labels", False
        )
        metrics["threshold_uses_real_anomaly_masks"] = threshold_policy.get(
            "uses_real_anomaly_masks", False
        )
    threshold = float(metrics.get("threshold") or 0.5)
    if not np.isfinite(threshold):
        threshold = 0.5
    for idx, heatmap in enumerate(heatmaps_np):
        pred_mask = (heatmap >= threshold).astype(np.uint8) * 255
        save_image(run_dir / "masks" / f"{idx:05d}.png", pred_mask)
        save_image(run_dir / "images" / f"{idx:05d}" / "final_mask.png", pred_mask)
    save_json(metrics, run_dir / "metrics.json")
    _save_metrics_csv(metrics, run_dir / "metrics.csv")
    _save_metrics_csv(eff_metrics, run_dir / "efficiency.csv")
    np.savez_compressed(
        run_dir / "predictions.npz",
        labels=labels_np,
        image_scores=scores_np,
        masks=masks_np,
        **prediction_heatmap_payload(
            heatmaps_np,
            final_heatmaps_np,
            score_heatmaps_np,
        ),
        paths=np.asarray([r["path"] for r in score_rows]),
        ablation=np.asarray(args.ablation),
    )
    np.save(run_dir / "crv_score_drop.npy", np.asarray(crv_drops, dtype=np.float32))

    with (run_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "path", "label", "image_score", "latency_ms", "nfe", "ablation"])
        writer.writeheader()
        writer.writerows(score_rows)
    with (run_dir / "component_latency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "detector_latency_ms",
                "hn_sev_latency_ms",
                "repair_latency_ms",
                "end_to_end_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(component_latency_rows)
    save_json(budget_rows, run_dir / "roi_budget.json")
    with (run_dir / "roi_budget.jsonl").open("w", encoding="utf-8") as f:
        for row in budget_rows:
            f.write(json.dumps(row) + "\n")
    write_pareto(run_dir / "pareto.csv", pareto_rows)
    print(f"Saved inference outputs to {run_dir}")


if __name__ == "__main__":
    main()
