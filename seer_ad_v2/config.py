from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def cfg_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def cfg_first(cfg: dict[str, Any], dotted_paths: list[str] | tuple[str, ...], default: Any = None) -> Any:
    for path in dotted_paths:
        value = cfg_get(cfg, path, None)
        if value is not None:
            return value
    return default


def cfg_int(cfg: dict[str, Any], dotted_paths: list[str] | tuple[str, ...], default: int) -> int:
    return int(cfg_first(cfg, dotted_paths, default))


def cfg_float(cfg: dict[str, Any], dotted_paths: list[str] | tuple[str, ...], default: float) -> float:
    return float(cfg_first(cfg, dotted_paths, default))


def cfg_bool(cfg: dict[str, Any], dotted_paths: list[str] | tuple[str, ...], default: bool) -> bool:
    value = cfg_first(cfg, dotted_paths, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def cfg_seed(cfg: dict[str, Any], override: int | None = None) -> int:
    return int(override if override is not None else cfg_first(cfg, ("seed", "training.seed"), 7))


def cfg_device(cfg: dict[str, Any], override: str | None = None) -> str:
    return str(override if override is not None else cfg_first(cfg, ("device", "training.device"), "auto"))


def dataset_category(cfg: dict[str, Any], override: str | None = None) -> str:
    if override:
        return override
    category = cfg_get(cfg, "dataset.category", None)
    if category:
        return str(category)
    categories = cfg_get(cfg, "dataset.categories", None)
    if isinstance(categories, list) and categories:
        return str(categories[0])
    if isinstance(categories, str) and categories != "all":
        return categories
    raise ValueError("No category supplied and config does not define dataset.category or a concrete dataset.categories value.")


def image_size(cfg: dict[str, Any], override: int | None = None, checkpoint_value: Any = None) -> int:
    if override is not None:
        return int(override)
    if checkpoint_value is not None:
        return int(checkpoint_value)
    return cfg_int(cfg, ("dataset.image_size",), 256)


def diffusion_base_channels(cfg: dict[str, Any], checkpoint_value: Any = None) -> int:
    if checkpoint_value is not None:
        return int(checkpoint_value)
    return cfg_int(cfg, ("model.base_channels", "diffusion.base_channels"), 32)


def diffusion_timesteps(cfg: dict[str, Any], checkpoint_value: Any = None) -> int:
    if checkpoint_value is not None:
        return int(checkpoint_value)
    return cfg_int(cfg, ("model.diffusion_timesteps", "diffusion.timesteps"), 100)


def reconstruction_steps(cfg: dict[str, Any]) -> int:
    return cfg_int(cfg, ("model.reconstruction_steps", "diffusion.reconstruct_steps"), 5)


def max_regions(cfg: dict[str, Any]) -> int:
    return cfg_int(cfg, ("model.max_regions", "roi.max_rois"), 5)


def patch_size(cfg: dict[str, Any]) -> int:
    return cfg_int(cfg, ("hn_sev.patch_size", "sev.patch_size", "roi.patch_size"), 64)


def latency_budget_ms(cfg: dict[str, Any]) -> float:
    return cfg_float(cfg, ("lc_rds.latency_budget_ms", "scheduler.latency_budget_ms"), 60.0)


def resolve_device(name: str = "auto") -> str:
    if name == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return name


def make_run_dir(cfg: dict[str, Any], run_name: str | None) -> Path:
    root = Path(cfg_first(cfg, ("output.root", "runs.root"), "runs"))
    name = run_name or "default"
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path
