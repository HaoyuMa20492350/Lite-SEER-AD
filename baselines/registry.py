from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    display_name: str
    role: str
    implementation_variant: str
    official_implementation: bool = False
    source_path: str = ""
    reference_key: str = ""
    expected_artifact: str = "predictions.npz"
    local_runner: bool = False


BASELINES = {
    "patchcore": BaselineSpec(
        "patchcore",
        "PatchCore-Local",
        "memory/prototype",
        "local_reimplementation",
        source_path="baselines/run_baseline.py",
        reference_key="Roth2022PatchCore",
        local_runner=True,
    ),
    "padim": BaselineSpec(
        "padim",
        "PaDiM-Local",
        "memory/statistical",
        "local_reimplementation",
        source_path="baselines/run_baseline.py",
        reference_key="Defard2021PaDiM",
        local_runner=True,
    ),
    "simplenet": BaselineSpec(
        "simplenet",
        "SimpleNet-Lite",
        "efficient-ad",
        "lite_reimplementation",
        source_path="baselines/run_baseline.py",
        reference_key="Liu2023SimpleNet",
        local_runner=True,
    ),
    "draem": BaselineSpec(
        "draem",
        "DRAEM-Lite",
        "reconstruction/synthetic",
        "lite_reimplementation",
        source_path="baselines/run_baseline.py",
        reference_key="Zavrtanik2021DRAEM",
        local_runner=True,
    ),
    "rd4ad": BaselineSpec(
        "rd4ad",
        "RD4AD-Lite",
        "distillation",
        "lite_reimplementation",
        source_path="baselines/run_baseline.py",
        reference_key="Deng2022RD4AD",
        local_runner=True,
    ),
    "uniad": BaselineSpec(
        "uniad",
        "UniAD-Lite",
        "unified-transformer",
        "lite_reimplementation",
        source_path="baselines/run_baseline.py",
        reference_key="You2022UniAD",
        local_runner=True,
    ),
    "diffusionad": BaselineSpec(
        "diffusionad",
        "DiffusionAD-Lite",
        "diffusion",
        "lite_reimplementation",
        source_path="baselines/run_baseline.py",
        reference_key="Zhang2023DiffusionAD",
        local_runner=True,
    ),
    "ddad": BaselineSpec(
        "ddad",
        "DDAD-Lite",
        "diffusion",
        "lite_reimplementation",
        source_path="baselines/run_baseline.py",
        reference_key="Mousakhan2023DDAD",
        local_runner=True,
    ),
    "fastflow": BaselineSpec(
        "fastflow",
        "FastFlow-External",
        "flow",
        "external_predictions_unverified",
    ),
    "invad": BaselineSpec(
        "invad",
        "InvAD-External",
        "optional-diffusion",
        "external_predictions_unverified",
        reference_key="Yao2025InvAD",
    ),
}

REQUIRED_MVTEC15_BASELINES = ("patchcore", "padim", "simplenet", "draem", "rd4ad", "uniad", "diffusionad", "ddad")
LOCAL_BASELINES = tuple(name for name, spec in BASELINES.items() if spec.local_runner)


def require_baseline_artifact(name: str, run_dir: str | Path) -> Path:
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline: {name}")
    path = Path(run_dir) / BASELINES[name].expected_artifact
    if not path.exists():
        raise FileNotFoundError(
            f"{name} artifact not found at {path}. Baseline adapters are contracts only in this MVP: "
            "run or export the third-party method so the directory contains predictions.npz with "
            "labels, image_scores, masks, and heatmaps arrays."
        )
    return path
