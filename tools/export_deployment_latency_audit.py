"""Export a synchronized deployment latency audit.

This audit measures batch=1 component smoke callables with the repository's
CUDA-synchronized timing helper. It provides the required table shape and a
strict evidence label, while keeping the release gate closed until production
detector/verifier/repair components are measured end-to-end.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.metrics_efficiency import benchmark_callable
from seer_ad_v2.models.scheduler.lc_rds import ExpectedUtilityScheduler


IMAGE_SIZE = (128, 128)
COMPONENTS = ["io", "detector", "verifier", "scheduler", "repair", "end_to_end"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_sample_image(root: Path) -> Path | None:
    patterns = [
        "SEER-AD-dataset/MVTec-AD/*/train/good/*",
        "SEER-AD-dataset/VisA/*/Data/Images/Normal/*",
        "SEER-AD-dataset/MPDD/official/MPDD/MPDD/*/train/good/*",
    ]
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                return path
    return None


def load_image_tensor(path: Path | None, device: str) -> torch.Tensor:
    if path is not None and path.exists():
        image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    else:
        yy, xx = np.mgrid[: IMAGE_SIZE[1], : IMAGE_SIZE[0]]
        arr = np.stack(
            [
                (xx % 31) / 31.0,
                (yy % 37) / 37.0,
                ((xx + yy) % 41) / 41.0,
            ],
            axis=-1,
        ).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor.contiguous()


def synthetic_mask(device: str) -> torch.Tensor:
    mask = torch.zeros((1, 1, IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=torch.float32, device=device)
    mask[:, :, 42:78, 44:82] = 1.0
    return mask


def make_callables(image: torch.Tensor, mask: torch.Tensor) -> dict[str, Callable[[], Any]]:
    kernels = {
        "sobel_x": torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            device=image.device,
        ).repeat(3, 1, 1, 1),
        "sobel_y": torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            device=image.device,
        ).repeat(3, 1, 1, 1),
    }
    scheduler = ExpectedUtilityScheduler(latency_budget_ms=75.0)
    features = np.asarray([0.04, 0.2, 0.2, 0.8, 0.7, 0.6, 0.5, 4.0], dtype=np.float32)

    def io_call() -> torch.Tensor:
        return image.clone()

    def detector_call() -> torch.Tensor:
        gx = F.conv2d(image, kernels["sobel_x"], padding=1, groups=3)
        gy = F.conv2d(image, kernels["sobel_y"], padding=1, groups=3)
        return torch.sqrt(gx.square() + gy.square() + 1e-8).mean(dim=1, keepdim=True)

    def verifier_call() -> torch.Tensor:
        heatmap = detector_call()
        roi = heatmap[:, :, 42:78, 44:82]
        return torch.sigmoid(torch.stack([roi.mean(), roi.max(), roi.std()]).sum())

    def scheduler_call() -> Any:
        return scheduler.choose(features, spent_ms=0.0, expected_gain=0.35)

    def repair_call() -> torch.Tensor:
        blurred = F.avg_pool2d(image, kernel_size=9, stride=1, padding=4)
        return image * (1.0 - mask) + blurred * mask

    def end_to_end_call() -> torch.Tensor:
        _ = io_call()
        heatmap = detector_call()
        _ = verifier_call()
        _ = scheduler_call()
        repaired = repair_call()
        return repaired + 0.0 * heatmap.mean()

    return {
        "io": io_call,
        "detector": detector_call,
        "verifier": verifier_call,
        "scheduler": scheduler_call,
        "repair": repair_call,
        "end_to_end": end_to_end_call,
    }


def hardware_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_memory_mb": 0.0,
        "energy_joules": None,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_memory_mb"] = float(torch.cuda.max_memory_allocated(idx) / (1024**2))
    return info


def benchmark_components(
    root: Path,
    *,
    device: str,
    warmups: int,
    repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample = collect_sample_image(root)
    image = load_image_tensor(sample, device)
    mask = synthetic_mask(device)
    callables = make_callables(image, mask)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rows = []
    for component in COMPONENTS:
        result = benchmark_callable(
            callables[component],
            device=device,
            warmups=warmups,
            repeats=repeats,
            batch_size=1,
        )
        rows.append({"component": component, **result})

    info = hardware_info(device)
    info["sample_image"] = sample.as_posix() if sample is not None else None
    return rows, info


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


def write_outputs(
    root: Path,
    out_dir: Path,
    *,
    device: str,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    rows, info = benchmark_components(root, device=device, warmups=warmups, repeats=repeats)
    budget = read_json(root / "tables/lc_rds_budget_audit/summary.json")
    end_to_end = next(row for row in rows if row["component"] == "end_to_end")
    summary = {
        "schema": "lite-seer-ad-deployment-latency-audit-v1",
        "evidence_level": "synchronized_component_smoke_v1",
        "release_gate_passed": False,
        "release_gate_reason": (
            "This table is a synchronized batch=1 component smoke benchmark. "
            "Production detector/verifier/scheduler/repair/IO timing and cross-hardware "
            "measurements are still required for deployment claims."
        ),
        "latency_protocol": "synchronized_batch_latency_v1",
        "latency_batch_size": 1,
        "latency_warmups": warmups,
        "latency_repeats": repeats,
        "components": COMPONENTS,
        "latency_ms_mean": end_to_end["latency_ms_mean"],
        "latency_ms_p50": end_to_end["latency_ms_p50"],
        "latency_ms_p95": end_to_end["latency_ms_p95"],
        "latency_ms_p99": end_to_end["latency_ms_p99"],
        "fps": end_to_end["fps"],
        "gpu_memory_mb": info["gpu_memory_mb"],
        "energy_joules": info["energy_joules"],
        "budget_violation_rate": budget.get("max_budget_violation_rate"),
        "budget_violation_source": "tables/lc_rds_budget_audit/summary.json",
        "hardware": info,
        "required_for_release": [
            "production detector/verifier/scheduler/repair/IO timing breakdown",
            "true LC-RDS measured budget sweep with wall-clock violation rate",
            "at least one primary hardware run; preferably one cross-hardware run",
            "GPU memory measured for the production model",
            "energy measurement or explicit not-measured statement required by the target venue",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_component_latency.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("tables/deployment_latency"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    start = time.perf_counter()
    summary = write_outputs(
        args.root,
        args.out_dir,
        device=args.device,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    elapsed = time.perf_counter() - start
    print(
        f"Wrote deployment latency audit to {args.out_dir} "
        f"(release_gate_passed={summary['release_gate_passed']}, elapsed={elapsed:.2f}s)"
    )


if __name__ == "__main__":
    main()
