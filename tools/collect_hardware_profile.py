"""Collect a hardware profile sidecar for deployment cross-hardware audits.

The production deployment exporter reads JSON files from
``tables/deployment_production_latency/hardware_profiles``. Run this script on a
second machine after a production probe to add a distinct hardware profile
without hand-editing summary tables.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


SCHEMA = "lite-seer-ad-hardware-profile-v1"


def nvidia_smi_query() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "gpus": []}
    if result.returncode != 0:
        return {"available": False, "gpus": [], "error": result.stderr.strip()}

    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "name": parts[0] if len(parts) > 0 else "",
                "driver_version": parts[1] if len(parts) > 1 else "",
                "memory_total_mb": _float_or_none(parts[2]) if len(parts) > 2 else None,
                "power_limit_watts": _float_or_none(parts[3]) if len(parts) > 3 else None,
            }
        )
    return {"available": True, "gpus": gpus}


def _float_or_none(value: str) -> float | None:
    text = str(value).strip()
    if text.upper() in {"", "N/A", "[N/A]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def torch_profile() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        return {"available": False, "error": type(exc).__name__}

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else ""
    return {
        "available": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": str(torch.version.cuda or ""),
        "device": "cuda" if cuda_available else "cpu",
        "gpu_name": gpu_name,
    }


def collect_profile(label: str, validation_run: str | None = None) -> dict[str, Any]:
    torch_info = torch_profile()
    smi = nvidia_smi_query()
    profile = {
        "schema": SCHEMA,
        "label": label,
        "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch_info.get("version", ""),
        "cuda_available": bool(torch_info.get("cuda_available", False)),
        "cuda_version": torch_info.get("cuda_version", ""),
        "device": torch_info.get("device", "cpu"),
        "gpu_name": torch_info.get("gpu_name", ""),
        "nvidia_smi": smi,
    }
    if validation_run:
        profile["validation_run"] = validation_run
    return profile


def write_profile(path: Path, profile: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tables/deployment_production_latency/hardware_profiles/current.json"),
    )
    parser.add_argument("--label", default="current_hardware")
    parser.add_argument("--validation-run", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = collect_profile(args.label, validation_run=args.validation_run)
    write_profile(args.output, profile)
    print(
        "Wrote hardware profile: "
        f"output={args.output} device={profile.get('device')} gpu={profile.get('gpu_name')}"
    )


if __name__ == "__main__":
    main()
