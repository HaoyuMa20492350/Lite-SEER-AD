"""Measure GPU energy for a production command with nvidia-smi polling.

The exporter in ``tools/export_production_deployment_latency.py`` consumes JSON
sidecars from ``tables/deployment_production_latency/energy_measurements``.
This wrapper creates those sidecars without inventing energy values: it runs the
given command, polls ``nvidia-smi`` for power draw, integrates watt-seconds, and
records the command return code.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "lite-seer-ad-production-energy-measurement-v1"


def parse_float(value: str, default: float = 0.0) -> float:
    text = str(value).strip().replace(" W", "").replace(" MiB", "")
    if text.upper() in {"N/A", "[N/A]", ""}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_nvidia_smi_line(line: str, elapsed_sec: float) -> dict[str, Any]:
    parts = [part.strip() for part in line.split(",")]
    power = parse_float(parts[0]) if parts else 0.0
    memory = parse_float(parts[1]) if len(parts) > 1 else 0.0
    gpu_name = ",".join(parts[2:]).strip() if len(parts) > 2 else ""
    return {
        "elapsed_sec": float(elapsed_sec),
        "power_watts": power,
        "memory_used_mb": memory,
        "gpu_name": gpu_name,
    }


def poll_nvidia_smi(gpu_index: int, elapsed_sec: float) -> dict[str, Any] | None:
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu=power.draw,memory.used,name",
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
        return None
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return parse_nvidia_smi_line(lines[0], elapsed_sec)


def integrate_energy(samples: list[dict[str, Any]], duration_sec: float) -> float:
    ordered = sorted(
        (
            {
                "elapsed_sec": max(0.0, float(sample.get("elapsed_sec", 0.0))),
                "power_watts": max(0.0, float(sample.get("power_watts", 0.0))),
            }
            for sample in samples
        ),
        key=lambda item: item["elapsed_sec"],
    )
    if not ordered or duration_sec <= 0.0:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]["power_watts"] * duration_sec

    energy = 0.0
    previous = ordered[0]
    for current in ordered[1:]:
        dt = max(0.0, current["elapsed_sec"] - previous["elapsed_sec"])
        energy += 0.5 * (previous["power_watts"] + current["power_watts"]) * dt
        previous = current
    tail = max(0.0, duration_sec - previous["elapsed_sec"])
    energy += previous["power_watts"] * tail
    return energy


def summarize_samples(samples: list[dict[str, Any]], duration_sec: float) -> dict[str, Any]:
    powers = [float(sample.get("power_watts", 0.0)) for sample in samples]
    memories = [float(sample.get("memory_used_mb", 0.0)) for sample in samples]
    gpu_names = [str(sample.get("gpu_name") or "") for sample in samples if sample.get("gpu_name")]
    energy = integrate_energy(samples, duration_sec)
    return {
        "sample_count": len(samples),
        "duration_sec": float(duration_sec),
        "energy_joules": float(energy),
        "power_watts_mean": float(sum(powers) / len(powers)) if powers else 0.0,
        "power_watts_max": float(max(powers)) if powers else 0.0,
        "memory_used_mb_max": float(max(memories)) if memories else 0.0,
        "gpu_name": gpu_names[0] if gpu_names else "",
    }


def run_measured_command(
    command: list[str],
    *,
    output: Path,
    label: str,
    gpu_index: int,
    interval_sec: float,
    cwd: Path,
) -> dict[str, Any]:
    if not command:
        raise ValueError("command must not be empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start = time.perf_counter()
    samples: list[dict[str, Any]] = []
    first_sample = poll_nvidia_smi(gpu_index, 0.0)
    if first_sample is not None:
        samples.append(first_sample)

    process = subprocess.Popen(command, cwd=str(cwd))
    while process.poll() is None:
        time.sleep(max(interval_sec, 0.1))
        elapsed = time.perf_counter() - start
        sample = poll_nvidia_smi(gpu_index, elapsed)
        if sample is not None:
            samples.append(sample)
    returncode = int(process.returncode or 0)
    duration = time.perf_counter() - start
    final_sample = poll_nvidia_smi(gpu_index, duration)
    if final_sample is not None:
        samples.append(final_sample)

    summary = summarize_samples(samples, duration)
    record = {
        "schema": SCHEMA,
        "protocol": "nvidia_smi_power_poll_v1",
        "label": label,
        "command": command,
        "returncode": returncode,
        "started_utc": started_wall,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cwd": str(cwd),
        "gpu_index": gpu_index,
        "poll_interval_sec": float(interval_sec),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "samples": samples,
        **summary,
    }
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="production_energy_measurement")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval-sec", type=float, default=0.5)
    parser.add_argument("--cwd", type=Path, default=Path("."))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main() -> None:
    args = parse_args()
    record = run_measured_command(
        [str(part) for part in args.command],
        output=args.output,
        label=args.label,
        gpu_index=args.gpu_index,
        interval_sec=args.interval_sec,
        cwd=args.cwd,
    )
    print(
        "Measured production energy: "
        f"energy_joules={record['energy_joules']:.3f} "
        f"samples={record['sample_count']} output={args.output}"
    )


if __name__ == "__main__":
    main()
