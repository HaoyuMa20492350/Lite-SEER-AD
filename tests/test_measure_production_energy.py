from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.measure_production_energy import (
    integrate_energy,
    parse_nvidia_smi_line,
    run_measured_command,
    summarize_samples,
)


def test_parse_nvidia_smi_line_handles_numeric_and_na_fields() -> None:
    row = parse_nvidia_smi_line("32.5, 1024, NVIDIA GeForce RTX 4090 Laptop GPU", 1.25)

    assert row["elapsed_sec"] == 1.25
    assert row["power_watts"] == 32.5
    assert row["memory_used_mb"] == 1024.0
    assert row["gpu_name"] == "NVIDIA GeForce RTX 4090 Laptop GPU"

    na = parse_nvidia_smi_line("[N/A], [N/A], GPU", 0.0)
    assert na["power_watts"] == 0.0
    assert na["memory_used_mb"] == 0.0


def test_integrate_energy_uses_trapezoids_and_tail() -> None:
    samples = [
        {"elapsed_sec": 0.0, "power_watts": 10.0},
        {"elapsed_sec": 1.0, "power_watts": 20.0},
    ]

    assert integrate_energy(samples, duration_sec=2.0) == pytest.approx(35.0)


def test_summarize_samples_reports_positive_energy() -> None:
    summary = summarize_samples(
        [
            {
                "elapsed_sec": 0.0,
                "power_watts": 30.0,
                "memory_used_mb": 100.0,
                "gpu_name": "gpu-a",
            },
            {
                "elapsed_sec": 1.0,
                "power_watts": 40.0,
                "memory_used_mb": 200.0,
                "gpu_name": "gpu-a",
            },
        ],
        duration_sec=1.0,
    )

    assert summary["sample_count"] == 2
    assert summary["energy_joules"] == pytest.approx(35.0)
    assert summary["power_watts_mean"] == pytest.approx(35.0)
    assert summary["power_watts_max"] == 40.0
    assert summary["memory_used_mb_max"] == 200.0


def test_run_measured_command_writes_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tools.measure_production_energy as module

    counter = {"value": 0}

    def fake_poll(gpu_index: int, elapsed_sec: float) -> dict:
        counter["value"] += 1
        return {
            "elapsed_sec": elapsed_sec,
            "power_watts": 25.0 + counter["value"],
            "memory_used_mb": 512.0,
            "gpu_name": "fake-gpu",
        }

    monkeypatch.setattr(module, "poll_nvidia_smi", fake_poll)
    output = tmp_path / "energy.json"

    record = run_measured_command(
        ["python", "-c", "print('ok')"],
        output=output,
        label="unit",
        gpu_index=0,
        interval_sec=0.1,
        cwd=tmp_path,
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["schema"] == "lite-seer-ad-production-energy-measurement-v1"
    assert persisted["label"] == "unit"
    assert persisted["returncode"] == 0
    assert persisted["sample_count"] >= 2
    assert persisted["energy_joules"] > 0.0
    assert record["energy_joules"] == persisted["energy_joules"]


def test_run_measured_command_writes_failed_returncode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tools.measure_production_energy as module

    monkeypatch.setattr(
        module,
        "poll_nvidia_smi",
        lambda gpu_index, elapsed_sec: {
            "elapsed_sec": elapsed_sec,
            "power_watts": 10.0,
            "memory_used_mb": 0.0,
            "gpu_name": "fake-gpu",
        },
    )
    output = tmp_path / "failed.json"

    with pytest.raises(subprocess.CalledProcessError):
        run_measured_command(
            ["python", "-c", "raise SystemExit(3)"],
            output=output,
            label="failed",
            gpu_index=0,
            interval_sec=0.1,
            cwd=tmp_path,
        )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["returncode"] == 3
