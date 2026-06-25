from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate_second_hardware_package import build_rows, build_summary, write_outputs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def existing_profile(root: Path) -> None:
    write_json(
        root / "tables/deployment_latency/summary.json",
        {
            "hardware": {
                "platform": "Windows",
                "processor": "cpu-a",
                "cpu_count": 8,
                "device": "cuda",
                "gpu_name": "gpu-a",
                "python": "3.12",
                "torch": "2.4",
            }
        },
    )


def profile_payload(**updates) -> dict:
    payload = {
        "schema": "lite-seer-ad-hardware-profile-v1",
        "label": "second_gpu",
        "platform": "Linux",
        "processor": "cpu-b",
        "cpu_count": 16,
        "python": "3.12",
        "torch": "2.4",
        "device": "cuda",
        "gpu_name": "gpu-b",
    }
    payload.update(updates)
    return payload


def energy_payload(**updates) -> dict:
    payload = {
        "schema": "lite-seer-ad-production-energy-measurement-v1",
        "returncode": 0,
        "energy_joules": 12.5,
        "sample_count": 3,
        "command": ["python", "infer.py", "--config", "configs/mvtec.yaml"],
        "platform": "Linux",
        "python": "3.12",
        "gpu_name": "gpu-b",
    }
    payload.update(updates)
    return payload


def test_second_hardware_package_passes_for_distinct_profile(tmp_path: Path) -> None:
    existing_profile(tmp_path)
    profile = tmp_path / "incoming/second_profile.json"
    energy = tmp_path / "incoming/second_energy.json"
    write_json(profile, profile_payload())
    write_json(energy, energy_payload())

    rows = build_rows(tmp_path, profile, energy)
    summary = build_summary(rows)

    assert summary["validation_passed"] is True
    assert summary["counts"] == {"pass": len(rows)}


def test_second_hardware_package_rejects_duplicate_profile(tmp_path: Path) -> None:
    existing_profile(tmp_path)
    profile = tmp_path / "incoming/duplicate_profile.json"
    energy = tmp_path / "incoming/second_energy.json"
    write_json(
        profile,
        profile_payload(
            platform="Windows",
            processor="cpu-a",
            cpu_count=8,
            gpu_name="gpu-a",
        ),
    )
    write_json(energy, energy_payload())

    rows = build_rows(tmp_path, profile, energy)
    summary = build_summary(rows)

    assert summary["validation_passed"] is False
    assert "hardware:distinct_from_existing_profiles" in summary["blocking_requirements"]


def test_write_outputs_can_stage_valid_package(tmp_path: Path) -> None:
    existing_profile(tmp_path)
    profile = tmp_path / "incoming/second_profile.json"
    energy = tmp_path / "incoming/second_energy.json"
    write_json(profile, profile_payload())
    write_json(energy, energy_payload())

    summary = write_outputs(
        tmp_path,
        tmp_path / "tables/deployment_production_latency/second_hardware_validation",
        profile,
        energy,
        stage=True,
    )

    assert summary["validation_passed"] is True
    assert (tmp_path / summary["staged_outputs"]["hardware_profile"]).is_file()
    assert (tmp_path / summary["staged_outputs"]["energy_measurement"]).is_file()


def test_write_outputs_refuses_to_stage_invalid_package(tmp_path: Path) -> None:
    existing_profile(tmp_path)
    profile = tmp_path / "incoming/second_profile.json"
    energy = tmp_path / "incoming/second_energy.json"
    write_json(profile, profile_payload())
    write_json(energy, energy_payload(energy_joules=0.0))

    with pytest.raises(ValueError):
        write_outputs(
            tmp_path,
            tmp_path / "tables/deployment_production_latency/second_hardware_validation",
            profile,
            energy,
            stage=True,
        )


def test_second_hardware_package_rejects_mismatched_profile_and_energy(tmp_path: Path) -> None:
    existing_profile(tmp_path)
    profile = tmp_path / "incoming/second_profile.json"
    energy = tmp_path / "incoming/second_energy.json"
    write_json(profile, profile_payload())
    write_json(energy, energy_payload(gpu_name="different-gpu"))

    rows = build_rows(tmp_path, profile, energy)
    summary = build_summary(rows)

    assert summary["validation_passed"] is False
    assert "hardware_energy:gpu_name_match" in summary["blocking_requirements"]
