from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.collect_hardware_profile import (
    collect_profile,
    nvidia_smi_query,
    write_profile,
)


def test_nvidia_smi_query_parses_gpu_rows(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = "GPU A, 555.12, 24576, 320.0\nGPU B, 555.12, [N/A], [N/A]\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Result())

    info = nvidia_smi_query()

    assert info["available"] is True
    assert len(info["gpus"]) == 2
    assert info["gpus"][0]["name"] == "GPU A"
    assert info["gpus"][0]["memory_total_mb"] == 24576.0
    assert info["gpus"][1]["power_limit_watts"] is None


def test_collect_profile_uses_mocked_torch_and_smi(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.collect_hardware_profile.torch_profile",
        lambda: {
            "available": True,
            "version": "2.4.0",
            "cuda_available": True,
            "cuda_version": "12.4",
            "device": "cuda",
            "gpu_name": "GPU A",
        },
    )
    monkeypatch.setattr(
        "tools.collect_hardware_profile.nvidia_smi_query",
        lambda: {"available": True, "gpus": [{"name": "GPU A"}]},
    )

    profile = collect_profile("second-box", validation_run="runs/probe")

    assert profile["schema"] == "lite-seer-ad-hardware-profile-v1"
    assert profile["label"] == "second-box"
    assert profile["device"] == "cuda"
    assert profile["gpu_name"] == "GPU A"
    assert profile["validation_run"] == "runs/probe"


def test_write_profile_creates_json(tmp_path: Path) -> None:
    output = tmp_path / "profiles" / "profile.json"

    write_profile(output, {"schema": "x", "label": "unit"})

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == {"schema": "x", "label": "unit"}
