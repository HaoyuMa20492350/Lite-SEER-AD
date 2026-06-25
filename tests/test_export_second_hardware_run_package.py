from __future__ import annotations

import json
from pathlib import Path

from tools.export_second_hardware_run_package import default_infer_command, write_outputs


def test_default_infer_command_records_production_budget_probe() -> None:
    command = default_infer_command(
        category="bottle",
        run_name="second_hardware_probe",
        budget_ms=10,
        max_samples=8,
        seed=7,
        image_size=128,
    )

    assert command[:2] == ["python", "infer.py"]
    assert "--ablation" in command
    assert "utility_lc_rds" in command
    assert "--latency-budget-ms" in command
    assert "10" in command
    assert "--reconstruction-steps" in command
    assert "5" in command


def test_write_outputs_creates_runnable_second_hardware_package(tmp_path: Path) -> None:
    manifest = write_outputs(tmp_path)

    assert manifest["schema"] == "lite-seer-ad-second-hardware-run-package-v1"
    assert manifest["expected_return_files"] == [
        "second_hardware_return_package/second_hardware_probe_energy.json",
        "second_hardware_return_package/second_hardware_probe_hardware_profile.json",
    ]
    assert (tmp_path / "run_second_hardware_probe.ps1").is_file()
    assert (tmp_path / "validate_returned_second_hardware_package.ps1").is_file()
    assert (tmp_path / "README.md").is_file()
    assert (tmp_path / "manifest.json").is_file()

    loaded = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert loaded["infer_command"] == manifest["infer_command"]


def test_generated_scripts_include_measure_collect_and_stage_commands(tmp_path: Path) -> None:
    write_outputs(tmp_path)

    run_script = (tmp_path / "run_second_hardware_probe.ps1").read_text(encoding="utf-8")
    validate_script = (tmp_path / "validate_returned_second_hardware_package.ps1").read_text(encoding="utf-8")
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert "tools/measure_production_energy.py" in run_script
    assert "tools/collect_hardware_profile.py" in run_script
    assert "RepoRoot" in run_script
    assert "Push-Location $RepoRoot" in run_script
    assert "Resolve-Path -LiteralPath $RepoRoot" in run_script
    assert "[System.IO.Path]::IsPathRooted($ReturnDir)" in run_script
    assert "[System.IO.Path]::GetFullPath($ReturnDir)" in run_script
    assert "Join-Path $RepoRoot $ReturnDir" in run_script
    assert "Pop-Location" in run_script
    assert "--max-samples" in run_script
    assert "tools/validate_second_hardware_package.py" in validate_script
    assert "--stage" in validate_script
    assert "RepoRoot" in validate_script
    assert "Resolve-Path -LiteralPath $RepoRoot" in validate_script
    assert "[System.IO.Path]::IsPathRooted($ReturnDir)" in validate_script
    assert "Test-Path -LiteralPath $ReturnDir" in validate_script
    assert "Join-Path $RepoRoot $ReturnDir" in validate_script
    assert "Push-Location $RepoRoot" in validate_script
    assert "export_deployment_readiness.py" in validate_script
    assert "validate_returned_second_hardware_package.ps1" in readme
    assert "absolute-or-repo-root-relative-return-directory" in readme
    assert "relative values are created under `RepoRoot`" in readme
    assert "absolute-or-caller-relative-return-directory" in readme
    assert "falls back to `<RepoRoot>/second_hardware_return_package`" in readme


def test_run_script_resolves_return_dir_before_repo_root_push(tmp_path: Path) -> None:
    write_outputs(tmp_path)

    run_script = (tmp_path / "run_second_hardware_probe.ps1").read_text(encoding="utf-8")

    return_dir_index = run_script.index("[System.IO.Path]::IsPathRooted($ReturnDir)")
    push_index = run_script.index("Push-Location $RepoRoot")
    create_index = run_script.index("New-Item -ItemType Directory -Force -Path $ReturnDir")

    assert return_dir_index < push_index < create_index


def test_validate_script_resolves_return_dir_before_repo_root_push(tmp_path: Path) -> None:
    write_outputs(tmp_path)

    validate_script = (tmp_path / "validate_returned_second_hardware_package.ps1").read_text(encoding="utf-8")

    return_dir_index = validate_script.index("[System.IO.Path]::IsPathRooted($ReturnDir)")
    push_index = validate_script.index("Push-Location $RepoRoot")
    energy_index = validate_script.index("$EnergyPath = Join-Path $ReturnDir")

    assert return_dir_index < push_index < energy_index
