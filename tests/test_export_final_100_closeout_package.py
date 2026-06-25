from __future__ import annotations

import json
from pathlib import Path

from tools.export_final_100_closeout_package import (
    FINALIZATION_COMMANDS,
    OPTIONAL_TEST_COMMAND,
    SUCCESS_EVIDENCE,
    build_manifest,
    write_outputs,
)


def test_manifest_declares_all_external_inputs_and_success_evidence() -> None:
    manifest = build_manifest()

    assert manifest["schema"] == "lite-seer-ad-final-100-closeout-package-v1"
    assert manifest["required_external_inputs"] == [
        "second_hardware_profile.json",
        "second_hardware_energy.json",
        "release_metadata.json",
        "submission_metadata.json",
    ]
    assert SUCCESS_EVIDENCE == manifest["success_evidence"]
    assert any("packet_ready=true" in item for item in manifest["success_evidence"])
    assert any("final_input_packet_valid=true" in item for item in manifest["success_evidence"])
    assert any("input_packet_matches_external_handoff=pass" in item for item in manifest["success_evidence"])
    assert any("closeout:package_manifest_valid=pass" in item for item in manifest["success_evidence"])
    assert any("closeout:package_artifacts_match_manifest=pass" in item for item in manifest["success_evidence"])
    assert any("closeout:required_external_inputs_present=pass" in item for item in manifest["success_evidence"])
    assert any("closeout:success_evidence_expectations=pass" in item for item in manifest["success_evidence"])
    assert any("consistency:manuscript_title=pass" in item for item in manifest["success_evidence"])
    assert any("consistency:citation_authors=pass" in item for item in manifest["success_evidence"])
    assert any("consistency:zenodo_creators=pass" in item for item in manifest["success_evidence"])
    assert any("final_manifest_coverage_ready=true" in item for item in manifest["success_evidence"])
    assert any("default_100_ready=true" in item for item in manifest["success_evidence"])


def test_finalization_commands_stage_external_inputs_before_readiness_refresh() -> None:
    assert FINALIZATION_COMMANDS[0].startswith("python tools/validate_second_hardware_package.py")
    assert FINALIZATION_COMMANDS[1] == "python tools/validate_release_submission_consistency.py --fail-on-inconsistent"
    assert FINALIZATION_COMMANDS[2] == "python tools/render_release_metadata.py --input release_metadata.json"
    assert "python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json" in FINALIZATION_COMMANDS
    handoff_index = FINALIZATION_COMMANDS.index("python tools/export_final_external_handoff.py")
    input_packet_index = FINALIZATION_COMMANDS.index("python tools/export_final_input_packet.py")
    input_packet_validation_index = FINALIZATION_COMMANDS.index(
        "python tools/validate_final_input_packet.py --fail-on-invalid"
    )
    assert handoff_index < input_packet_index < input_packet_validation_index
    tail = FINALIZATION_COMMANDS[-8:]
    assert tail == [
        "python tools/validate_final_100_closeout.py",
        "python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json",
        "python tools/validate_final_manifest_coverage.py --fail-on-missing",
        "python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json",
        "python tools/validate_final_100_closeout.py --fail-on-incomplete",
        "python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json",
        "python tools/validate_final_manifest_coverage.py --fail-on-missing",
        "python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json",
    ]


def test_finalization_commands_rebuild_manifest_after_final_validation() -> None:
    manifest_indices = [
        index
        for index, command in enumerate(FINALIZATION_COMMANDS)
        if command == "python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json"
    ]
    preliminary_validation_index = FINALIZATION_COMMANDS.index("python tools/validate_final_100_closeout.py")
    strict_validation_index = FINALIZATION_COMMANDS.index(
        "python tools/validate_final_100_closeout.py --fail-on-incomplete"
    )
    coverage_indices = [
        index
        for index, command in enumerate(FINALIZATION_COMMANDS)
        if command == "python tools/validate_final_manifest_coverage.py --fail-on-missing"
    ]
    input_packet_index = FINALIZATION_COMMANDS.index("python tools/export_final_input_packet.py")
    input_packet_validation_index = FINALIZATION_COMMANDS.index("python tools/validate_final_input_packet.py --fail-on-invalid")

    assert len(manifest_indices) == 5
    assert len(coverage_indices) == 2
    assert manifest_indices[0] < FINALIZATION_COMMANDS.index("python tools/export_release_readiness.py")
    assert input_packet_index < input_packet_validation_index < preliminary_validation_index
    assert preliminary_validation_index < manifest_indices[1] < coverage_indices[0] < manifest_indices[2]
    assert manifest_indices[2] < strict_validation_index < manifest_indices[3] < coverage_indices[1] < manifest_indices[4]


def test_write_outputs_creates_final_closeout_package(tmp_path: Path) -> None:
    manifest = write_outputs(tmp_path)

    assert (tmp_path / "README.md").is_file()
    assert (tmp_path / "finalize_100_percent.ps1").is_file()
    assert (tmp_path / "completion_audit_template.md").is_file()
    assert (tmp_path / "manifest.json").is_file()
    loaded = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert loaded["required_external_inputs"] == manifest["required_external_inputs"]
    assert any("input_packet_matches_external_handoff=pass" in item for item in loaded["success_evidence"])
    assert any("closeout:package_manifest_valid=pass" in item for item in loaded["success_evidence"])
    assert any("closeout:package_artifacts_match_manifest=pass" in item for item in loaded["success_evidence"])
    assert any("closeout:required_external_inputs_present=pass" in item for item in loaded["success_evidence"])
    assert any("closeout:success_evidence_expectations=pass" in item for item in loaded["success_evidence"])
    assert any("consistency:manuscript_title=pass" in item for item in loaded["success_evidence"])
    assert any("consistency:citation_authors=pass" in item for item in loaded["success_evidence"])


def test_final_script_requires_inputs_and_runs_tests_by_default(tmp_path: Path) -> None:
    write_outputs(tmp_path)
    script = (tmp_path / "finalize_100_percent.ps1").read_text(encoding="utf-8")

    assert "SecondHardwareProfile" in script
    assert "SecondHardwareEnergy" in script
    assert "RepoRoot" in script
    assert "Push-Location $RepoRoot" in script
    assert "Pop-Location" in script
    assert "Resolve-Path -LiteralPath $SecondHardwareProfile" in script
    assert "Resolve-Path -LiteralPath $SecondHardwareEnergy" in script
    assert "Resolve-Path -LiteralPath $RepoRoot" in script
    assert "CanonicalSecondHardwareProfile" in script
    assert "CanonicalSecondHardwareEnergy" in script
    assert "Copy-Item -LiteralPath $SecondHardwareProfile -Destination $CanonicalSecondHardwareProfile -Force" in script
    assert "Copy-Item -LiteralPath $SecondHardwareEnergy -Destination $CanonicalSecondHardwareEnergy -Force" in script
    assert "Test-Path -LiteralPath $Path" in script
    assert "release_metadata.json" in script
    assert "submission_metadata.json" in script
    assert OPTIONAL_TEST_COMMAND in script
    assert "SkipTests" in script
    assert "validate_final_100_closeout.py --fail-on-incomplete" in script
    assert "export_final_input_packet.py" in script
    assert "validate_final_input_packet.py --fail-on-invalid" in script
    assert "validate_release_submission_consistency.py --fail-on-inconsistent" in script
    assert "validate_final_manifest_coverage.py --fail-on-missing" in script


def test_manifest_uses_workspace_local_pytest_basetemp() -> None:
    manifest = build_manifest()

    assert manifest["optional_test_command"] == OPTIONAL_TEST_COMMAND
    assert "--basetemp .tmp_pytest_final_100" in manifest["optional_test_command"]
    assert "-p no:cacheprovider" in manifest["optional_test_command"]


def test_final_script_resolves_second_hardware_paths_before_repo_root_push(tmp_path: Path) -> None:
    write_outputs(tmp_path)
    script = (tmp_path / "finalize_100_percent.ps1").read_text(encoding="utf-8")

    push_index = script.index("Push-Location $RepoRoot")
    profile_index = script.index("$SecondHardwareProfile = (Resolve-Path -LiteralPath $SecondHardwareProfile).Path")
    energy_index = script.index("$SecondHardwareEnergy = (Resolve-Path -LiteralPath $SecondHardwareEnergy).Path")

    assert profile_index < energy_index < push_index


def test_final_script_canonicalizes_second_hardware_inputs_before_validation(tmp_path: Path) -> None:
    write_outputs(tmp_path)
    script = (tmp_path / "finalize_100_percent.ps1").read_text(encoding="utf-8")

    copy_profile_index = script.index(
        "Copy-Item -LiteralPath $SecondHardwareProfile -Destination $CanonicalSecondHardwareProfile -Force"
    )
    copy_energy_index = script.index(
        "Copy-Item -LiteralPath $SecondHardwareEnergy -Destination $CanonicalSecondHardwareEnergy -Force"
    )
    validate_index = script.index("python tools/validate_second_hardware_package.py")
    manifest_index = script.index("python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json")

    assert copy_profile_index < validate_index
    assert copy_energy_index < validate_index
    assert validate_index < manifest_index


def test_readme_documents_external_input_path_resolution(tmp_path: Path) -> None:
    write_outputs(tmp_path)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert "absolute-or-caller-relative-second_hardware_profile.json" in readme
    assert "paths relative to the shell location where you launch the script" in readme


def test_audit_template_lists_input_packet_handoff_match(tmp_path: Path) -> None:
    write_outputs(tmp_path)
    audit = (tmp_path / "completion_audit_template.md").read_text(encoding="utf-8")

    assert "handoff:input_packet_matches_external_handoff=pass" in audit
    assert "table_final_100_closeout_validation.csv" in audit


def test_audit_template_lists_success_evidence_self_check(tmp_path: Path) -> None:
    write_outputs(tmp_path)
    audit = (tmp_path / "completion_audit_template.md").read_text(encoding="utf-8")

    assert "closeout:package_manifest_valid=pass" in audit
    assert "closeout:package_artifacts_match_manifest=pass" in audit
    assert "closeout:required_external_inputs_present=pass" in audit
    assert "closeout:success_evidence_expectations=pass" in audit
    assert "table_final_100_closeout_validation.csv" in audit


def test_audit_template_lists_release_author_consistency(tmp_path: Path) -> None:
    write_outputs(tmp_path)
    audit = (tmp_path / "completion_audit_template.md").read_text(encoding="utf-8")

    assert "consistency:citation_authors=pass" in audit
    assert "consistency:zenodo_creators=pass" in audit
    assert "table_final_metadata_consistency.csv" in audit


def test_audit_template_lists_release_title_consistency(tmp_path: Path) -> None:
    write_outputs(tmp_path)
    audit = (tmp_path / "completion_audit_template.md").read_text(encoding="utf-8")

    assert "consistency:manuscript_title=pass" in audit
    assert "table_final_metadata_consistency.csv" in audit
