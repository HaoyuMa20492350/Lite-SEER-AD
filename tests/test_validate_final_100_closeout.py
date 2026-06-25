from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.export_final_100_closeout_package import FINALIZATION_COMMANDS
from tools.validate_final_100_closeout import (
    EXPECTED_CLOSEOUT_PACKAGE_SCHEMA,
    REQUIRED_CLOSEOUT_COMMANDS,
    REQUIRED_CLOSEOUT_INPUTS,
    check_rows,
    main,
    write_outputs,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_minimal_closeout_package(root: Path, evidence_items: list[str] | None = None) -> None:
    package = root / "tables/final_external_handoff/final_100_closeout_package"
    evidence_items = evidence_items or ["tables/final_external_handoff/final_input_packet/summary.json::packet_ready=true"]
    write_json(
        package / "manifest.json",
        {
            "schema": EXPECTED_CLOSEOUT_PACKAGE_SCHEMA,
            "required_external_inputs": REQUIRED_CLOSEOUT_INPUTS,
            "commands": FINALIZATION_COMMANDS,
            "success_evidence": evidence_items,
        },
    )
    write_text(
        package / "README.md",
        "\n".join([*REQUIRED_CLOSEOUT_INPUTS, *evidence_items]) + "\n",
    )
    write_text(
        package / "finalize_100_percent.ps1",
        "\n".join(FINALIZATION_COMMANDS) + "\n",
    )
    audit_lines = []
    for evidence in evidence_items:
        source, expectation = evidence.split("::", 1)
        audit_lines.append(f"| `{expectation}` | `{source}` |")
    write_text(
        package / "completion_audit_template.md",
        "\n".join(audit_lines) + "\n",
    )


def make_summaries(root: Path, *, ready: bool) -> None:
    write_json(root / "second_hardware_profile.json", {"hardware_profile": {"platform": "test"}})
    write_json(root / "second_hardware_energy.json", {"energy_measurement": {"joules": 1.0}})
    write_json(root / "release_metadata.json", {"schema": "lite-seer-ad-release-metadata-v1"})
    write_json(root / "submission_metadata.json", {"target_journal": {"journal": "Pattern Recognition"}})
    unresolved = [] if ready else ["external_link:github_release_url"]
    write_json(
        root / "tables/final_external_handoff/final_input_packet/summary.json",
        {
            "packet_ready": True,
            "external_handoff_rows": 2,
            "unresolved_requirements": unresolved,
        },
    )
    write_json(
        root / "tables/final_external_handoff/final_input_packet_validation/summary.json",
        {"final_input_packet_valid": True},
    )
    write_json(
        root / "tables/final_external_handoff/final_manifest_coverage/summary.json",
        {"final_manifest_coverage_ready": ready},
    )
    write_json(
        root / "tables/final_external_handoff/final_metadata_consistency/summary.json",
        {"final_metadata_consistent": ready},
    )
    write_json(root / "tables/deployment_readiness/summary.json", {"production_deployment_ready": ready})
    write_json(root / "tables/release_readiness/summary.json", {"release_gate_passed": ready})
    write_json(root / "tables/submission_package_readiness/summary.json", {"final_upload_ready": ready})
    write_json(root / "tables/completion_gap_matrix/summary.json", {"default_100_ready": ready})
    write_json(
        root / "tables/final_external_handoff/summary.json",
        {
            "all_external_actions_complete": ready,
            "rows": 2,
            "unresolved_requirements": unresolved,
        },
    )
    write_minimal_closeout_package(root)


def test_check_rows_fails_when_any_required_gate_is_false(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    write_json(tmp_path / "tables/release_readiness/summary.json", {"release_gate_passed": False})

    rows = check_rows(tmp_path)

    by_requirement = {row["requirement"]: row for row in rows}
    assert by_requirement["public_release:release_gate_passed"]["status"] == "fail"
    assert by_requirement["deployment:production_deployment_ready"]["status"] == "pass"


def test_write_outputs_reports_final_100_ready_for_all_passing_gates(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)

    summary = write_outputs(tmp_path, tmp_path / "tables/final_external_handoff/final_100_closeout_validation")

    assert summary["final_100_ready"] is True
    assert summary["blocking_requirements"] == []
    assert "closeout:package_manifest_valid" in summary["checked_requirements"]
    assert "closeout:package_artifacts_match_manifest" in summary["checked_requirements"]
    assert "closeout:success_evidence_expectations" in summary["checked_requirements"]
    assert (tmp_path / "tables/final_external_handoff/final_100_closeout_validation/summary.json").is_file()
    assert (
        tmp_path
        / "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv"
    ).is_file()
    assert (
        tmp_path
        / "tables/final_external_handoff/final_100_closeout_validation/final_100_closeout_validation.md"
    ).is_file()


def test_write_outputs_reports_blockers_for_incomplete_current_state(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=False)

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert len(summary["blocking_requirements"]) == 7
    assert "handoff:final_manifest_coverage_ready" in summary["blocking_requirements"]
    assert "handoff:final_input_packet_ready" not in summary["blocking_requirements"]


def test_missing_final_input_packet_blocks_even_when_final_gates_pass(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    (tmp_path / "tables/final_external_handoff/final_input_packet/summary.json").unlink()

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert "handoff:final_input_packet_ready" in summary["blocking_requirements"]
    assert "handoff:input_packet_matches_external_handoff" in summary["blocking_requirements"]
    assert "closeout:success_evidence_expectations" in summary["blocking_requirements"]


def test_missing_final_input_packet_validation_blocks_even_when_final_gates_pass(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    (tmp_path / "tables/final_external_handoff/final_input_packet_validation/summary.json").unlink()

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert summary["blocking_requirements"] == ["handoff:final_input_packet_valid"]


def test_missing_final_manifest_coverage_blocks_even_when_final_gates_pass(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    (tmp_path / "tables/final_external_handoff/final_manifest_coverage/summary.json").unlink()

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert summary["blocking_requirements"] == ["handoff:final_manifest_coverage_ready"]


def test_closeout_success_evidence_blocks_when_expectation_is_not_met(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    write_json(
        tmp_path / "tables/final_external_handoff/final_100_closeout_package/manifest.json",
        {
            "schema": EXPECTED_CLOSEOUT_PACKAGE_SCHEMA,
            "required_external_inputs": REQUIRED_CLOSEOUT_INPUTS,
            "commands": FINALIZATION_COMMANDS,
            "success_evidence": [
                "tables/release_readiness/summary.json::release_gate_passed=false"
            ]
        },
    )

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert "closeout:success_evidence_expectations" in summary["blocking_requirements"]


def test_closeout_success_evidence_can_use_current_in_memory_validation_rows(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    write_minimal_closeout_package(
        tmp_path,
        [
            "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv::handoff:input_packet_matches_external_handoff=pass",
            "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv::closeout:package_manifest_valid=pass",
            "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv::closeout:package_artifacts_match_manifest=pass",
            "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv::closeout:success_evidence_expectations=pass",
            "tables/final_external_handoff/final_100_closeout_validation/summary.json::final_100_ready=true",
        ],
    )

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is True


def test_closeout_package_manifest_blocks_when_required_command_is_missing(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    write_json(
        tmp_path / "tables/final_external_handoff/final_100_closeout_package/manifest.json",
        {
            "schema": EXPECTED_CLOSEOUT_PACKAGE_SCHEMA,
            "required_external_inputs": REQUIRED_CLOSEOUT_INPUTS,
            "commands": [command for command in FINALIZATION_COMMANDS if command != REQUIRED_CLOSEOUT_COMMANDS[-1]],
            "success_evidence": [
                "tables/final_external_handoff/final_input_packet/summary.json::packet_ready=true"
            ],
        },
    )

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert "closeout:package_manifest_valid" in summary["blocking_requirements"]


def test_closeout_package_artifacts_block_when_readme_is_stale(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    (tmp_path / "tables/final_external_handoff/final_100_closeout_package/README.md").write_text(
        "# stale\n",
        encoding="utf-8",
    )

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert "closeout:package_artifacts_match_manifest" in summary["blocking_requirements"]


def test_required_external_input_files_block_final_closeout(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    (tmp_path / "release_metadata.json").unlink()

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert "closeout:required_external_inputs_present" in summary["blocking_requirements"]


def test_closeout_package_artifacts_block_when_script_is_stale(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    (tmp_path / "tables/final_external_handoff/final_100_closeout_package/finalize_100_percent.ps1").write_text(
        "Write-Host 'stale'\n",
        encoding="utf-8",
    )

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert "closeout:package_artifacts_match_manifest" in summary["blocking_requirements"]


def test_closeout_package_manifest_blocks_when_command_order_is_wrong(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    commands = list(FINALIZATION_COMMANDS)
    first = commands.index("python tools/export_final_external_handoff.py")
    second = commands.index("python tools/export_final_input_packet.py")
    commands[first], commands[second] = commands[second], commands[first]
    write_json(
        tmp_path / "tables/final_external_handoff/final_100_closeout_package/manifest.json",
        {
            "schema": EXPECTED_CLOSEOUT_PACKAGE_SCHEMA,
            "required_external_inputs": REQUIRED_CLOSEOUT_INPUTS,
            "commands": commands,
            "success_evidence": [
                "tables/final_external_handoff/final_input_packet/summary.json::packet_ready=true"
            ],
        },
    )

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert "closeout:package_manifest_valid" in summary["blocking_requirements"]


def test_closeout_package_manifest_blocks_malformed_success_evidence(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    write_json(
        tmp_path / "tables/final_external_handoff/final_100_closeout_package/manifest.json",
        {
            "schema": EXPECTED_CLOSEOUT_PACKAGE_SCHEMA,
            "required_external_inputs": REQUIRED_CLOSEOUT_INPUTS,
            "commands": FINALIZATION_COMMANDS,
            "success_evidence": [
                "tables/final_external_handoff/final_input_packet/summary.json::packet_ready=true",
                "tables/final_external_handoff/final_input_packet/README.md::ready=true",
                "../outside.json::ready=true",
                "missing-separators",
            ],
        },
    )

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert "closeout:package_manifest_valid" in summary["blocking_requirements"]


def test_closeout_package_manifest_blocks_duplicate_success_evidence(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    evidence = "tables/final_external_handoff/final_input_packet/summary.json::packet_ready=true"
    write_json(
        tmp_path / "tables/final_external_handoff/final_100_closeout_package/manifest.json",
        {
            "schema": EXPECTED_CLOSEOUT_PACKAGE_SCHEMA,
            "required_external_inputs": REQUIRED_CLOSEOUT_INPUTS,
            "commands": FINALIZATION_COMMANDS,
            "success_evidence": [evidence, evidence],
        },
    )

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert "closeout:package_manifest_valid" in summary["blocking_requirements"]


def test_stale_final_input_packet_blocks_even_when_other_gates_pass(tmp_path: Path) -> None:
    make_summaries(tmp_path, ready=True)
    write_json(
        tmp_path / "tables/final_external_handoff/final_input_packet/summary.json",
        {
            "packet_ready": True,
            "external_handoff_rows": 1,
            "unresolved_requirements": ["external_link:github_release_url"],
        },
    )

    summary = write_outputs(tmp_path, tmp_path / "out")

    assert summary["final_100_ready"] is False
    assert "handoff:input_packet_matches_external_handoff" in summary["blocking_requirements"]


def test_cli_can_fail_on_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_summaries(tmp_path, ready=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["validate_final_100_closeout.py", "--fail-on-incomplete"],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
