from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.test_export_final_input_packet import make_packet_repo
from tools.export_final_input_packet import write_outputs as write_packet_outputs
from tools.validate_final_input_packet import build_rows, main, write_outputs


def corrupt_csv_value(path: Path, *, requirement: str, column: str, value: str) -> None:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    for row in rows:
        if row.get("requirement") == requirement:
            row[column] = value
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_packet_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_packet(tmp_path: Path) -> Path:
    make_packet_repo(tmp_path)
    packet_dir = tmp_path / "tables/final_external_handoff/final_input_packet"
    write_packet_outputs(tmp_path, packet_dir)
    return packet_dir


def test_valid_packet_passes_all_checks(tmp_path: Path) -> None:
    packet_dir = make_packet(tmp_path)

    summary = write_outputs(packet_dir, tmp_path / "validation")

    assert summary["schema"] == "lite-seer-ad-final-input-packet-validation-v1"
    assert summary["final_input_packet_valid"] is True
    assert summary["blocking_requirements"] == []
    assert summary["counts"]["pass"] == 18
    assert (tmp_path / "validation/summary.json").is_file()
    assert (tmp_path / "validation/table_final_input_packet_validation.csv").is_file()
    assert (tmp_path / "validation/final_input_packet_validation.md").is_file()


def test_missing_coverage_table_fails(tmp_path: Path) -> None:
    packet_dir = make_packet(tmp_path)
    (packet_dir / "table_handoff_input_coverage.csv").unlink()

    rows = build_rows(packet_dir)
    by_requirement = {row["requirement"]: row for row in rows}

    assert by_requirement["coverage:csv_present"]["status"] == "fail"
    assert by_requirement["coverage:all_unresolved_requirements_covered"]["status"] == "fail"


def test_coverage_false_fails_packet_validation(tmp_path: Path) -> None:
    packet_dir = make_packet(tmp_path)
    corrupt_csv_value(
        packet_dir / "table_handoff_input_coverage.csv",
        requirement="production:cross_hardware",
        column="covered",
        value="false",
    )

    summary = write_outputs(packet_dir, tmp_path / "validation")

    assert summary["final_input_packet_valid"] is False
    assert "coverage:all_unresolved_requirements_covered" in summary["blocking_requirements"]


def test_coverage_command_mismatch_fails_packet_validation(tmp_path: Path) -> None:
    packet_dir = make_packet(tmp_path)
    corrupt_csv_value(
        packet_dir / "table_handoff_input_coverage.csv",
        requirement="production:cross_hardware",
        column="completion_command",
        value="python stale_command.py",
    )

    summary = write_outputs(packet_dir, tmp_path / "validation")

    assert summary["final_input_packet_valid"] is False
    assert "coverage:matches_external_handoff" in summary["blocking_requirements"]


def test_missing_second_hardware_return_files_fails(tmp_path: Path) -> None:
    packet_dir = make_packet(tmp_path)
    second_hardware_path = packet_dir / "second_hardware_inputs.json"
    payload = json.loads(second_hardware_path.read_text(encoding="utf-8"))
    payload.pop("expected_return_files")
    write_packet_json(second_hardware_path, payload)

    summary = write_outputs(packet_dir, tmp_path / "validation")

    assert summary["final_input_packet_valid"] is False
    assert "second_hardware:expected_return_files" in summary["blocking_requirements"]
    assert "fields:second_hardware_hints_reference_return_files" in summary["blocking_requirements"]


def test_second_hardware_hint_mismatch_fails(tmp_path: Path) -> None:
    packet_dir = make_packet(tmp_path)
    corrupt_csv_value(
        packet_dir / "table_final_input_fields.csv",
        requirement="external_input:second_hardware_profile.json",
        column="suggested_value_or_hint",
        value="Use the generated profile.",
    )

    summary = write_outputs(packet_dir, tmp_path / "validation")

    assert summary["final_input_packet_valid"] is False
    assert "fields:second_hardware_hints_reference_return_files" in summary["blocking_requirements"]


def test_invalid_metadata_draft_fails_preview_validation(tmp_path: Path) -> None:
    packet_dir = make_packet(tmp_path)
    release_draft = packet_dir / "release_metadata.draft.json"
    payload = json.loads(release_draft.read_text(encoding="utf-8"))
    payload["schema"] = "wrong-schema"
    write_packet_json(release_draft, payload)

    summary = write_outputs(packet_dir, tmp_path / "validation")

    assert summary["final_input_packet_valid"] is False
    assert "metadata:drafts_preview_valid" in summary["blocking_requirements"]


def test_cli_can_fail_on_invalid_packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet_dir = make_packet(tmp_path)
    (packet_dir / "table_final_input_fields.csv").unlink()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_final_input_packet.py",
            "--packet-dir",
            str(packet_dir),
            "--fail-on-invalid",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
