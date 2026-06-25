"""Validate the final external-input packet.

The final input packet is generated before real release/submission/second-
hardware inputs are available. This validator checks the packet itself is
complete and internally consistent, so the final 100% validator does not have
to trust a stale summary blindly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.render_release_metadata import validate_release_metadata
from tools.render_submission_statements import validate_metadata as validate_submission_metadata
from tools.export_final_external_handoff import build_rows as build_handoff_rows


SCHEMA = "lite-seer-ad-final-input-packet-validation-v1"
PACKET_SCHEMA = "lite-seer-ad-final-input-packet-v1"
DEFAULT_PACKET_DIR = Path("tables/final_external_handoff/final_input_packet")
DEFAULT_OUT_DIR = Path("tables/final_external_handoff/final_input_packet_validation")
FIELD_COLUMNS = [
    "source",
    "requirement",
    "destination_file",
    "json_path",
    "prompt",
    "suggested_value_or_hint",
    "validator",
]
COVERAGE_COLUMNS = ["requirement", "status", "covered", "input_files", "field_hints", "completion_command"]
SECOND_HARDWARE_SCHEMA = "lite-seer-ad-second-hardware-inputs-v1"
SOURCE_DRAFTS = {
    "release": "release_metadata.draft.json",
    "submission": "submission_metadata.draft.json",
}
SOURCE_DRAFT_VALIDATORS = {
    "release": validate_release_metadata,
    "submission": validate_submission_metadata,
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row(requirement: str, status: str, evidence: str, detail: str) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "detail": detail,
    }


def has_columns(rows: list[dict[str, str]], columns: list[str]) -> bool:
    if not rows:
        return False
    return all(column in rows[0] for column in columns)


def inferred_root_from_packet_dir(packet_dir: Path) -> Path:
    parts = packet_dir.parts
    if len(parts) >= 3 and parts[-3:] == ("tables", "final_external_handoff", "final_input_packet"):
        return Path(*parts[:-3]) if parts[:-3] else Path(".")
    return Path.cwd()


def read_external_handoff_rows(packet_dir: Path) -> tuple[list[dict[str, str]], str]:
    handoff_path = packet_dir.parent / "table_final_external_handoff.csv"
    handoff_rows = read_csv(handoff_path)
    if handoff_rows:
        return handoff_rows, str(handoff_path)
    root = inferred_root_from_packet_dir(packet_dir)
    rebuilt_rows = build_handoff_rows(root)
    return [{key: str(value) for key, value in item.items()} for item in rebuilt_rows], f"rebuilt from {root}"


def handoff_coverage_mismatches(
    coverage_rows: list[dict[str, str]],
    handoff_rows: list[dict[str, str]],
) -> list[str]:
    unresolved = [item for item in handoff_rows if item.get("status") != "pass"]
    coverage_requirements = [item.get("requirement", "") for item in coverage_rows]
    coverage_by_requirement = {item.get("requirement", ""): item for item in coverage_rows}
    handoff_by_requirement = {item.get("requirement", ""): item for item in unresolved}
    mismatches: list[str] = []

    duplicates = sorted({item for item in coverage_requirements if coverage_requirements.count(item) > 1})
    if duplicates:
        mismatches.append("duplicate coverage requirements: " + ",".join(duplicates))

    missing = sorted(set(handoff_by_requirement) - set(coverage_by_requirement))
    if missing:
        mismatches.append("missing from coverage: " + ",".join(missing))

    extra = sorted(set(coverage_by_requirement) - set(handoff_by_requirement))
    if extra:
        mismatches.append("extra in coverage: " + ",".join(extra))

    for requirement, handoff_row in sorted(handoff_by_requirement.items()):
        coverage_row = coverage_by_requirement.get(requirement)
        if coverage_row is None:
            continue
        if coverage_row.get("status", "").strip() != handoff_row.get("status", "").strip():
            mismatches.append(f"status mismatch: {requirement}")
        if coverage_row.get("completion_command", "").strip() != handoff_row.get("completion_command", "").strip():
            mismatches.append(f"completion_command mismatch: {requirement}")
    return mismatches


def valid_second_hardware_return_files(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    files = [str(item).strip() for item in value if str(item).strip()]
    has_energy = any(item.endswith("_energy.json") for item in files)
    has_profile = any(item.endswith("_hardware_profile.json") for item in files)
    if len(files) != 2 or not has_energy or not has_profile:
        return []
    return files


def second_hardware_hints_reference_return_files(
    field_rows: list[dict[str, str]],
    return_files: list[str],
) -> bool:
    second_rows = [item for item in field_rows if item.get("source") == "second_hardware"]
    if len(second_rows) != 2 or len(return_files) != 2:
        return False
    hints = "\n".join(item.get("suggested_value_or_hint", "") for item in second_rows)
    return all(path in hints for path in return_files)


def draft_preview_errors(packet_dir: Path, metadata_sources: list[Any]) -> list[str]:
    errors: list[str] = []
    for source in metadata_sources:
        if not isinstance(source, dict) or not source.get("template_exists"):
            continue
        name = str(source.get("source", ""))
        draft = SOURCE_DRAFTS.get(name)
        validator = SOURCE_DRAFT_VALIDATORS.get(name)
        if not draft or validator is None:
            continue
        draft_path = packet_dir / draft
        payload = read_json(draft_path)
        if not payload:
            errors.append(f"{draft}: missing or invalid JSON")
            continue
        draft_errors = validator(payload, allow_placeholders=True)
        if draft_errors:
            errors.append(f"{draft}: " + "; ".join(draft_errors[:5]))
    return errors


def build_rows(packet_dir: Path) -> list[dict[str, Any]]:
    summary_path = packet_dir / "summary.json"
    fields_path = packet_dir / "table_final_input_fields.csv"
    coverage_path = packet_dir / "table_handoff_input_coverage.csv"
    second_hardware_path = packet_dir / "second_hardware_inputs.json"
    readme_path = packet_dir / "README.md"

    summary = read_json(summary_path)
    field_rows = read_csv(fields_path)
    coverage_rows = read_csv(coverage_path)
    handoff_rows, handoff_evidence = read_external_handoff_rows(packet_dir)
    second_hardware = read_json(second_hardware_path)
    rows: list[dict[str, Any]] = []

    rows.append(
        row(
            "summary:schema",
            "pass" if summary.get("schema") == PACKET_SCHEMA else "fail",
            str(summary_path),
            f"schema={summary.get('schema')}",
        )
    )
    rows.append(
        row(
            "summary:packet_ready",
            "pass" if summary.get("packet_ready") is True and not summary.get("packet_blocking_requirements") else "fail",
            str(summary_path),
            f"packet_ready={summary.get('packet_ready')} blockers={len(summary.get('packet_blocking_requirements') or [])}",
        )
    )
    rows.append(
        row(
            "fields:csv_present",
            "pass" if has_columns(field_rows, FIELD_COLUMNS) else "fail",
            str(fields_path),
            f"rows={len(field_rows)}",
        )
    )
    rows.append(
        row(
            "fields:row_count_matches_summary",
            "pass" if len(field_rows) == int(summary.get("field_rows", -1) or -1) else "fail",
            str(fields_path),
            f"csv_rows={len(field_rows)} summary_rows={summary.get('field_rows')}",
        )
    )
    rows.append(
        row(
            "fields:second_hardware_rows",
            "pass" if sum(1 for item in field_rows if item.get("source") == "second_hardware") == 2 else "fail",
            str(fields_path),
            f"second_hardware_rows={sum(1 for item in field_rows if item.get('source') == 'second_hardware')}",
        )
    )
    rows.append(
        row(
            "fields:all_rows_actionable",
            "pass"
            if field_rows and all(item.get("destination_file") and item.get("prompt") and item.get("validator") for item in field_rows)
            else "fail",
            str(fields_path),
            "every row has destination_file, prompt, and validator" if field_rows else "no field rows",
        )
    )
    rows.append(
        row(
            "coverage:csv_present",
            "pass" if has_columns(coverage_rows, COVERAGE_COLUMNS) else "fail",
            str(coverage_path),
            f"rows={len(coverage_rows)}",
        )
    )
    rows.append(
        row(
            "coverage:row_count_matches_summary",
            "pass"
            if len(coverage_rows) == int(summary.get("handoff_input_coverage_rows", -1) or -1)
            else "fail",
            str(coverage_path),
            f"csv_rows={len(coverage_rows)} summary_rows={summary.get('handoff_input_coverage_rows')}",
        )
    )
    rows.append(
        row(
            "coverage:all_unresolved_requirements_covered",
            "pass"
            if coverage_rows and all(item.get("covered") == "true" for item in coverage_rows)
            else "fail",
            str(coverage_path),
            "all covered=true" if coverage_rows else "no coverage rows",
        )
    )
    rows.append(
        row(
            "coverage:all_rows_have_inputs",
            "pass"
            if coverage_rows
            and all(item.get("input_files") and item.get("field_hints") and item.get("completion_command") for item in coverage_rows)
            else "fail",
            str(coverage_path),
            "every coverage row has input_files, field_hints, completion_command" if coverage_rows else "no coverage rows",
        )
    )
    handoff_mismatches = handoff_coverage_mismatches(coverage_rows, handoff_rows)
    rows.append(
        row(
            "coverage:matches_external_handoff",
            "pass" if coverage_rows and handoff_rows and not handoff_mismatches else "fail",
            handoff_evidence,
            (
                f"coverage rows match unresolved handoff rows and completion commands; rows={len(coverage_rows)}"
                if coverage_rows and handoff_rows and not handoff_mismatches
                else " | ".join(handoff_mismatches or ["missing coverage or handoff rows"])
            ),
        )
    )
    required_inputs = second_hardware.get("required_inputs") or []
    second_hardware_return_files = valid_second_hardware_return_files(
        second_hardware.get("expected_return_files")
    )
    rows.append(
        row(
            "second_hardware:manifest",
            "pass" if second_hardware.get("schema") == SECOND_HARDWARE_SCHEMA and len(required_inputs) == 2 else "fail",
            str(second_hardware_path),
            f"schema={second_hardware.get('schema')} required_inputs={len(required_inputs) if isinstance(required_inputs, list) else 0}",
        )
    )
    rows.append(
        row(
            "second_hardware:expected_return_files",
            "pass" if second_hardware_return_files else "fail",
            str(second_hardware_path),
            ";".join(second_hardware_return_files)
            if second_hardware_return_files
            else "expected one *_energy.json and one *_hardware_profile.json",
        )
    )
    hints_reference_return_files = second_hardware_hints_reference_return_files(
        field_rows,
        second_hardware_return_files,
    )
    rows.append(
        row(
            "fields:second_hardware_hints_reference_return_files",
            "pass" if hints_reference_return_files else "fail",
            str(fields_path),
            "second_hardware field hints reference expected return files"
            if hints_reference_return_files
            else "second_hardware field hints do not reference expected return files",
        )
    )
    metadata_sources = summary.get("metadata_input_sources") or []
    metadata_ok = bool(metadata_sources) and all(
        bool(source.get("template_exists") or source.get("final_exists")) for source in metadata_sources
    )
    rows.append(
        row(
            "metadata:sources_available",
            "pass" if metadata_ok else "fail",
            str(summary_path),
            f"sources={len(metadata_sources) if isinstance(metadata_sources, list) else 0}",
        )
    )
    draft_missing = []
    if isinstance(metadata_sources, list):
        for source in metadata_sources:
            name = str(source.get("source", ""))
            draft = SOURCE_DRAFTS.get(name)
            if source.get("template_exists") and draft and not (packet_dir / draft).is_file():
                draft_missing.append(draft)
    rows.append(
        row(
            "metadata:drafts_present_when_templates_exist",
            "pass" if not draft_missing else "fail",
            str(packet_dir),
            "drafts present" if not draft_missing else "missing " + ";".join(draft_missing),
        )
    )
    preview_errors = draft_preview_errors(packet_dir, metadata_sources if isinstance(metadata_sources, list) else [])
    rows.append(
        row(
            "metadata:drafts_preview_valid",
            "pass" if not preview_errors else "fail",
            str(packet_dir),
            "drafts are valid with placeholders allowed"
            if not preview_errors
            else " | ".join(preview_errors),
        )
    )
    rows.append(
        row(
            "readme:present",
            "pass" if readme_path.is_file() and readme_path.stat().st_size > 0 else "fail",
            str(readme_path),
            "present" if readme_path.is_file() else "missing",
        )
    )
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in rows:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    blockers = [item["requirement"] for item in rows if item["status"] != "pass"]
    return {
        "schema": SCHEMA,
        "final_input_packet_valid": not blockers,
        "counts": counts,
        "blocking_requirements": blockers,
        "checked_requirements": [item["requirement"] for item in rows],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["requirement", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Final Input Packet Validation",
        "",
        f"- Final input packet valid: `{summary['final_input_packet_valid']}`",
        "",
        "| Requirement | Status | Detail |",
        "|---|---|---|",
    ]
    for item in rows:
        lines.append(f"| `{item['requirement']}` | `{item['status']}` | {item['detail']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(packet_dir: Path, out_dir: Path) -> dict[str, Any]:
    rows = build_rows(packet_dir)
    summary = build_summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_final_input_packet_validation.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "final_input_packet_validation.md", summary, rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Exit non-zero if the final input packet is incomplete or internally inconsistent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_outputs(args.packet_dir, args.out_dir)
    print(
        f"Wrote final input packet validation to {args.out_dir} "
        f"(final_input_packet_valid={summary['final_input_packet_valid']})"
    )
    if args.fail_on_invalid and not summary["final_input_packet_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
