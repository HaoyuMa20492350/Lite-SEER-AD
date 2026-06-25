"""Validate the final plan-defined 100% readiness gates.

This validator is intentionally strict: it checks the refreshed readiness
summaries that define the end state, writes an audit table, and can fail the
process when any required gate is still incomplete.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCHEMA = "lite-seer-ad-final-100-closeout-validation-v1"
DEFAULT_OUT_DIR = Path("tables/final_external_handoff/final_100_closeout_validation")
CLOSEOUT_PACKAGE_MANIFEST = Path("tables/final_external_handoff/final_100_closeout_package/manifest.json")
CLOSEOUT_PACKAGE_README = Path("tables/final_external_handoff/final_100_closeout_package/README.md")
CLOSEOUT_PACKAGE_SCRIPT = Path("tables/final_external_handoff/final_100_closeout_package/finalize_100_percent.ps1")
CLOSEOUT_PACKAGE_AUDIT_TEMPLATE = Path(
    "tables/final_external_handoff/final_100_closeout_package/completion_audit_template.md"
)
EXPECTED_CLOSEOUT_PACKAGE_SCHEMA = "lite-seer-ad-final-100-closeout-package-v1"
REQUIRED_CLOSEOUT_INPUTS = [
    "second_hardware_profile.json",
    "second_hardware_energy.json",
    "release_metadata.json",
    "submission_metadata.json",
]
REQUIRED_CLOSEOUT_COMMANDS = [
    "python tools/validate_second_hardware_package.py --hardware-profile $SecondHardwareProfile --energy-measurement $SecondHardwareEnergy --stage",
    "python tools/validate_release_submission_consistency.py --fail-on-inconsistent",
    "python tools/render_release_metadata.py --input release_metadata.json",
    "python tools/render_submission_statements.py --input submission_metadata.json --out docs/submission_statement_placeholders.md",
    "python tools/validate_final_input_packet.py --fail-on-invalid",
    "python tools/validate_final_100_closeout.py --fail-on-incomplete",
    "python tools/validate_final_manifest_coverage.py --fail-on-missing",
]
REQUIRED_CLOSEOUT_COMMAND_SUBSEQUENCES = [
    [
        "python tools/validate_second_hardware_package.py --hardware-profile $SecondHardwareProfile --energy-measurement $SecondHardwareEnergy --stage",
        "python tools/validate_release_submission_consistency.py --fail-on-inconsistent",
        "python tools/render_release_metadata.py --input release_metadata.json",
        "python tools/render_submission_statements.py --input submission_metadata.json --out docs/submission_statement_placeholders.md",
    ],
    [
        "python tools/export_final_external_handoff.py",
        "python tools/export_final_input_packet.py",
        "python tools/validate_final_input_packet.py --fail-on-invalid",
        "python tools/validate_final_100_closeout.py",
        "python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json",
        "python tools/validate_final_manifest_coverage.py --fail-on-missing",
        "python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json",
        "python tools/validate_final_100_closeout.py --fail-on-incomplete",
        "python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json",
        "python tools/validate_final_manifest_coverage.py --fail-on-missing",
        "python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json",
    ],
]
SELF_SUMMARY_EVIDENCE = "tables/final_external_handoff/final_100_closeout_validation/summary.json"
SELF_TABLE_EVIDENCE = "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv"
CHECKS = [
    {
        "requirement": "handoff:final_input_packet_ready",
        "evidence": "tables/final_external_handoff/final_input_packet/summary.json",
        "key": "packet_ready",
        "expected": True,
    },
    {
        "requirement": "handoff:final_input_packet_valid",
        "evidence": "tables/final_external_handoff/final_input_packet_validation/summary.json",
        "key": "final_input_packet_valid",
        "expected": True,
    },
    {
        "requirement": "handoff:final_manifest_coverage_ready",
        "evidence": "tables/final_external_handoff/final_manifest_coverage/summary.json",
        "key": "final_manifest_coverage_ready",
        "expected": True,
    },
    {
        "requirement": "metadata:final_metadata_consistent",
        "evidence": "tables/final_external_handoff/final_metadata_consistency/summary.json",
        "key": "final_metadata_consistent",
        "expected": True,
    },
    {
        "requirement": "deployment:production_deployment_ready",
        "evidence": "tables/deployment_readiness/summary.json",
        "key": "production_deployment_ready",
        "expected": True,
    },
    {
        "requirement": "public_release:release_gate_passed",
        "evidence": "tables/release_readiness/summary.json",
        "key": "release_gate_passed",
        "expected": True,
    },
    {
        "requirement": "submission:final_upload_ready",
        "evidence": "tables/submission_package_readiness/summary.json",
        "key": "final_upload_ready",
        "expected": True,
    },
    {
        "requirement": "completion:default_100_ready",
        "evidence": "tables/completion_gap_matrix/summary.json",
        "key": "default_100_ready",
        "expected": True,
    },
    {
        "requirement": "handoff:all_external_actions_complete",
        "evidence": "tables/final_external_handoff/summary.json",
        "key": "all_external_actions_complete",
        "expected": True,
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def parse_expected_value(value: str) -> Any:
    text = value.strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    return text


def json_value_matches(path: Path, key: str, expected: Any) -> tuple[bool, Any]:
    payload = read_json(path)
    actual = payload.get(key)
    return actual == expected, actual


def csv_requirement_matches(path: Path, requirement: str, expected: Any) -> tuple[bool, Any]:
    if not path.exists():
        return False, None
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("requirement") == requirement:
                actual = row.get("status")
                return actual == expected, actual
    return False, None


def in_memory_requirement_matches(rows: list[dict[str, Any]], requirement: str, expected: Any) -> tuple[bool, Any]:
    for row in rows:
        if row.get("requirement") == requirement:
            actual = row.get("status")
            return actual == expected, actual
    return False, None


def evaluate_success_evidence(root: Path, evidence: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if "::" not in evidence:
        return {
            "evidence": evidence,
            "status": "fail",
            "detail": "missing :: expectation separator",
        }
    source, expectation = evidence.split("::", 1)
    if "=" not in expectation:
        return {
            "evidence": evidence,
            "status": "fail",
            "detail": "missing = expected value separator",
        }
    key, expected_text = expectation.split("=", 1)
    expected = parse_expected_value(expected_text)
    source = source.strip().replace("\\", "/")
    key = key.strip()
    if source == SELF_SUMMARY_EVIDENCE and key == "final_100_ready":
        return {
            "evidence": evidence,
            "status": "deferred",
            "detail": "self-referential final_100_ready is determined by this validator summary",
        }
    if source == SELF_TABLE_EVIDENCE and key == "closeout:success_evidence_expectations":
        return {
            "evidence": evidence,
            "status": "deferred",
            "detail": "self-referential success_evidence row is determined by this validator",
        }
    if source == SELF_TABLE_EVIDENCE:
        ok, actual = in_memory_requirement_matches(rows, key, expected)
    elif source.endswith(".csv"):
        ok, actual = csv_requirement_matches(root / source, key, expected)
    elif source.endswith(".json"):
        ok, actual = json_value_matches(root / source, key, expected)
    else:
        return {
            "evidence": evidence,
            "status": "fail",
            "detail": "unsupported success evidence source type",
        }
    return {
        "evidence": evidence,
        "status": "pass" if ok else "fail",
        "detail": f"expected {expected!r}, actual {actual!r}",
    }


def check_closeout_success_evidence(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = read_json(root / CLOSEOUT_PACKAGE_MANIFEST)
    evidence_items = [str(item) for item in manifest.get("success_evidence", []) or []]
    evaluations = [evaluate_success_evidence(root, item, rows) for item in evidence_items]
    failed = [item for item in evaluations if item["status"] == "fail"]
    passed = [item for item in evaluations if item["status"] == "pass"]
    deferred = [item for item in evaluations if item["status"] == "deferred"]
    ok = bool(evidence_items) and not failed
    actual = {
        "passed": len(passed),
        "failed": len(failed),
        "deferred": len(deferred),
        "failed_evidence": [item["evidence"] for item in failed],
        "deferred_evidence": [item["evidence"] for item in deferred],
    }
    return {
        "requirement": "closeout:success_evidence_expectations",
        "status": "pass" if ok else "fail",
        "evidence": str(CLOSEOUT_PACKAGE_MANIFEST),
        "key": "success_evidence",
        "expected": "all non-self evidence expectations pass",
        "actual": json.dumps(actual, ensure_ascii=False, sort_keys=True),
    }


def check_closeout_package_manifest(root: Path) -> dict[str, Any]:
    manifest = read_json(root / CLOSEOUT_PACKAGE_MANIFEST)
    inputs = [str(item) for item in manifest.get("required_external_inputs", []) or []]
    commands = [str(item) for item in manifest.get("commands", []) or []]
    success_evidence = [str(item) for item in manifest.get("success_evidence", []) or []]
    errors: list[str] = []
    if manifest.get("schema") != EXPECTED_CLOSEOUT_PACKAGE_SCHEMA:
        errors.append("schema mismatch")
    missing_inputs = [item for item in REQUIRED_CLOSEOUT_INPUTS if item not in inputs]
    if missing_inputs:
        errors.append(f"missing required inputs: {missing_inputs}")
    missing_commands = [item for item in REQUIRED_CLOSEOUT_COMMANDS if item not in commands]
    if missing_commands:
        errors.append(f"missing required commands: {missing_commands}")
    missing_sequences = [
        sequence
        for sequence in REQUIRED_CLOSEOUT_COMMAND_SUBSEQUENCES
        if not command_sequence_present(commands, sequence)
    ]
    if missing_sequences:
        errors.append(f"required command order is missing: {missing_sequences}")
    if not success_evidence:
        errors.append("success_evidence is empty")
    evidence_errors = validate_success_evidence_manifest_entries(success_evidence)
    errors.extend(evidence_errors)
    actual = {
        "required_external_inputs": inputs,
        "commands": len(commands),
        "success_evidence": len(success_evidence),
        "errors": errors,
    }
    return {
        "requirement": "closeout:package_manifest_valid",
        "status": "pass" if not errors else "fail",
        "evidence": str(CLOSEOUT_PACKAGE_MANIFEST),
        "key": "schema/required_external_inputs/commands/success_evidence",
        "expected": "valid final closeout package manifest",
        "actual": json.dumps(actual, ensure_ascii=False, sort_keys=True),
    }


def check_closeout_package_artifacts(root: Path) -> dict[str, Any]:
    manifest = read_json(root / CLOSEOUT_PACKAGE_MANIFEST)
    inputs = [str(item) for item in manifest.get("required_external_inputs", []) or []]
    commands = [str(item) for item in manifest.get("commands", []) or []]
    success_evidence = [str(item) for item in manifest.get("success_evidence", []) or []]
    readme = read_text(root / CLOSEOUT_PACKAGE_README)
    script = read_text(root / CLOSEOUT_PACKAGE_SCRIPT)
    audit = read_text(root / CLOSEOUT_PACKAGE_AUDIT_TEMPLATE)
    errors: list[str] = []
    if not readme:
        errors.append("README.md is missing or empty")
    if not script:
        errors.append("finalize_100_percent.ps1 is missing or empty")
    if not audit:
        errors.append("completion_audit_template.md is missing or empty")
    missing_readme_inputs = [item for item in inputs if item not in readme]
    if missing_readme_inputs:
        errors.append(f"README missing required inputs: {missing_readme_inputs}")
    missing_readme_evidence = [item for item in success_evidence if item not in readme]
    if missing_readme_evidence:
        errors.append(f"README missing success evidence: {missing_readme_evidence}")
    missing_script_commands = [item for item in commands if item not in script]
    if missing_script_commands:
        errors.append(f"finalize script missing commands: {missing_script_commands}")
    missing_audit_evidence = []
    for evidence in success_evidence:
        if "::" not in evidence:
            missing_audit_evidence.append(evidence)
            continue
        source, expectation = evidence.split("::", 1)
        if source not in audit or expectation not in audit:
            missing_audit_evidence.append(evidence)
    if missing_audit_evidence:
        errors.append(f"audit template missing success evidence rows: {missing_audit_evidence}")
    actual = {
        "errors": errors,
        "required_external_inputs": len(inputs),
        "commands": len(commands),
        "success_evidence": len(success_evidence),
    }
    return {
        "requirement": "closeout:package_artifacts_match_manifest",
        "status": "pass" if not errors else "fail",
        "evidence": (
            f"{CLOSEOUT_PACKAGE_MANIFEST}; {CLOSEOUT_PACKAGE_README}; "
            f"{CLOSEOUT_PACKAGE_SCRIPT}; {CLOSEOUT_PACKAGE_AUDIT_TEMPLATE}"
        ),
        "key": "manifest/rendered_artifacts",
        "expected": "README, finalization script, and audit template match manifest",
        "actual": json.dumps(actual, ensure_ascii=False, sort_keys=True),
    }


def check_required_external_inputs_present(root: Path) -> dict[str, Any]:
    manifest = read_json(root / CLOSEOUT_PACKAGE_MANIFEST)
    inputs = [str(item) for item in manifest.get("required_external_inputs", []) or []]
    errors: list[str] = []
    present: list[str] = []
    for item in inputs:
        path = root / item
        if not path.is_file():
            errors.append(f"missing: {item}")
            continue
        if path.stat().st_size <= 0:
            errors.append(f"empty: {item}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON: {item}: {exc.msg}")
            continue
        if not isinstance(payload, dict) or not payload:
            errors.append(f"expected non-empty JSON object: {item}")
            continue
        present.append(item)
    missing_declared_inputs = [item for item in REQUIRED_CLOSEOUT_INPUTS if item not in inputs]
    if missing_declared_inputs:
        errors.append(f"manifest does not declare required inputs: {missing_declared_inputs}")
    actual = {
        "declared": inputs,
        "present_valid_json_objects": present,
        "errors": errors,
    }
    return {
        "requirement": "closeout:required_external_inputs_present",
        "status": "pass" if not errors and len(present) == len(inputs) and bool(inputs) else "fail",
        "evidence": str(CLOSEOUT_PACKAGE_MANIFEST),
        "key": "required_external_inputs",
        "expected": "all required final external input files exist at repo root as non-empty JSON objects",
        "actual": json.dumps(actual, ensure_ascii=False, sort_keys=True),
    }


def validate_success_evidence_manifest_entries(evidence_items: list[str]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, evidence in enumerate(evidence_items):
        if evidence in seen:
            errors.append(f"duplicate success_evidence[{index}]: {evidence}")
            continue
        seen.add(evidence)
        if "::" not in evidence:
            errors.append(f"success_evidence[{index}] missing :: separator: {evidence}")
            continue
        source, expectation = evidence.split("::", 1)
        source = source.strip().replace("\\", "/")
        if not source or source.startswith("/") or ".." in Path(source).parts:
            errors.append(f"success_evidence[{index}] must use a repository-relative source path: {source}")
        if not (source.endswith(".json") or source.endswith(".csv")):
            errors.append(f"success_evidence[{index}] source must be .json or .csv: {source}")
        if "=" not in expectation:
            errors.append(f"success_evidence[{index}] missing = expected value: {evidence}")
            continue
        key, expected = expectation.split("=", 1)
        if not key.strip():
            errors.append(f"success_evidence[{index}] has empty key: {evidence}")
        if not expected.strip():
            errors.append(f"success_evidence[{index}] has empty expected value: {evidence}")
    return errors


def command_sequence_present(commands: list[str], sequence: list[str]) -> bool:
    cursor = 0
    for command in commands:
        if cursor < len(sequence) and command == sequence[cursor]:
            cursor += 1
    return cursor == len(sequence)


def check_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for check in CHECKS:
        path = root / str(check["evidence"])
        payload = read_json(path)
        actual = payload.get(str(check["key"]))
        expected = check["expected"]
        rows.append(
            {
                "requirement": check["requirement"],
                "status": "pass" if actual is expected else "fail",
                "evidence": check["evidence"],
                "key": check["key"],
                "expected": json.dumps(expected),
                "actual": json.dumps(actual),
            }
        )
    rows.append(check_closeout_package_manifest(root))
    rows.append(check_closeout_package_artifacts(root))
    rows.append(check_required_external_inputs_present(root))
    rows.append(check_input_packet_matches_handoff(root))
    rows.append(check_closeout_success_evidence(root, rows))
    return rows


def check_input_packet_matches_handoff(root: Path) -> dict[str, Any]:
    packet_path = root / "tables/final_external_handoff/final_input_packet/summary.json"
    handoff_path = root / "tables/final_external_handoff/summary.json"
    packet = read_json(packet_path)
    handoff = read_json(handoff_path)
    packet_unresolved = set(str(item) for item in packet.get("unresolved_requirements", []) or [])
    handoff_unresolved = set(str(item) for item in handoff.get("unresolved_requirements", []) or [])
    rows_match = packet.get("external_handoff_rows") == handoff.get("rows")
    unresolved_match = packet_unresolved == handoff_unresolved
    ok = rows_match and unresolved_match
    detail = {
        "packet_rows": packet.get("external_handoff_rows"),
        "handoff_rows": handoff.get("rows"),
        "packet_unresolved": sorted(packet_unresolved),
        "handoff_unresolved": sorted(handoff_unresolved),
    }
    return {
        "requirement": "handoff:input_packet_matches_external_handoff",
        "status": "pass" if ok else "fail",
        "evidence": "tables/final_external_handoff/final_input_packet/summary.json; tables/final_external_handoff/summary.json",
        "key": "external_handoff_rows/unresolved_requirements",
        "expected": "match",
        "actual": json.dumps(detail, ensure_ascii=False, sort_keys=True),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    blockers = [row["requirement"] for row in rows if row["status"] != "pass"]
    return {
        "schema": SCHEMA,
        "final_100_ready": not blockers,
        "counts": counts,
        "blocking_requirements": blockers,
        "checked_requirements": [row["requirement"] for row in rows],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["requirement", "status", "evidence", "key", "expected", "actual"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Final 100% Closeout Validation",
        "",
        f"- Final 100% ready: `{summary['final_100_ready']}`",
        "",
        "| Requirement | Status | Evidence | Expected | Actual |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['requirement']}` | `{row['status']}` | `{row['evidence']}::{row['key']}` | "
            f"`{row['expected']}` | `{row['actual']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(root: Path, out_dir: Path) -> dict[str, Any]:
    rows = check_rows(root)
    summary = build_summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_final_100_closeout_validation.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "final_100_closeout_validation.md", summary, rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Exit non-zero when any final 100% gate is incomplete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_outputs(args.root, args.out_dir)
    print(
        f"Wrote final 100% validation to {args.out_dir} "
        f"(final_100_ready={summary['final_100_ready']})"
    )
    if args.fail_on_incomplete and not summary["final_100_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
