"""Validate that the artifact manifest covers final closeout evidence.

The final release manifest should not only contain model/table artifacts; it
must also hash the evidence that proves the last-mile external handoff path.
This validator checks that the generated SHA256 manifest includes those
closeout packages and validation reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "lite-seer-ad-final-manifest-coverage-v1"
DEFAULT_OUT_DIR = Path("tables/final_external_handoff/final_manifest_coverage")
CLOSEOUT_PACKAGE_MANIFEST = Path("tables/final_external_handoff/final_100_closeout_package/manifest.json")
SELF_GENERATED_PREFIX = "tables/final_external_handoff/final_manifest_coverage/"
REQUIRED_MANIFEST_PATHS = [
    "tables/deployment_readiness/summary.json",
    "tables/deployment_readiness/table_deployment_readiness.csv",
    "tables/release_readiness/summary.json",
    "tables/release_readiness/table_release_readiness.csv",
    "tables/submission_package_readiness/summary.json",
    "tables/submission_package_readiness/table_submission_readiness.csv",
    "tables/completion_gap_matrix/summary.json",
    "tables/completion_gap_matrix/table_completion_gap_matrix.csv",
    "tables/final_external_handoff/summary.json",
    "tables/final_external_handoff/table_final_external_handoff.csv",
    "tables/final_external_handoff/final_external_handoff.md",
    "tables/deployment_production_latency/second_hardware_run_package/manifest.json",
    "tables/deployment_production_latency/second_hardware_run_package/README.md",
    "tables/deployment_production_latency/second_hardware_run_package/run_second_hardware_probe.ps1",
    "tables/deployment_production_latency/second_hardware_run_package/validate_returned_second_hardware_package.ps1",
    "tables/submission_package_readiness/final_upload_closeout_package/manifest.json",
    "tables/submission_package_readiness/final_upload_closeout_package/README.md",
    "tables/submission_package_readiness/final_upload_closeout_package/finalize_submission_upload.ps1",
    "tables/final_external_handoff/final_metadata_consistency/summary.json",
    "tables/final_external_handoff/final_metadata_consistency/table_final_metadata_consistency.csv",
    "tables/final_external_handoff/final_metadata_consistency/final_metadata_consistency.md",
    "tables/final_external_handoff/final_input_packet/summary.json",
    "tables/final_external_handoff/final_input_packet/table_final_input_fields.csv",
    "tables/final_external_handoff/final_input_packet/table_handoff_input_coverage.csv",
    "tables/final_external_handoff/final_input_packet/README.md",
    "tables/final_external_handoff/final_input_packet/second_hardware_inputs.json",
    "tables/final_external_handoff/final_input_packet/release_metadata.draft.json",
    "tables/final_external_handoff/final_input_packet/submission_metadata.draft.json",
    "tables/final_external_handoff/final_input_packet_validation/summary.json",
    "tables/final_external_handoff/final_input_packet_validation/table_final_input_packet_validation.csv",
    "tables/final_external_handoff/final_input_packet_validation/final_input_packet_validation.md",
    "tables/final_external_handoff/final_100_closeout_package/manifest.json",
    "tables/final_external_handoff/final_100_closeout_package/README.md",
    "tables/final_external_handoff/final_100_closeout_package/finalize_100_percent.ps1",
    "tables/final_external_handoff/final_100_closeout_package/completion_audit_template.md",
    "tables/final_external_handoff/final_100_closeout_validation/summary.json",
    "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv",
    "tables/final_external_handoff/final_100_closeout_validation/final_100_closeout_validation.md",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def closeout_success_evidence_paths(root: Path) -> list[str]:
    manifest = read_json(root / CLOSEOUT_PACKAGE_MANIFEST)
    paths: list[str] = []
    for evidence in manifest.get("success_evidence", []) or []:
        if not isinstance(evidence, str) or "::" not in evidence:
            continue
        source, _expectation = evidence.split("::", 1)
        for item in source.split(";"):
            path = item.strip().replace("\\", "/")
            if path:
                paths.append(path)
    return paths


def closeout_required_external_input_paths(root: Path) -> list[str]:
    manifest = read_json(root / CLOSEOUT_PACKAGE_MANIFEST)
    paths: list[str] = []
    for item in manifest.get("required_external_inputs", []) or []:
        if not isinstance(item, str):
            continue
        path = item.strip().replace("\\", "/")
        if path:
            paths.append(path)
    return paths


def required_manifest_paths(root: Path) -> list[str]:
    seen = set()
    paths: list[str] = []
    for path in [
        *REQUIRED_MANIFEST_PATHS,
        *closeout_success_evidence_paths(root),
        *closeout_required_external_input_paths(root),
    ]:
        normalized = path.replace("\\", "/")
        if normalized not in seen:
            seen.add(normalized)
            paths.append(normalized)
    return paths


def valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text.lower())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rows(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    entries = {
        str(entry.get("path", "")): entry
        for entry in manifest.get("files", []) or []
        if isinstance(entry, dict)
    }
    rows = []
    for required in required_manifest_paths(root):
        entry = entries.get(required)
        path = root / required
        present = entry is not None
        exists = path.is_file() and path.stat().st_size > 0
        expected_bytes = path.stat().st_size if exists else None
        expected_sha = file_sha256(path) if exists else None
        manifest_bytes = int(entry.get("bytes", 0) or 0) if present else None
        manifest_sha = str(entry.get("sha256", "")) if present else ""
        self_generated = required.startswith(SELF_GENERATED_PREFIX)
        sha_ok = present and valid_sha256(manifest_sha) and (self_generated or expected_sha == manifest_sha)
        bytes_ok = present and manifest_bytes > 0 and (self_generated or manifest_bytes == expected_bytes)
        ok = present and sha_ok and bytes_ok and exists
        detail = (
            f"manifested bytes={manifest_bytes} sha256={manifest_sha}"
            if present
            else "missing from artifact manifest"
        )
        if present and not exists:
            detail = "manifest entry exists but local file is missing or empty"
        elif present and exists and self_generated:
            detail = (
                "self-generated coverage output manifested with valid sha256/bytes; "
                "exact hash is refreshed by the next artifact manifest rebuild"
            )
        elif present and exists and not bytes_ok:
            detail = f"manifest bytes mismatch: manifest={manifest_bytes} actual={expected_bytes}"
        elif present and exists and not sha_ok:
            detail = f"manifest sha256 mismatch: manifest={manifest_sha} actual={expected_sha}"
        rows.append(
            {
                "requirement": f"manifest_path:{required}",
                "status": "pass" if ok else "fail",
                "evidence": str(manifest_path),
                "detail": detail,
            }
        )
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    blockers = [row["requirement"] for row in rows if row["status"] != "pass"]
    return {
        "schema": SCHEMA,
        "final_manifest_coverage_ready": not blockers,
        "counts": counts,
        "blocking_requirements": blockers,
        "required_paths": [row["requirement"].removeprefix("manifest_path:") for row in rows],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["requirement", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Final Manifest Coverage",
        "",
        f"- Final manifest coverage ready: `{summary['final_manifest_coverage_ready']}`",
        "",
        "| Requirement | Status | Detail |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['requirement']}` | `{row['status']}` | {row['detail']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(root: Path, manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    rows = build_rows(root, manifest_path)
    summary = build_summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_final_manifest_coverage.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "final_manifest_coverage.md", summary, rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit non-zero when final closeout evidence is missing from the artifact manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_outputs(args.root, args.manifest, args.out_dir)
    print(
        f"Wrote final manifest coverage to {args.out_dir} "
        f"(final_manifest_coverage_ready={summary['final_manifest_coverage_ready']})"
    )
    if args.fail_on_missing and not summary["final_manifest_coverage_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
