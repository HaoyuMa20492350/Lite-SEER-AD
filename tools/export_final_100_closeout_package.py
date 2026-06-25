"""Export the final 100% closeout package.

The individual blockers already have focused tools. This package ties them
together into a single auditable finalization path, without marking external
work complete before the real second-hardware evidence and release/submission
metadata exist.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA = "lite-seer-ad-final-100-closeout-package-v1"
DEFAULT_OUT_DIR = Path("tables/final_external_handoff/final_100_closeout_package")
REQUIRED_EXTERNAL_INPUTS = [
    "second_hardware_profile.json",
    "second_hardware_energy.json",
    "release_metadata.json",
    "submission_metadata.json",
]
SUCCESS_EVIDENCE = [
    "tables/final_external_handoff/final_input_packet/summary.json::packet_ready=true",
    "tables/final_external_handoff/final_input_packet_validation/summary.json::final_input_packet_valid=true",
    "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv::handoff:input_packet_matches_external_handoff=pass",
    "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv::closeout:package_manifest_valid=pass",
    "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv::closeout:package_artifacts_match_manifest=pass",
    "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv::closeout:required_external_inputs_present=pass",
    "tables/final_external_handoff/final_100_closeout_validation/table_final_100_closeout_validation.csv::closeout:success_evidence_expectations=pass",
    "tables/final_external_handoff/final_metadata_consistency/summary.json::final_metadata_consistent=true",
    "tables/final_external_handoff/final_metadata_consistency/table_final_metadata_consistency.csv::consistency:manuscript_title=pass",
    "tables/final_external_handoff/final_metadata_consistency/table_final_metadata_consistency.csv::consistency:citation_authors=pass",
    "tables/final_external_handoff/final_metadata_consistency/table_final_metadata_consistency.csv::consistency:zenodo_creators=pass",
    "tables/final_external_handoff/final_manifest_coverage/summary.json::final_manifest_coverage_ready=true",
    "tables/deployment_readiness/summary.json::production_deployment_ready=true",
    "tables/release_readiness/summary.json::release_gate_passed=true",
    "tables/submission_package_readiness/summary.json::final_upload_ready=true",
    "tables/completion_gap_matrix/summary.json::default_100_ready=true",
    "tables/final_external_handoff/summary.json::all_external_actions_complete=true",
    "tables/final_external_handoff/final_100_closeout_validation/summary.json::final_100_ready=true",
]
FINALIZATION_COMMANDS = [
    "python tools/validate_second_hardware_package.py --hardware-profile $SecondHardwareProfile --energy-measurement $SecondHardwareEnergy --stage",
    "python tools/validate_release_submission_consistency.py --fail-on-inconsistent",
    "python tools/render_release_metadata.py --input release_metadata.json",
    "python tools/render_submission_statements.py --input submission_metadata.json --out docs/submission_statement_placeholders.md",
    "python tools/export_production_deployment_latency.py --command-run-prefix production_lc_rds_budget_guarded",
    "python tools/export_deployment_readiness.py",
    "python scripts/release/build_artifact_manifest.py --out artifacts/manifest.json",
    "python tools/export_release_readiness.py",
    "python tools/export_submission_package_readiness.py",
    "python tools/export_completion_gap_matrix.py",
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
]
OPTIONAL_TEST_COMMAND = "python -m pytest -q --basetemp .tmp_pytest_final_100 -p no:cacheprovider"


def render_final_script(commands: list[str]) -> str:
    lines = [
        "param(",
        "  [Parameter(Mandatory=$true)] [string]$SecondHardwareProfile,",
        "  [Parameter(Mandatory=$true)] [string]$SecondHardwareEnergy,",
        "  [string]$RepoRoot = '',",
        "  [switch]$SkipTests",
        ")",
        "",
        "$ErrorActionPreference = 'Stop'",
        "",
        "$SecondHardwareProfile = (Resolve-Path -LiteralPath $SecondHardwareProfile).Path",
        "$SecondHardwareEnergy = (Resolve-Path -LiteralPath $SecondHardwareEnergy).Path",
        "",
        "if (-not $RepoRoot) {",
        "  $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\\..\\..')).Path",
        "}",
        "else {",
        "  $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path",
        "}",
        "$CanonicalSecondHardwareProfile = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot 'second_hardware_profile.json'))",
        "$CanonicalSecondHardwareEnergy = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot 'second_hardware_energy.json'))",
        "if ($SecondHardwareProfile -ne $CanonicalSecondHardwareProfile) {",
        "  Copy-Item -LiteralPath $SecondHardwareProfile -Destination $CanonicalSecondHardwareProfile -Force",
        "}",
        "if ($SecondHardwareEnergy -ne $CanonicalSecondHardwareEnergy) {",
        "  Copy-Item -LiteralPath $SecondHardwareEnergy -Destination $CanonicalSecondHardwareEnergy -Force",
        "}",
        "$SecondHardwareProfile = $CanonicalSecondHardwareProfile",
        "$SecondHardwareEnergy = $CanonicalSecondHardwareEnergy",
        "Push-Location $RepoRoot",
        "try {",
        "  foreach ($Path in @($SecondHardwareProfile, $SecondHardwareEnergy, 'release_metadata.json', 'submission_metadata.json')) {",
        "    if (-not (Test-Path -LiteralPath $Path)) {",
        "      throw \"Required final closeout input is missing: $Path\"",
        "    }",
        "  }",
        "",
    ]
    for command in commands:
        lines.append(f"  {command}")
        lines.append("  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }")
        lines.append("")
    lines.extend(
        [
            "  if (-not $SkipTests) {",
            f"    {OPTIONAL_TEST_COMMAND}",
            "    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
            "  }",
            "",
            "  Write-Host 'Final closeout commands completed. Inspect readiness summaries before claiming 100%.'",
            "}",
            "finally {",
            "  Pop-Location",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Final 100% Closeout Package",
        "",
        "This package is the last-mile checklist for the plan-defined 100% state.",
        "It does not replace the readiness gates; it runs the tools that can make those gates pass once the real external inputs exist.",
        "",
        "Required external inputs:",
        "",
    ]
    for item in manifest["required_external_inputs"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "Before running the final script, generate and use the focused helper packages:",
            "",
            "- `tables/deployment_production_latency/second_hardware_run_package/` for the second-machine probe.",
            "- `tables/submission_package_readiness/final_upload_closeout_package/` for release/submission metadata rendering.",
            "- `tables/final_external_handoff/final_input_packet/` for the field-level final input checklist.",
            "",
            "Final command:",
            "",
            "```powershell",
            ".\\finalize_100_percent.ps1 `",
            "  -RepoRoot <path-to-LITE-SEER-AD> `",
            "  -SecondHardwareProfile <absolute-or-caller-relative-second_hardware_profile.json> `",
            "  -SecondHardwareEnergy <absolute-or-caller-relative-second_hardware_energy.json>",
            "```",
            "",
            "The second-hardware paths are resolved before the script enters `RepoRoot`; pass absolute paths or paths relative to the shell location where you launch the script.",
            "",
            "Success evidence:",
            "",
        ]
    )
    for item in manifest["success_evidence"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "Only claim 100% after every evidence item above is true in the refreshed summaries.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_audit_template(manifest: dict[str, Any]) -> str:
    lines = [
        "# Final 100% Completion Audit",
        "",
        "| Requirement | Evidence | Status | Notes |",
        "|---|---|---|---|",
    ]
    for evidence in manifest["success_evidence"]:
        source, expectation = evidence.split("::", 1)
        lines.append(f"| `{expectation}` | `{source}` | pending | Fill after final closeout run. |")
    return "\n".join(lines) + "\n"


def build_manifest() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "required_external_inputs": REQUIRED_EXTERNAL_INPUTS,
        "commands": FINALIZATION_COMMANDS,
        "success_evidence": SUCCESS_EVIDENCE,
        "optional_test_command": OPTIONAL_TEST_COMMAND,
    }


def write_outputs(out_dir: Path) -> dict[str, Any]:
    manifest = build_manifest()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(render_readme(manifest), encoding="utf-8")
    (out_dir / "finalize_100_percent.ps1").write_text(
        render_final_script(FINALIZATION_COMMANDS),
        encoding="utf-8",
    )
    (out_dir / "completion_audit_template.md").write_text(
        render_audit_template(manifest),
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_outputs(args.out_dir)
    print(
        f"Wrote final 100% closeout package to {args.out_dir} "
        f"(required_inputs={len(manifest['required_external_inputs'])})"
    )


if __name__ == "__main__":
    main()
