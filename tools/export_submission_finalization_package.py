"""Export a final-upload closeout package for manuscript submission.

The package keeps journal/author metadata and public release identifiers as
explicit external inputs. It provides the commands that turn those inputs into
the statement page and refreshed readiness summaries once the real values are
available.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA = "lite-seer-ad-submission-finalization-package-v1"
DEFAULT_OUT_DIR = Path("tables/submission_package_readiness/final_upload_closeout_package")
REQUIRED_INPUTS = [
    "release_metadata.json",
    "submission_metadata.json",
]
GENERATED_OUTPUTS = [
    "tables/final_external_handoff/final_metadata_consistency/summary.json",
    "release_links.json",
    "CITATION.cff",
    ".zenodo.json",
    "docs/public_release_identifiers.md",
    "docs/submission_statement_placeholders.md",
    "tables/release_readiness/summary.json",
    "tables/submission_package_readiness/summary.json",
    "tables/completion_gap_matrix/summary.json",
    "tables/final_external_handoff/summary.json",
]
FINAL_COMMANDS = [
    "python tools/validate_release_submission_consistency.py --fail-on-inconsistent",
    "python tools/render_release_metadata.py --input release_metadata.json",
    "python tools/render_submission_statements.py --input submission_metadata.json --out docs/submission_statement_placeholders.md",
    "python tools/export_release_readiness.py",
    "python tools/export_submission_package_readiness.py",
    "python tools/export_completion_gap_matrix.py",
    "python tools/export_final_external_handoff.py",
]


def render_powershell(commands: list[str]) -> str:
    lines = [
        "param(",
        "  [string]$RepoRoot = ''",
        ")",
        "",
        "$ErrorActionPreference = 'Stop'",
        "",
        "if (-not $RepoRoot) {",
        "  $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\\..\\..')).Path",
        "}",
        "Push-Location $RepoRoot",
        "try {",
        "  foreach ($Path in @('release_metadata.json', 'submission_metadata.json')) {",
        "    if (-not (Test-Path $Path)) {",
        "      throw \"Required submission finalization input is missing: $Path\"",
        "    }",
        "  }",
        "",
    ]
    for command in commands:
        lines.append(f"  {command}")
        lines.append("  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }")
        lines.append("")
    lines.append("  Write-Host 'Submission finalization commands completed.'")
    lines.append("}")
    lines.append("finally {")
    lines.append("  Pop-Location")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Submission Final Upload Closeout",
        "",
        "Fill the private metadata files at the repository root before running the closeout script.",
        "",
        "Required inputs:",
        "",
    ]
    for item in manifest["required_inputs"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "Command:",
            "",
            "```powershell",
            ".\\finalize_submission_upload.ps1 -RepoRoot <path-to-LITE-SEER-AD>",
            "```",
            "",
            "Expected refreshed outputs:",
            "",
        ]
    )
    for item in manifest["generated_outputs"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "The package does not make the final upload ready by itself; readiness is proven only when the refreshed summaries report passing release, submission, and completion gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_manifest() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "required_inputs": REQUIRED_INPUTS,
        "generated_outputs": GENERATED_OUTPUTS,
        "commands": FINAL_COMMANDS,
        "readiness_evidence": [
            "tables/final_external_handoff/final_metadata_consistency/summary.json",
            "tables/release_readiness/summary.json",
            "tables/submission_package_readiness/summary.json",
            "tables/completion_gap_matrix/summary.json",
            "tables/final_external_handoff/summary.json",
        ],
    }


def write_outputs(out_dir: Path) -> dict[str, Any]:
    manifest = build_manifest()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(render_readme(manifest), encoding="utf-8")
    (out_dir / "finalize_submission_upload.ps1").write_text(
        render_powershell(FINAL_COMMANDS),
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
        f"Wrote submission finalization package to {args.out_dir} "
        f"(commands={len(manifest['commands'])})"
    )


if __name__ == "__main__":
    main()
