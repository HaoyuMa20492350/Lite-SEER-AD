"""Export a runnable package for the second-hardware deployment probe.

The final deployment gate needs evidence from another machine. This exporter
does not fake that evidence; it writes a small runbook and PowerShell scripts
that make the external run deterministic and easy to validate once the JSON
files are returned.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


SCHEMA = "lite-seer-ad-second-hardware-run-package-v1"
DEFAULT_OUT_DIR = Path("tables/deployment_production_latency/second_hardware_run_package")
DEFAULT_RETURN_DIR = "second_hardware_return_package"
DEFAULT_RUN_NAME = "second_hardware_probe"
DEFAULT_LABEL = "second_hardware"


def default_infer_command(
    *,
    category: str,
    run_name: str,
    budget_ms: int,
    max_samples: int,
    seed: int,
    image_size: int,
) -> list[str]:
    return [
        "python",
        "infer.py",
        "--config",
        "configs/mvtec.yaml",
        "--category",
        category,
        "--checkpoint",
        f"runs/fulltest_mvtec15_{category}_models/diffusion.pt",
        "--sev-checkpoint",
        f"runs/fulltest_mvtec15_{category}_models/hn_sev.pt",
        "--feature-prior-checkpoint",
        f"runs/fulltest_mvtec15_{category}_models/feature_prior.pt",
        "--run-name",
        run_name,
        "--ablation",
        "utility_lc_rds",
        "--latency-budget-ms",
        str(budget_ms),
        "--image-size",
        str(image_size),
        "--max-samples",
        str(max_samples),
        "--seed",
        str(seed),
        "--device",
        "auto",
        "--image-score-mode",
        "top5",
        "--image-score-source",
        "feature_raw_cosine",
        "--pixel-heatmap-source",
        "feature_raw",
        "--reconstruction-steps",
        "5",
    ]


def ps_single_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def ps_join(parts: list[str]) -> str:
    return " ".join(ps_single_quote(part) for part in parts)


def render_repo_root_setup() -> list[str]:
    return [
        "if (-not $RepoRoot) {",
        "  $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\\..\\..')).Path",
        "}",
        "else {",
        "  $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path",
        "}",
        "Push-Location $RepoRoot",
        "try {",
    ]


def indent_ps(lines: list[str]) -> list[str]:
    return [f"  {line}" if line else "" for line in lines]


def render_run_script(
    *,
    label: str,
    run_name: str,
    return_dir: str,
    infer_command: list[str],
    energy_filename: str,
    profile_filename: str,
) -> str:
    infer_tail = ps_join(infer_command)
    body = [
            "New-Item -ItemType Directory -Force -Path $ReturnDir | Out-Null",
            f"$EnergyPath = Join-Path $ReturnDir {ps_single_quote(energy_filename)}",
            f"$ProfilePath = Join-Path $ReturnDir {ps_single_quote(profile_filename)}",
            "",
            "python tools/measure_production_energy.py `",
            "  --output $EnergyPath `",
            "  --label $Label `",
            f"  -- {infer_tail}",
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
            "",
            "python tools/collect_hardware_profile.py `",
            "  --output $ProfilePath `",
            "  --label $Label `",
            "  --validation-run \"runs/$RunName\"",
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
            "",
            "Write-Host \"Second-hardware package ready:\"",
            "Write-Host \"  $EnergyPath\"",
            "Write-Host \"  $ProfilePath\"",
    ]
    lines = [
        "param(",
        f"  [string]$Label = {ps_single_quote(label)},",
        f"  [string]$RunName = {ps_single_quote(run_name)},",
        f"  [string]$ReturnDir = {ps_single_quote(return_dir)},",
        "  [string]$RepoRoot = ''",
        ")",
        "",
        "$ErrorActionPreference = 'Stop'",
        "if (-not $RepoRoot) {",
        "  $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\\..\\..')).Path",
        "}",
        "else {",
        "  $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path",
        "}",
        "if ([System.IO.Path]::IsPathRooted($ReturnDir)) {",
        "  $ReturnDir = [System.IO.Path]::GetFullPath($ReturnDir)",
        "}",
        "else {",
        "  $ReturnDir = Join-Path $RepoRoot $ReturnDir",
        "}",
        "Push-Location $RepoRoot",
        "try {",
        *indent_ps(body),
        "}",
        "finally {",
        "  Pop-Location",
        "}",
    ]
    return "\n".join(lines) + "\n"


def render_validate_script(
    *,
    return_dir: str,
    energy_filename: str,
    profile_filename: str,
) -> str:
    body = [
            f"$EnergyPath = Join-Path $ReturnDir {ps_single_quote(energy_filename)}",
            f"$ProfilePath = Join-Path $ReturnDir {ps_single_quote(profile_filename)}",
            "",
            "python tools/validate_second_hardware_package.py `",
            "  --hardware-profile $ProfilePath `",
            "  --energy-measurement $EnergyPath `",
            "  --stage",
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
            "",
            "python tools/export_production_deployment_latency.py --command-run-prefix production_lc_rds_budget_guarded",
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
            "",
            "python tools/export_deployment_readiness.py",
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
            "",
            "python tools/export_completion_gap_matrix.py",
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
    ]
    lines = [
        "param(",
        f"  [string]$ReturnDir = {ps_single_quote(return_dir)},",
        "  [string]$RepoRoot = ''",
        ")",
        "",
        "$ErrorActionPreference = 'Stop'",
        "if (-not $RepoRoot) {",
        "  $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\\..\\..')).Path",
        "}",
        "else {",
        "  $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path",
        "}",
        "if ([System.IO.Path]::IsPathRooted($ReturnDir)) {",
        "  $ReturnDir = (Resolve-Path -LiteralPath $ReturnDir).Path",
        "}",
        "elseif (Test-Path -LiteralPath $ReturnDir) {",
        "  $ReturnDir = (Resolve-Path -LiteralPath $ReturnDir).Path",
        "}",
        "else {",
        "  $ReturnDir = (Resolve-Path -LiteralPath (Join-Path $RepoRoot $ReturnDir)).Path",
        "}",
        "Push-Location $RepoRoot",
        "try {",
        *indent_ps(body),
        "}",
        "finally {",
        "  Pop-Location",
        "}",
    ]
    return "\n".join(lines) + "\n"


def render_readme(manifest: dict[str, Any]) -> str:
    outputs = manifest["expected_return_files"]
    lines = [
        "# Second-Hardware Deployment Probe",
        "",
        "Run this package on a second machine that has the Lite-SEER-AD workspace, dependencies, checkpoints, and dataset paths available.",
        "",
        "## On The Second Machine",
        "",
        "```powershell",
        ".\\run_second_hardware_probe.ps1 `",
        "  -RepoRoot <path-to-LITE-SEER-AD> `",
        "  -ReturnDir <absolute-or-repo-root-relative-return-directory>",
        "```",
        "",
        "On the second machine, absolute `-ReturnDir` values are used directly; relative values are created under `RepoRoot`.",
        "",
        "Return these files to the main workspace:",
        "",
    ]
    for item in outputs:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## On The Main Workspace",
            "",
            "Copy the returned directory into the repository root, or pass its location explicitly:",
            "",
        "```powershell",
        ".\\validate_returned_second_hardware_package.ps1 `",
        "  -RepoRoot <path-to-LITE-SEER-AD> `",
        "  -ReturnDir <absolute-or-caller-relative-return-directory>",
            "```",
            "",
            "If `-ReturnDir` is omitted, the script first checks the launch directory and then falls back to `<RepoRoot>/second_hardware_return_package`.",
            "",
            "The validation command stages only schema-valid, distinct second-hardware evidence and refreshes deployment/completion readiness summaries.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_manifest(
    *,
    label: str,
    run_name: str,
    return_dir: str,
    infer_command: list[str],
    energy_filename: str,
    profile_filename: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": label,
        "run_name": run_name,
        "return_dir": return_dir,
        "expected_return_files": [
            f"{return_dir}/{energy_filename}",
            f"{return_dir}/{profile_filename}",
        ],
        "second_machine_command": "./run_second_hardware_probe.ps1",
        "main_workspace_validation_command": "./validate_returned_second_hardware_package.ps1",
        "infer_command": infer_command,
    }


def write_outputs(
    out_dir: Path,
    *,
    label: str = DEFAULT_LABEL,
    run_name: str = DEFAULT_RUN_NAME,
    return_dir: str = DEFAULT_RETURN_DIR,
    category: str = "bottle",
    budget_ms: int = 10,
    max_samples: int = 8,
    seed: int = 7,
    image_size: int = 128,
) -> dict[str, Any]:
    energy_filename = f"{run_name}_energy.json"
    profile_filename = f"{run_name}_hardware_profile.json"
    infer_command = default_infer_command(
        category=category,
        run_name=run_name,
        budget_ms=budget_ms,
        max_samples=max_samples,
        seed=seed,
        image_size=image_size,
    )
    manifest = build_manifest(
        label=label,
        run_name=run_name,
        return_dir=return_dir,
        infer_command=infer_command,
        energy_filename=energy_filename,
        profile_filename=profile_filename,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_second_hardware_probe.ps1").write_text(
        render_run_script(
            label=label,
            run_name=run_name,
            return_dir=return_dir,
            infer_command=infer_command,
            energy_filename=energy_filename,
            profile_filename=profile_filename,
        ),
        encoding="utf-8",
    )
    (out_dir / "validate_returned_second_hardware_package.ps1").write_text(
        render_validate_script(
            return_dir=return_dir,
            energy_filename=energy_filename,
            profile_filename=profile_filename,
        ),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(render_readme(manifest), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--return-dir", default=DEFAULT_RETURN_DIR)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--budget-ms", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_outputs(
        args.out_dir,
        label=args.label,
        run_name=args.run_name,
        return_dir=args.return_dir,
        category=args.category,
        budget_ms=args.budget_ms,
        max_samples=args.max_samples,
        seed=args.seed,
        image_size=args.image_size,
    )
    print(
        f"Wrote second-hardware run package to {args.out_dir} "
        f"(return_files={len(manifest['expected_return_files'])})"
    )


if __name__ == "__main__":
    main()
