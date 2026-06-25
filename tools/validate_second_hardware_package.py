"""Validate and optionally stage a second-hardware deployment package.

The deployment gate requires a distinct hardware profile plus production-style
energy evidence from another machine. This tool checks the returned JSON files
before they are copied into the release evidence directories, so cross-hardware
readiness cannot be opened by malformed files or a duplicate local profile.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.export_production_deployment_latency import (
    collect_hardware_profiles,
    energy_joules,
    hardware_profile_key,
)


HARDWARE_SCHEMA = "lite-seer-ad-hardware-profile-v1"
ENERGY_SCHEMA = "lite-seer-ad-production-energy-measurement-v1"
TARGET_PROFILE_DIR = Path("tables/deployment_production_latency/hardware_profiles")
TARGET_ENERGY_DIR = Path("tables/deployment_production_latency/energy_measurements")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def status_row(requirement: str, status: str, evidence: str, detail: str) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "detail": detail,
    }


def nonempty(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    return value is not None and str(value).strip() != ""


def command_mentions_infer(command: Any) -> bool:
    if isinstance(command, list):
        return any(str(part).endswith("infer.py") or str(part) == "infer.py" for part in command)
    return "infer.py" in str(command)


def validate_hardware_profile(
    root: Path,
    profile_path: Path,
    *,
    require_distinct: bool = True,
) -> list[dict[str, Any]]:
    profile = read_json(profile_path)
    rows = [
        status_row(
            "hardware:schema",
            "pass" if profile.get("schema") == HARDWARE_SCHEMA else "fail",
            str(profile_path),
            str(profile.get("schema") or "missing"),
        ),
    ]
    for key in ["label", "platform", "cpu_count", "python", "torch", "device"]:
        rows.append(
            status_row(
                f"hardware:field:{key}",
                "pass" if nonempty(profile, key) else "fail",
                str(profile_path),
                str(profile.get(key) or "missing"),
            )
        )
    if str(profile.get("device", "")).lower() == "cuda":
        rows.append(
            status_row(
                "hardware:cuda_gpu_name",
                "pass" if nonempty(profile, "gpu_name") else "fail",
                str(profile_path),
                str(profile.get("gpu_name") or "missing"),
            )
        )

    incoming_key = hardware_profile_key(profile)
    existing_keys = {
        hardware_profile_key(item)
        for item in collect_hardware_profiles(root)
        if any(hardware_profile_key(item))
    }
    distinct = bool(incoming_key) and incoming_key not in existing_keys
    rows.append(
        status_row(
            "hardware:distinct_from_existing_profiles",
            "pass" if distinct or not require_distinct else "fail",
            str(profile_path),
            f"existing_profiles={len(existing_keys)}; distinct={distinct}",
        )
    )
    return rows


def validate_energy_measurement(energy_path: Path) -> list[dict[str, Any]]:
    energy = read_json(energy_path)
    return [
        status_row(
            "energy:schema",
            "pass" if energy.get("schema") == ENERGY_SCHEMA else "fail",
            str(energy_path),
            str(energy.get("schema") or "missing"),
        ),
        status_row(
            "energy:returncode_zero",
            "pass" if int(energy.get("returncode", -1)) == 0 else "fail",
            str(energy_path),
            f"returncode={energy.get('returncode')}",
        ),
        status_row(
            "energy:positive_joules",
            "pass" if energy_joules(energy) > 0.0 else "fail",
            str(energy_path),
            f"energy_joules={energy_joules(energy)}",
        ),
        status_row(
            "energy:samples_present",
            "pass" if int(energy.get("sample_count", 0) or 0) > 0 else "fail",
            str(energy_path),
            f"sample_count={energy.get('sample_count')}",
        ),
        status_row(
            "energy:production_command_recorded",
            "pass" if command_mentions_infer(energy.get("command")) else "fail",
            str(energy_path),
            "command contains infer.py" if command_mentions_infer(energy.get("command")) else "missing infer.py command",
        ),
    ]


def validate_profile_energy_consistency(profile_path: Path, energy_path: Path) -> list[dict[str, Any]]:
    profile = read_json(profile_path)
    energy = read_json(energy_path)
    profile_device = str(profile.get("device", "")).strip().lower()
    profile_gpu = str(profile.get("gpu_name", "")).strip()
    energy_gpu = str(energy.get("gpu_name", "")).strip()
    platform_match = bool(profile.get("platform")) and profile.get("platform") == energy.get("platform")
    python_match = bool(profile.get("python")) and profile.get("python") == energy.get("python")
    gpu_match = profile_device != "cuda" or (bool(profile_gpu) and profile_gpu == energy_gpu)
    return [
        status_row(
            "hardware_energy:platform_match",
            "pass" if platform_match else "fail",
            f"{profile_path}; {energy_path}",
            f"profile={profile.get('platform') or 'missing'}; energy={energy.get('platform') or 'missing'}",
        ),
        status_row(
            "hardware_energy:python_match",
            "pass" if python_match else "fail",
            f"{profile_path}; {energy_path}",
            f"profile={profile.get('python') or 'missing'}; energy={energy.get('python') or 'missing'}",
        ),
        status_row(
            "hardware_energy:gpu_name_match",
            "pass" if gpu_match else "fail",
            f"{profile_path}; {energy_path}",
            f"profile={profile_gpu or 'missing'}; energy={energy_gpu or 'missing'}; device={profile_device or 'missing'}",
        ),
    ]


def build_rows(
    root: Path,
    hardware_profile: Path,
    energy_measurement: Path,
    *,
    require_distinct: bool = True,
) -> list[dict[str, Any]]:
    return validate_hardware_profile(
        root,
        hardware_profile,
        require_distinct=require_distinct,
    ) + validate_energy_measurement(energy_measurement) + validate_profile_energy_consistency(
        hardware_profile,
        energy_measurement,
    )


def build_summary(rows: list[dict[str, Any]], staged: dict[str, str] | None = None) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    failed = [row["requirement"] for row in rows if row["status"] != "pass"]
    return {
        "schema": "lite-seer-ad-second-hardware-package-validation-v1",
        "validation_passed": not failed,
        "counts": counts,
        "blocking_requirements": failed,
        "staged_outputs": staged or {},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["requirement", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def stage_file(source: Path, target_dir: Path, *, force: bool = False) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; use --force to overwrite")
    shutil.copy2(source, target)
    return target


def write_outputs(
    root: Path,
    out_dir: Path,
    hardware_profile: Path,
    energy_measurement: Path,
    *,
    require_distinct: bool = True,
    stage: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    rows = build_rows(
        root,
        hardware_profile,
        energy_measurement,
        require_distinct=require_distinct,
    )
    validation_passed = all(row["status"] == "pass" for row in rows)
    staged: dict[str, str] = {}
    if stage:
        if not validation_passed:
            raise ValueError("Cannot stage a second-hardware package that failed validation")
        profile_target = stage_file(
            hardware_profile,
            root / TARGET_PROFILE_DIR,
            force=force,
        )
        energy_target = stage_file(
            energy_measurement,
            root / TARGET_ENERGY_DIR,
            force=force,
        )
        staged = {
            "hardware_profile": profile_target.relative_to(root).as_posix(),
            "energy_measurement": energy_target.relative_to(root).as_posix(),
        }
    summary = build_summary(rows, staged)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_second_hardware_package_validation.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--hardware-profile", type=Path, required=True)
    parser.add_argument("--energy-measurement", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tables/deployment_production_latency/second_hardware_validation"),
    )
    parser.add_argument("--allow-duplicate-hardware", action="store_true")
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_outputs(
        args.root,
        args.out_dir,
        args.hardware_profile,
        args.energy_measurement,
        require_distinct=not args.allow_duplicate_hardware,
        stage=args.stage,
        force=args.force,
    )
    print(
        f"Wrote second-hardware validation to {args.out_dir} "
        f"(validation_passed={summary['validation_passed']})"
    )


if __name__ == "__main__":
    main()
