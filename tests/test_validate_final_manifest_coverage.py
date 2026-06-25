from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from tools.validate_final_manifest_coverage import (
    REQUIRED_MANIFEST_PATHS,
    build_rows,
    required_manifest_paths,
    main,
    write_outputs,
)


def write_text(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root: Path, *, include_all: bool = True) -> Path:
    entries = []
    paths = REQUIRED_MANIFEST_PATHS if include_all else REQUIRED_MANIFEST_PATHS[:-1]
    for index, relative in enumerate(paths):
        write_text(root / relative, f"artifact {index}\n")
        entries.append(
            {
                "path": relative,
                "sha256": sha256(root / relative),
                "bytes": (root / relative).stat().st_size,
            }
        )
    manifest = root / "artifacts/manifest.json"
    write_text(
        manifest,
        json.dumps({"schema": "lite-seer-ad-artifact-manifest-v1", "files": entries}),
    )
    return manifest


def write_manifest_for_paths(root: Path, paths: list[str]) -> Path:
    entries = []
    for index, relative in enumerate(paths):
        path = root / relative
        if not path.exists():
            write_text(path, f"artifact {index}\n")
        entries.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = root / "artifacts/manifest.json"
    write_text(
        manifest,
        json.dumps({"schema": "lite-seer-ad-artifact-manifest-v1", "files": entries}),
    )
    return manifest


def test_build_rows_passes_when_all_closeout_paths_are_manifested(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)

    rows = build_rows(tmp_path, manifest)

    assert "tables/final_external_handoff/final_input_packet/summary.json" in REQUIRED_MANIFEST_PATHS
    assert "tables/final_external_handoff/final_input_packet_validation/summary.json" in REQUIRED_MANIFEST_PATHS
    assert "tables/release_readiness/summary.json" in REQUIRED_MANIFEST_PATHS
    assert "tables/deployment_readiness/summary.json" in REQUIRED_MANIFEST_PATHS
    assert "tables/submission_package_readiness/summary.json" in REQUIRED_MANIFEST_PATHS
    assert "tables/completion_gap_matrix/summary.json" in REQUIRED_MANIFEST_PATHS
    assert "tables/final_external_handoff/summary.json" in REQUIRED_MANIFEST_PATHS
    assert len(rows) == len(REQUIRED_MANIFEST_PATHS)
    assert all(row["status"] == "pass" for row in rows)


def test_build_rows_fails_when_a_closeout_path_is_missing_from_manifest(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, include_all=False)

    rows = build_rows(tmp_path, manifest)

    assert rows[-1]["status"] == "fail"
    assert "missing from artifact manifest" in rows[-1]["detail"]


def test_build_rows_fails_when_manifest_hash_is_stale(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    required = REQUIRED_MANIFEST_PATHS[0]
    write_text(tmp_path / required, "changed after manifest\n")

    rows = build_rows(tmp_path, manifest)
    by_requirement = {row["requirement"]: row for row in rows}

    row = by_requirement[f"manifest_path:{required}"]
    assert row["status"] == "fail"
    assert "manifest" in row["detail"]
    assert "mismatch" in row["detail"]


def test_build_rows_allows_self_generated_coverage_output_to_be_refreshed(tmp_path: Path) -> None:
    closeout_manifest = tmp_path / "tables/final_external_handoff/final_100_closeout_package/manifest.json"
    self_generated = "tables/final_external_handoff/final_manifest_coverage/summary.json"
    write_text(
        closeout_manifest,
        json.dumps({"success_evidence": [f"{self_generated}::final_manifest_coverage_ready=true"]}),
    )
    paths = required_manifest_paths(tmp_path)
    manifest = write_manifest_for_paths(tmp_path, paths)
    write_text(tmp_path / self_generated, "coverage summary refreshed after manifest\n")

    rows = build_rows(tmp_path, manifest)
    by_requirement = {row["requirement"]: row for row in rows}

    row = by_requirement[f"manifest_path:{self_generated}"]
    assert row["status"] == "pass"
    assert "self-generated coverage output" in row["detail"]


def test_build_rows_includes_dynamic_closeout_success_evidence_paths(tmp_path: Path) -> None:
    closeout_manifest = tmp_path / "tables/final_external_handoff/final_100_closeout_package/manifest.json"
    write_text(
        closeout_manifest,
        json.dumps(
            {
                "success_evidence": [
                    "tables/final_external_handoff/custom_gate/summary.json::custom_ready=true"
                ]
            }
        ),
    )
    paths = required_manifest_paths(tmp_path)
    manifest = write_manifest_for_paths(tmp_path, paths)

    rows = build_rows(tmp_path, manifest)
    requirements = {row["requirement"]: row for row in rows}

    assert "tables/final_external_handoff/custom_gate/summary.json" in paths
    assert requirements["manifest_path:tables/final_external_handoff/custom_gate/summary.json"]["status"] == "pass"


def test_build_rows_includes_closeout_required_external_inputs(tmp_path: Path) -> None:
    closeout_manifest = tmp_path / "tables/final_external_handoff/final_100_closeout_package/manifest.json"
    required_inputs = [
        "release_metadata.json",
        "submission_metadata.json",
        "second_hardware_profile.json",
        "second_hardware_energy.json",
    ]
    write_text(
        closeout_manifest,
        json.dumps({"required_external_inputs": required_inputs}),
    )
    for relative in required_inputs:
        write_text(tmp_path / relative, '{"ready": true}\n')
    paths = required_manifest_paths(tmp_path)
    manifest = write_manifest_for_paths(tmp_path, paths)

    rows = build_rows(tmp_path, manifest)
    requirements = {row["requirement"]: row for row in rows}

    for relative in required_inputs:
        assert relative in paths
        assert requirements[f"manifest_path:{relative}"]["status"] == "pass"


def test_write_outputs_creates_manifest_coverage_artifacts(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)

    summary = write_outputs(tmp_path, manifest, tmp_path / "out")

    assert summary["final_manifest_coverage_ready"] is True
    assert (tmp_path / "out/summary.json").is_file()
    assert (tmp_path / "out/table_final_manifest_coverage.csv").is_file()
    assert (tmp_path / "out/final_manifest_coverage.md").is_file()


def test_cli_can_fail_on_missing_manifest_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_manifest(tmp_path, include_all=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["validate_final_manifest_coverage.py", "--fail-on-missing"])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
