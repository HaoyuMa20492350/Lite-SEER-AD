from __future__ import annotations

import json
from pathlib import Path

from tools.export_submission_finalization_package import (
    FINAL_COMMANDS,
    build_manifest,
    write_outputs,
)


def test_manifest_declares_required_private_inputs() -> None:
    manifest = build_manifest()

    assert manifest["schema"] == "lite-seer-ad-submission-finalization-package-v1"
    assert manifest["required_inputs"] == ["release_metadata.json", "submission_metadata.json"]
    assert "tables/completion_gap_matrix/summary.json" in manifest["readiness_evidence"]


def test_final_commands_render_release_before_submission_statements() -> None:
    assert FINAL_COMMANDS[0] == "python tools/validate_release_submission_consistency.py --fail-on-inconsistent"
    assert FINAL_COMMANDS[1] == "python tools/render_release_metadata.py --input release_metadata.json"
    assert FINAL_COMMANDS[2].startswith("python tools/render_submission_statements.py")
    assert FINAL_COMMANDS[-1] == "python tools/export_final_external_handoff.py"


def test_write_outputs_creates_closeout_package(tmp_path: Path) -> None:
    manifest = write_outputs(tmp_path)

    assert (tmp_path / "README.md").is_file()
    assert (tmp_path / "finalize_submission_upload.ps1").is_file()
    assert (tmp_path / "manifest.json").is_file()
    loaded = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert loaded["commands"] == manifest["commands"]


def test_closeout_script_contains_all_gate_refresh_commands(tmp_path: Path) -> None:
    write_outputs(tmp_path)
    script = (tmp_path / "finalize_submission_upload.ps1").read_text(encoding="utf-8")

    assert "render_release_metadata.py" in script
    assert "validate_release_submission_consistency.py --fail-on-inconsistent" in script
    assert "render_submission_statements.py" in script
    assert "export_release_readiness.py" in script
    assert "export_submission_package_readiness.py" in script
    assert "export_completion_gap_matrix.py" in script
    assert "export_final_external_handoff.py" in script
    assert "RepoRoot" in script
    assert "Push-Location $RepoRoot" in script
    assert "Pop-Location" in script
