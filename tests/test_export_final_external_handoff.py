from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.export_final_external_handoff import build_rows, build_summary, write_outputs


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_handoff_repo(root: Path) -> None:
    write_json(
        root / "tables/deployment_readiness/summary.json",
        {
            "release_gate_passed": False,
            "blocking_requirements": ["production:cross_hardware"],
        },
    )
    write_json(
        root / "tables/deployment_production_latency/summary.json",
        {"hardware_profiles": 1, "cross_hardware_ready": False},
    )
    write_csv(
        root / "tables/release_readiness/table_release_readiness.csv",
        [
            {
                "requirement": "external_link:github_release_url",
                "status": "pending_external",
                "evidence": "release_links.json",
                "detail": "missing",
                "in_manifest": "False",
            },
            {
                "requirement": "local_file:README.md",
                "status": "pass",
                "evidence": "README.md",
                "detail": "present",
                "in_manifest": "True",
            },
            {
                "requirement": "external_consistency:citation_matches_release_links",
                "status": "pending_external",
                "evidence": "CITATION.cff; release_links.json",
                "detail": "release links are missing or not final",
                "in_manifest": "False",
            },
        ],
    )
    write_csv(
        root / "tables/submission_package_readiness/table_submission_readiness.csv",
        [
            {
                "requirement": "final_upload:authors_affiliations",
                "status": "pending_journal",
                "gate": "final_upload",
                "evidence": "docs/submission_statement_placeholders.md",
                "detail": "contains target-journal or author placeholder",
            }
        ],
    )


def test_handoff_collects_external_rows(tmp_path: Path) -> None:
    make_handoff_repo(tmp_path)

    rows = build_rows(tmp_path)
    summary = build_summary(rows)

    assert [row["requirement"] for row in rows] == [
        "production:cross_hardware",
        "external_link:github_release_url",
        "external_consistency:citation_matches_release_links",
        "final_upload:authors_affiliations",
    ]
    assert summary["all_external_actions_complete"] is False
    assert summary["counts"]["pending_external"] == 3
    assert summary["counts"]["pending_journal"] == 1
    deployment = rows[0]
    assert "run_second_hardware_probe.ps1" in deployment["owner_input"]
    assert "validate_second_hardware_package.py" in deployment["completion_command"]
    assert "--stage" in deployment["completion_command"]


def test_write_outputs_creates_handoff_artifacts(tmp_path: Path) -> None:
    make_handoff_repo(tmp_path)

    summary = write_outputs(tmp_path, tmp_path / "tables/final_external_handoff")

    assert summary["schema"] == "lite-seer-ad-final-external-handoff-v1"
    assert (tmp_path / "tables/final_external_handoff/summary.json").is_file()
    assert (tmp_path / "tables/final_external_handoff/table_final_external_handoff.csv").is_file()
    assert (tmp_path / "tables/final_external_handoff/final_external_handoff.md").is_file()
