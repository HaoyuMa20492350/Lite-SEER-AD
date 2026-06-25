from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.export_final_input_packet import (
    build_field_rows,
    build_handoff_input_coverage_rows,
    packet_blockers,
    write_outputs,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_packet_repo(root: Path) -> None:
    write_json(
        root / "release_metadata.template.json",
        {
            "schema": "lite-seer-ad-release-metadata-v1",
            "release": {
                "github_release_url": "https://github.com/<owner>/<repo>/releases/tag/v1.0-paper",
                "zenodo_doi": "10.5281/zenodo.<record_id>",
                "hf_model_url": "https://huggingface.co/<owner>/<model-repo>",
                "hf_dataset_url": "https://huggingface.co/datasets/<owner>/<dataset-repo>",
            },
            "citation": {
                "message": "If you use Lite-SEER-AD, please cite the paper package and release.",
                "title": "Lite-SEER-AD",
                "version": "1.0-paper",
                "date_released": "2026-06-21",
                "repository_code": "https://github.com/<owner>/<repo>",
                "license": "LicenseRef-All-Rights-Reserved",
                "authors": [{"family_names": "<family name>", "given_names": "<given name>"}],
            },
            "zenodo": {
                "title": "Lite-SEER-AD",
                "upload_type": "software",
                "description": "Claim-bounded paper package.",
                "creators": [{"name": "<family name>, <given name>"}],
                "license": "other-closed",
                "keywords": ["industrial anomaly detection"],
                "version": "1.0-paper",
            },
        },
    )
    write_json(
        root / "submission_metadata.template.json",
        {
            "target_journal": {
                "journal": "Pattern Recognition",
                "publisher_platform": "Elsevier / ScienceDirect",
                "article_type": "Full Length Article",
                "template_status": "Template selected",
                "required_word_page_limit": "Guide checked",
                "supplement_format": "Separate supplement",
                "guide_source_checked": "https://www.journals.elsevier.com/pattern-recognition",
            },
            "authors": {
                "corresponding_author": {
                    "name": "<name>",
                    "email": "<email>",
                    "affiliation": "<affiliation>",
                },
                "author_list": ["<ordered author names>"],
                "affiliations": ["<institution, department, city, country>"],
                "orcid_ids": ["<optional or required ORCID IDs>"],
            },
            "funding_statement": "<exact funding statement>",
            "conflict_of_interest_statement": "<exact conflict of interest statement>",
            "data_availability_statement": (
                "GitHub Release URL: <github release url>. "
                "Zenodo DOI: <zenodo doi>. "
                "Hugging Face URL: <hugging face model or dataset url>."
            ),
            "code_availability_statement": (
                "repository URL: <repository url>. The exact release commit is <commit sha>."
            ),
            "reproducibility_statement": "All retained claims are tied to fixed configs and hashes.",
            "ethics_statement": "This study does not involve human participants, personal data, or animal experiments.",
            "author_contributions": {"Conceptualization": ["<names>"]},
            "cover_letter": {
                "editor_salutation": "Dear Editor,",
                "manuscript_title": "Lite-SEER-AD",
                "article_type": "<article type>",
                "journal": "<journal>",
                "availability_sentence": "All retained claims are backed by final public artifact links.",
                "closing_name": "<corresponding author name>",
            },
        },
    )
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
    write_json(
        root / "tables/deployment_production_latency/second_hardware_run_package/manifest.json",
        {
            "schema": "lite-seer-ad-second-hardware-run-package-v1",
            "expected_return_files": [
                "second_hardware_return_package/second_hardware_probe_energy.json",
                "second_hardware_return_package/second_hardware_probe_hardware_profile.json",
            ],
        },
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
            }
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
                "detail": "placeholder",
            }
        ],
    )


def test_field_rows_collect_second_hardware_and_metadata_placeholders(tmp_path: Path) -> None:
    make_packet_repo(tmp_path)

    rows = build_field_rows(tmp_path)
    requirements = [row["requirement"] for row in rows]

    assert "external_input:second_hardware_profile.json" in requirements
    assert "external_input:second_hardware_energy.json" in requirements
    assert any("release_metadata.json::release.github_release_url" in item for item in requirements)
    assert any("submission_metadata.json::authors.corresponding_author.email" in item for item in requirements)
    assert all(row["destination_file"] for row in rows)
    assert all(row["validator"] for row in rows)
    assert all("suggested_value_or_hint" in row for row in rows)
    profile_row = next(row for row in rows if row["requirement"] == "external_input:second_hardware_profile.json")
    energy_row = next(row for row in rows if row["requirement"] == "external_input:second_hardware_energy.json")
    assert "second_hardware_probe_hardware_profile.json" in profile_row["suggested_value_or_hint"]
    assert "second_hardware_probe_energy.json" in energy_row["suggested_value_or_hint"]


def test_field_rows_prefill_github_suggestions_from_origin(tmp_path: Path) -> None:
    make_packet_repo(tmp_path)
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(
        '[remote "origin"]\n'
        "\turl = https://github.com/HaoyuMa20492350/Lite-SEER-AD.git\n",
        encoding="utf-8",
    )

    rows = build_field_rows(tmp_path)
    by_requirement = {row["requirement"]: row for row in rows}

    release_url = by_requirement[
        "metadata_field:release_metadata.json::release.github_release_url"
    ]["suggested_value_or_hint"]
    repository = by_requirement[
        "metadata_field:release_metadata.json::citation.repository_code"
    ]["suggested_value_or_hint"]

    assert release_url.startswith(
        "https://github.com/HaoyuMa20492350/Lite-SEER-AD/releases/tag/v1.0-paper"
    )
    assert "create this release before marking the gate complete" in release_url
    assert repository == "https://github.com/HaoyuMa20492350/Lite-SEER-AD"


def test_write_outputs_prefills_only_repository_backed_draft_fields(tmp_path: Path) -> None:
    make_packet_repo(tmp_path)
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text(
        '[remote "origin"]\n'
        "\turl = https://github.com/HaoyuMa20492350/Lite-SEER-AD.git\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "packet"

    summary = write_outputs(tmp_path, out_dir)
    release_draft = json.loads((out_dir / "release_metadata.draft.json").read_text(encoding="utf-8"))
    submission_draft = json.loads((out_dir / "submission_metadata.draft.json").read_text(encoding="utf-8"))
    readme = (out_dir / "README.md").read_text(encoding="utf-8")

    assert release_draft["citation"]["repository_code"] == "https://github.com/HaoyuMa20492350/Lite-SEER-AD"
    assert release_draft["release"]["github_release_url"] == "https://github.com/<owner>/<repo>/releases/tag/v1.0-paper"
    assert "https://github.com/HaoyuMa20492350/Lite-SEER-AD" in submission_draft["code_availability_statement"]
    assert "<commit sha>" in submission_draft["code_availability_statement"]
    assert "release_metadata.draft.json::citation.repository_code" in summary["draft_prefill_fields"]
    assert "submission_metadata.draft.json::code_availability_statement.repository_url" in summary["draft_prefill_fields"]
    assert "Draft prefilled fields:" in readme


def test_write_outputs_creates_final_input_packet(tmp_path: Path) -> None:
    make_packet_repo(tmp_path)
    out_dir = tmp_path / "tables/final_external_handoff/final_input_packet"

    summary = write_outputs(tmp_path, out_dir)

    assert summary["schema"] == "lite-seer-ad-final-input-packet-v1"
    assert summary["packet_ready"] is True
    assert summary["packet_blocking_requirements"] == []
    assert summary["metadata_input_sources"][0]["template_exists"] is True
    assert summary["metadata_input_sources"][0]["final_exists"] is False
    assert summary["unresolved_handoff_rows"] == 3
    assert summary["handoff_input_coverage_rows"] == 3
    assert summary["handoff_input_coverage_ready"] is True
    assert summary["second_hardware_inputs"] == 2
    assert summary["metadata_placeholder_fields"] >= 8
    assert (out_dir / "release_metadata.draft.json").is_file()
    assert (out_dir / "submission_metadata.draft.json").is_file()
    assert (out_dir / "second_hardware_inputs.json").is_file()
    second_hardware_inputs = json.loads((out_dir / "second_hardware_inputs.json").read_text(encoding="utf-8"))
    assert second_hardware_inputs["expected_return_files"] == [
        "second_hardware_return_package/second_hardware_probe_energy.json",
        "second_hardware_return_package/second_hardware_probe_hardware_profile.json",
    ]
    assert (out_dir / "table_final_input_fields.csv").is_file()
    assert (out_dir / "table_handoff_input_coverage.csv").is_file()
    assert (out_dir / "README.md").is_file()
    assert (out_dir / "summary.json").is_file()


def test_readme_keeps_final_gate_honest(tmp_path: Path) -> None:
    make_packet_repo(tmp_path)
    out_dir = tmp_path / "packet"

    write_outputs(tmp_path, out_dir)
    readme = (out_dir / "README.md").read_text(encoding="utf-8")

    assert "does not mark any external requirement complete" in readme
    assert "release_metadata.json" in readme
    assert "submission_metadata.json" in readme
    assert "second_hardware_profile.json" in readme
    assert "table_handoff_input_coverage.csv" in readme
    assert "validate_second_hardware_package.py" in readme


def test_missing_metadata_template_blocks_packet_ready(tmp_path: Path) -> None:
    make_packet_repo(tmp_path)
    (tmp_path / "submission_metadata.template.json").unlink()

    summary = write_outputs(tmp_path, tmp_path / "packet")

    assert summary["packet_ready"] is False
    assert "missing_metadata_input_source:submission" in summary["packet_blocking_requirements"]
    assert "Packet ready: `False`" in (tmp_path / "packet/README.md").read_text(encoding="utf-8")


def test_existing_final_metadata_can_replace_missing_template(tmp_path: Path) -> None:
    make_packet_repo(tmp_path)
    (tmp_path / "submission_metadata.template.json").unlink()
    write_json(tmp_path / "submission_metadata.json", {"placeholder": "handled by final metadata validator"})

    summary = write_outputs(tmp_path, tmp_path / "packet")

    assert summary["packet_ready"] is True
    assert summary["packet_blocking_requirements"] == []
    submission_source = next(row for row in summary["metadata_input_sources"] if row["source"] == "submission")
    assert submission_source["template_exists"] is False
    assert submission_source["final_exists"] is True


def test_unknown_unresolved_requirement_blocks_input_coverage(tmp_path: Path) -> None:
    make_packet_repo(tmp_path)
    field_rows = build_field_rows(tmp_path)
    handoff_rows = [
        {
            "requirement": "final_upload:unknown_external_value",
            "status": "pending_external",
            "owner_input": "provide unknown value",
            "completion_command": "run unknown validator",
        }
    ]

    blockers = packet_blockers(tmp_path, field_rows, handoff_rows)

    assert "missing_input_coverage:final_upload:unknown_external_value" in blockers


def test_handoff_input_coverage_maps_known_requirements() -> None:
    rows = build_handoff_input_coverage_rows(
        [
            {
                "requirement": "production:cross_hardware",
                "status": "pending_external",
                "completion_command": "run second hardware",
            }
        ]
    )

    assert rows == [
        {
            "requirement": "production:cross_hardware",
            "status": "pending_external",
            "covered": "true",
            "input_files": "second_hardware_profile.json;second_hardware_energy.json",
            "field_hints": "$SecondHardwareProfile;$SecondHardwareEnergy",
            "completion_command": "run second hardware",
        }
    ]


def test_handoff_input_coverage_maps_release_consistency_requirements() -> None:
    rows = build_handoff_input_coverage_rows(
        [
            {
                "requirement": "external_consistency:citation_matches_release_links",
                "status": "pending_external",
                "completion_command": "render release metadata",
            }
        ]
    )

    assert rows[0]["covered"] == "true"
    assert rows[0]["input_files"] == "release_metadata.json"
    assert "release.github_release_url" in rows[0]["field_hints"]
    assert "citation.repository_code" in rows[0]["field_hints"]
