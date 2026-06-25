from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate_release_submission_consistency import build_rows, main, write_outputs


def release_metadata(*, citation_author: str = "Ada Lovelace", zenodo_creator: str = "Lovelace, Ada") -> dict:
    if "," in citation_author:
        family, given = [part.strip() for part in citation_author.split(",", 1)]
        citation_authors = [{"family_names": family, "given_names": given}]
    else:
        citation_authors = [{"name": citation_author}]
    return {
        "schema": "lite-seer-ad-release-metadata-v1",
        "release": {
            "github_release_url": "https://github.com/org/lite-seer-ad/releases/tag/v1.0-paper",
            "zenodo_doi": "https://doi.org/10.5281/zenodo.1234567",
            "hf_model_url": "https://huggingface.co/org/lite-seer-ad-models",
            "hf_dataset_url": "https://huggingface.co/datasets/org/lite-seer-ad-artifacts",
        },
        "citation": {
            "message": "If you use Lite-SEER-AD, please cite the paper package and release.",
            "title": "Lite-SEER-AD",
            "version": "1.0-paper",
            "date_released": "2026-06-21",
            "repository_code": "https://github.com/org/lite-seer-ad",
            "license": "MIT",
            "authors": citation_authors,
        },
        "zenodo": {
            "title": "Lite-SEER-AD",
            "upload_type": "software",
            "description": "Claim-bounded paper package.",
            "creators": [{"name": zenodo_creator}],
            "license": "mit",
            "keywords": ["industrial anomaly detection"],
            "version": "1.0-paper",
        },
    }


def submission_metadata(*, mismatch: bool = False, title: str = "Lite-SEER-AD") -> dict:
    hf_model = "https://huggingface.co/org/other-model" if mismatch else "https://huggingface.co/org/lite-seer-ad-models"
    return {
        "target_journal": {
            "journal": "Pattern Recognition",
            "publisher_platform": "Elsevier / ScienceDirect",
            "article_type": "Full Length Article",
            "template_status": "Converted",
            "required_word_page_limit": "Guide checked",
            "supplement_format": "Separate supplement",
            "guide_source_checked": "https://www.journals.elsevier.com/pattern-recognition",
        },
        "authors": {
            "corresponding_author": {
                "name": "Ada Lovelace",
                "email": "ada@example.edu",
                "affiliation": "Example University",
            },
            "author_list": ["Ada Lovelace"],
            "affiliations": ["Example University, City, Country"],
            "orcid_ids": ["Ada Lovelace: 0000-0000-0000-0000"],
        },
        "funding_statement": "This research received no external funding.",
        "conflict_of_interest_statement": "The authors declare no competing interests.",
        "data_availability_statement": (
            "GitHub Release URL: https://github.com/org/lite-seer-ad/releases/tag/v1.0-paper. "
            "Zenodo DOI: 10.5281/zenodo.1234567. "
            f"Hugging Face URL: {hf_model}; https://huggingface.co/datasets/org/lite-seer-ad-artifacts."
        ),
        "code_availability_statement": (
            "repository URL: https://github.com/org/lite-seer-ad. "
            "The exact release commit is abcdef123456."
        ),
        "reproducibility_statement": "All retained quantitative claims are tied to fixed configs and hashes.",
        "ethics_statement": "This study does not involve human participants, personal data, or animal experiments.",
        "author_contributions": {"Conceptualization": ["Ada Lovelace"]},
        "cover_letter": {
            "editor_salutation": "Dear Editor,",
            "manuscript_title": title,
            "article_type": "Full Length Article",
            "journal": "Pattern Recognition",
            "availability_sentence": "All retained claims are backed by final public artifact links.",
            "closing_name": "Ada Lovelace",
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_consistency_rows_pass_when_public_identifiers_match(tmp_path: Path) -> None:
    release_path = tmp_path / "release_metadata.json"
    submission_path = tmp_path / "submission_metadata.json"
    write_json(release_path, release_metadata())
    write_json(submission_path, submission_metadata())

    rows = build_rows(release_path, submission_path)

    assert all(row["status"] == "pass" for row in rows)


def test_consistency_rows_fail_for_mismatched_hf_model_url(tmp_path: Path) -> None:
    release_path = tmp_path / "release_metadata.json"
    submission_path = tmp_path / "submission_metadata.json"
    write_json(release_path, release_metadata())
    write_json(submission_path, submission_metadata(mismatch=True))

    rows = build_rows(release_path, submission_path)
    by_requirement = {row["requirement"]: row for row in rows}

    assert by_requirement["consistency:hf_model_url"]["status"] == "fail"
    assert by_requirement["consistency:hf_dataset_url"]["status"] == "pass"


def test_consistency_rows_fail_for_mismatched_release_authors(tmp_path: Path) -> None:
    release_path = tmp_path / "release_metadata.json"
    submission_path = tmp_path / "submission_metadata.json"
    write_json(release_path, release_metadata(citation_author="Grace Hopper", zenodo_creator="Hopper, Grace"))
    write_json(submission_path, submission_metadata())

    rows = build_rows(release_path, submission_path)
    by_requirement = {row["requirement"]: row for row in rows}

    assert by_requirement["consistency:citation_authors"]["status"] == "fail"
    assert by_requirement["consistency:zenodo_creators"]["status"] == "fail"


def test_consistency_rows_fail_for_mismatched_manuscript_title(tmp_path: Path) -> None:
    release_path = tmp_path / "release_metadata.json"
    submission_path = tmp_path / "submission_metadata.json"
    write_json(release_path, release_metadata())
    write_json(submission_path, submission_metadata(title="Different Manuscript Title"))

    rows = build_rows(release_path, submission_path)
    by_requirement = {row["requirement"]: row for row in rows}

    assert by_requirement["consistency:manuscript_title"]["status"] == "fail"
    assert by_requirement["consistency:citation_authors"]["status"] == "pass"


def test_consistency_rows_fail_early_for_internally_invalid_release_authors(tmp_path: Path) -> None:
    release_path = tmp_path / "release_metadata.json"
    submission_path = tmp_path / "submission_metadata.json"
    write_json(release_path, release_metadata(zenodo_creator="Hopper, Grace"))
    write_json(submission_path, submission_metadata())

    rows = build_rows(release_path, submission_path)
    by_requirement = {row["requirement"]: row for row in rows}

    assert by_requirement["metadata:release_valid"]["status"] == "fail"
    assert by_requirement["consistency:citation_authors"]["detail"] == "metadata validation failed"
    assert by_requirement["consistency:zenodo_creators"]["status"] == "fail"


def test_write_outputs_creates_consistency_artifacts(tmp_path: Path) -> None:
    release_path = tmp_path / "release_metadata.json"
    submission_path = tmp_path / "submission_metadata.json"
    write_json(release_path, release_metadata())
    write_json(submission_path, submission_metadata())

    summary = write_outputs(release_path, submission_path, tmp_path / "out")

    assert summary["final_metadata_consistent"] is True
    assert (tmp_path / "out/summary.json").is_file()
    assert (tmp_path / "out/table_final_metadata_consistency.csv").is_file()
    assert (tmp_path / "out/final_metadata_consistency.md").is_file()


def test_missing_inputs_are_reported_as_inconsistent(tmp_path: Path) -> None:
    summary = write_outputs(
        tmp_path / "release_metadata.json",
        tmp_path / "submission_metadata.json",
        tmp_path / "out",
    )

    assert summary["final_metadata_consistent"] is False
    assert "metadata:release_valid" in summary["blocking_requirements"]
    assert "consistency:manuscript_title" in summary["blocking_requirements"]
    assert "consistency:citation_authors" in summary["blocking_requirements"]
    assert "consistency:zenodo_creators" in summary["blocking_requirements"]


def test_cli_can_fail_on_inconsistent_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["validate_release_submission_consistency.py", "--fail-on-inconsistent"],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
