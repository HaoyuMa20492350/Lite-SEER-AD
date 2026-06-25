from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.export_submission_package_readiness import check_final_placeholders
from tools.render_submission_statements import (
    render_submission_statements,
    validate_metadata,
)


def final_metadata() -> dict:
    return {
        "target_journal": {
            "journal": "Pattern Recognition",
            "publisher_platform": "Elsevier / ScienceDirect",
            "article_type": "Full Length Article",
            "template_status": "Elsevier article template selected",
            "required_word_page_limit": "regular article page limits checked",
            "supplement_format": "separate supplementary material",
            "guide_source_checked": "https://www.journals.elsevier.com/pattern-recognition",
        },
        "authors": {
            "corresponding_author": {
                "name": "Ada Lovelace",
                "email": "ada@example.edu",
                "affiliation": "Example University",
            },
            "author_list": ["Ada Lovelace", "Grace Hopper"],
            "affiliations": ["Example University, Computing Department, City, Country"],
            "orcid_ids": ["Ada Lovelace: 0000-0000-0000-0000"],
        },
        "funding_statement": "This research received no external funding.",
        "conflict_of_interest_statement": "The authors declare no competing interests.",
        "data_availability_statement": "GitHub Release URL: https://github.com/org/repo/releases/tag/v1.0-paper. Zenodo DOI: 10.5281/zenodo.1234567. Hugging Face URL: https://huggingface.co/datasets/org/data.",
        "code_availability_statement": "repository URL: https://github.com/org/repo. The source code is available under the MIT license. The exact release commit is abcdef123456.",
        "reproducibility_statement": "All retained quantitative claims are tied to fixed configs and hashes.",
        "ethics_statement": "This study does not involve human participants, personal data, or animal experiments.",
        "author_contributions": {
            "Conceptualization": ["Ada Lovelace"],
            "Methodology": ["Ada Lovelace"],
            "Software": ["Grace Hopper"],
        },
        "cover_letter": {
            "editor_salutation": "Dear Editor,",
            "manuscript_title": "Lite-SEER-AD",
            "article_type": "Full Length Article",
            "journal": "Pattern Recognition",
            "availability_sentence": "All retained claims are backed by final public artifact links.",
            "closing_name": "Ada Lovelace",
        },
    }


def test_validate_metadata_rejects_placeholders() -> None:
    payload = final_metadata()
    payload["authors"]["author_list"] = ["<ordered names>"]

    errors = validate_metadata(payload)

    assert any("placeholder remains" in error for error in errors)


def test_validate_metadata_rejects_corresponding_author_missing_from_author_list() -> None:
    payload = final_metadata()
    payload["authors"]["author_list"] = ["Grace Hopper"]

    errors = validate_metadata(payload)

    assert "authors.corresponding_author.name must appear in authors.author_list" in errors


def test_validate_metadata_rejects_cover_letter_journal_mismatch() -> None:
    payload = final_metadata()
    payload["cover_letter"]["journal"] = "Different Journal"

    errors = validate_metadata(payload)

    assert "cover_letter.journal must match target_journal.journal" in errors


def test_validate_metadata_rejects_cover_letter_closing_name_mismatch() -> None:
    payload = final_metadata()
    payload["cover_letter"]["closing_name"] = "Grace Hopper"

    errors = validate_metadata(payload)

    assert "cover_letter.closing_name must match authors.corresponding_author.name" in errors


def test_validate_metadata_rejects_unknown_author_contribution_name() -> None:
    payload = final_metadata()
    payload["author_contributions"]["Software"] = ["Unknown Person"]

    errors = validate_metadata(payload)

    assert any("author_contributions.Software includes names not in authors.author_list" in error for error in errors)


def test_validate_metadata_requires_every_author_in_contributions() -> None:
    payload = final_metadata()
    payload["author_contributions"] = {"Conceptualization": ["Ada Lovelace"]}

    errors = validate_metadata(payload)

    assert any("author_contributions must include every listed author at least once" in error for error in errors)


def test_rendered_statements_clear_final_placeholder_sections(tmp_path: Path) -> None:
    payload = final_metadata()
    text = render_submission_statements(payload)
    target = tmp_path / "docs/submission_statement_placeholders.md"
    target.parent.mkdir(parents=True)
    target.write_text(text, encoding="utf-8")

    rows = check_final_placeholders(tmp_path)

    final_rows = {row["requirement"]: row for row in rows}
    assert final_rows["final_upload:authors_affiliations"]["status"] == "pass"
    assert final_rows["final_upload:funding_conflicts"]["status"] == "pass"
    assert final_rows["final_upload:availability_links"]["status"] == "pass"
    assert final_rows["final_upload:cover_letter"]["status"] == "pass"


def test_template_can_be_rendered_only_when_placeholders_allowed(tmp_path: Path) -> None:
    template = json.loads(Path("submission_metadata.template.json").read_text(encoding="utf-8"))

    assert validate_metadata(template)
    assert not validate_metadata(template, allow_placeholders=True)
