from __future__ import annotations

import json
from pathlib import Path

from tools.export_release_readiness import check_external_links
from tools.render_release_metadata import (
    render_citation_cff,
    validate_release_metadata,
    write_release_metadata_outputs,
)


def final_release_metadata() -> dict:
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
            "authors": [
                {
                    "family_names": "Lovelace",
                    "given_names": "Ada",
                    "orcid": "https://orcid.org/0000-0000-0000-0000",
                }
            ],
        },
        "zenodo": {
            "title": "Lite-SEER-AD",
            "upload_type": "software",
            "description": "Claim-bounded paper package.",
            "creators": [
                {
                    "name": "Lovelace, Ada",
                    "affiliation": "Example University",
                    "orcid": "0000-0000-0000-0000",
                }
            ],
            "license": "mit",
            "keywords": ["industrial anomaly detection", "label-free threshold selection"],
            "version": "1.0-paper",
        },
    }


def test_validate_release_metadata_rejects_placeholders() -> None:
    payload = final_release_metadata()
    payload["release"]["github_release_url"] = "https://github.com/<owner>/<repo>/releases/tag/v1.0-paper"

    errors = validate_release_metadata(payload)

    assert any("invalid format" in error or "placeholder remains" in error for error in errors)


def test_validate_release_metadata_rejects_mismatched_release_repository() -> None:
    payload = final_release_metadata()
    payload["release"]["github_release_url"] = "https://github.com/other/repo/releases/tag/v1.0-paper"

    errors = validate_release_metadata(payload)

    assert any("repository must match citation.repository_code" in error for error in errors)


def test_validate_release_metadata_rejects_mismatched_release_tag() -> None:
    payload = final_release_metadata()
    payload["release"]["github_release_url"] = "https://github.com/org/lite-seer-ad/releases/tag/v2.0"

    errors = validate_release_metadata(payload)

    assert any("tag must match citation.version" in error for error in errors)


def test_validate_release_metadata_rejects_mismatched_zenodo_creators() -> None:
    payload = final_release_metadata()
    payload["zenodo"]["creators"] = [{"name": "Hopper, Grace"}]

    errors = validate_release_metadata(payload)

    assert any("citation.authors and zenodo.creators must list the same people" in error for error in errors)


def test_template_can_be_validated_only_when_placeholders_allowed() -> None:
    template = json.loads(Path("release_metadata.template.json").read_text(encoding="utf-8"))

    assert validate_release_metadata(template)
    assert not validate_release_metadata(template, allow_placeholders=True)


def test_render_citation_cff_contains_public_identifiers() -> None:
    text = render_citation_cff(final_release_metadata())

    assert 'repository-code: "https://github.com/org/lite-seer-ad"' in text
    assert 'url: "https://github.com/org/lite-seer-ad/releases/tag/v1.0-paper"' in text
    assert 'doi: "10.5281/zenodo.1234567"' in text


def test_rendered_outputs_clear_release_readiness_external_rows(tmp_path: Path) -> None:
    payload = final_release_metadata()
    write_release_metadata_outputs(payload, tmp_path)

    rows = check_external_links(tmp_path)
    statuses = {row["requirement"]: row["status"] for row in rows}

    assert statuses["external_link:github_release_url"] == "pass"
    assert statuses["external_link:zenodo_doi"] == "pass"
    assert statuses["external_link:hf_model_url"] == "pass"
    assert statuses["external_link:hf_dataset_url"] == "pass"
    assert statuses["no_placeholder:citation_repository"] == "pass"
    assert statuses["no_placeholder:zenodo_metadata"] == "pass"
    assert (tmp_path / "docs/public_release_identifiers.md").is_file()
