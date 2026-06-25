"""Render final submission statements from checked metadata.

The template keeps author, funding, availability, and cover-letter details out
of the repository until real values are known. This renderer validates that a
metadata JSON no longer contains placeholders before writing a final statements
page, unless ``--allow-placeholders`` is explicitly used for previews.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PLACEHOLDER_RE = re.compile(r"<[^>]+>|\[[^\]]+\]|TBD|TODO|to be selected|to be checked", re.IGNORECASE)
REQUIRED_TOP_LEVEL = [
    "target_journal",
    "authors",
    "funding_statement",
    "conflict_of_interest_statement",
    "data_availability_statement",
    "code_availability_statement",
    "reproducibility_statement",
    "ethics_statement",
    "author_contributions",
    "cover_letter",
]
TARGET_JOURNAL_FIELDS = [
    "journal",
    "publisher_platform",
    "article_type",
    "template_status",
    "required_word_page_limit",
    "supplement_format",
    "guide_source_checked",
]
AUTHOR_FIELDS = ["corresponding_author", "author_list", "affiliations", "orcid_ids"]
CORRESPONDING_AUTHOR_FIELDS = ["name", "email", "affiliation"]
AVAILABILITY_MARKERS = ["GitHub Release URL", "Zenodo DOI", "Hugging Face URL", "repository URL"]
COVER_FIELDS = [
    "editor_salutation",
    "manuscript_title",
    "article_type",
    "journal",
    "availability_sentence",
    "closing_name",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def normalized_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def normalized_name_set(values: list[str]) -> set[str]:
    return {normalized_text(value) for value in values if normalized_text(value)}


def article_type_compatible(cover_value: Any, target_value: Any) -> bool:
    cover = normalized_text(cover_value)
    target = normalized_text(target_value)
    return bool(cover and target) and (cover in target or target in cover)


def walk_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.extend(walk_strings(item, child))
        return out
    if isinstance(value, list):
        out = []
        for index, item in enumerate(value):
            out.extend(walk_strings(item, f"{prefix}[{index}]"))
        return out
    return [(prefix, "" if value is None else str(value))]


def validate_metadata(payload: dict[str, Any], *, allow_placeholders: bool = False) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in payload:
            errors.append(f"missing top-level field: {key}")

    target = payload.get("target_journal") or {}
    for key in TARGET_JOURNAL_FIELDS:
        if not str(target.get(key, "")).strip():
            errors.append(f"missing target_journal.{key}")

    authors = payload.get("authors") or {}
    for key in AUTHOR_FIELDS:
        if key not in authors:
            errors.append(f"missing authors.{key}")
    corresponding = authors.get("corresponding_author") or {}
    for key in CORRESPONDING_AUTHOR_FIELDS:
        if not str(corresponding.get(key, "")).strip():
            errors.append(f"missing authors.corresponding_author.{key}")
    if not as_list(authors.get("author_list")):
        errors.append("authors.author_list must contain at least one name")
    if not as_list(authors.get("affiliations")):
        errors.append("authors.affiliations must contain at least one affiliation")

    for key in [
        "funding_statement",
        "conflict_of_interest_statement",
        "data_availability_statement",
        "code_availability_statement",
        "reproducibility_statement",
        "ethics_statement",
    ]:
        if not str(payload.get(key, "")).strip():
            errors.append(f"missing {key}")

    combined_availability = (
        str(payload.get("data_availability_statement", ""))
        + "\n"
        + str(payload.get("code_availability_statement", ""))
    )
    for marker in AVAILABILITY_MARKERS:
        if marker not in combined_availability:
            errors.append(f"availability statements must include marker: {marker}")

    cover = payload.get("cover_letter") or {}
    for key in COVER_FIELDS:
        if not str(cover.get(key, "")).strip():
            errors.append(f"missing cover_letter.{key}")

    contributions = payload.get("author_contributions") or {}
    if not isinstance(contributions, dict) or not contributions:
        errors.append("author_contributions must be a non-empty object")

    if not allow_placeholders:
        author_names = set(as_list(authors.get("author_list")))
        normalized_authors = normalized_name_set(sorted(author_names))
        corresponding_name = str(corresponding.get("name", "")).strip()
        if corresponding_name and corresponding_name not in author_names:
            errors.append("authors.corresponding_author.name must appear in authors.author_list")
        contribution_names: set[str] = set()
        if isinstance(contributions, dict):
            for role, names in contributions.items():
                role_names = as_list(names)
                if not role_names:
                    errors.append(f"author_contributions.{role} must list at least one author")
                    continue
                normalized_role_names = normalized_name_set(role_names)
                unknown = sorted(normalized_role_names - normalized_authors)
                if unknown:
                    errors.append(
                        f"author_contributions.{role} includes names not in authors.author_list: "
                        + ", ".join(unknown)
                    )
                contribution_names.update(normalized_role_names)
        missing_contribution_authors = sorted(normalized_authors - contribution_names)
        if normalized_authors and missing_contribution_authors:
            errors.append(
                "author_contributions must include every listed author at least once: "
                + ", ".join(missing_contribution_authors)
            )
        if str(cover.get("journal", "")).strip() != str(target.get("journal", "")).strip():
            errors.append("cover_letter.journal must match target_journal.journal")
        if not article_type_compatible(cover.get("article_type", ""), target.get("article_type", "")):
            errors.append("cover_letter.article_type must match target_journal.article_type")
        if str(cover.get("closing_name", "")).strip() != corresponding_name:
            errors.append("cover_letter.closing_name must match authors.corresponding_author.name")
        for key, value in walk_strings(payload):
            if PLACEHOLDER_RE.search(value):
                errors.append(f"placeholder remains in {key}: {value}")
    return errors


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def inline_list(items: list[str]) -> str:
    return "; ".join(items)


def render_submission_statements(payload: dict[str, Any]) -> str:
    target = payload["target_journal"]
    authors = payload["authors"]
    corresponding = authors["corresponding_author"]
    cover = payload["cover_letter"]
    contributions = payload["author_contributions"]

    lines = [
        "# Lite-SEER-AD Submission Statements",
        "",
        "This file was rendered from `submission_metadata.json` by `tools/render_submission_statements.py`.",
        "",
        "## Target Journal",
        "",
        f"- Journal: {target['journal']}",
        f"- Publisher/platform: {target['publisher_platform']}",
        f"- Article type: {target['article_type']}",
        f"- Template status: {target['template_status']}",
        f"- Required word/page limit: {target['required_word_page_limit']}",
        f"- Supplement format: {target['supplement_format']}",
        f"- Guide source checked: {target['guide_source_checked']}",
        "",
        "## Authors And Affiliations",
        "",
        f"- Corresponding author: {corresponding['name']}, {corresponding['email']}, {corresponding['affiliation']}",
        f"- Author list: {inline_list(as_list(authors['author_list']))}",
        f"- Affiliations: {inline_list(as_list(authors['affiliations']))}",
        f"- ORCID IDs: {inline_list(as_list(authors.get('orcid_ids')))}",
        "",
        "## Funding Statement",
        "",
        str(payload["funding_statement"]),
        "",
        "## Conflict Of Interest Statement",
        "",
        str(payload["conflict_of_interest_statement"]),
        "",
        "## Data Availability Statement",
        "",
        str(payload["data_availability_statement"]),
        "",
        "## Code Availability Statement",
        "",
        str(payload["code_availability_statement"]),
        "",
        "## Reproducibility Statement",
        "",
        str(payload["reproducibility_statement"]),
        "",
        "## Ethics Statement",
        "",
        str(payload["ethics_statement"]),
        "",
        "## Author Contributions",
        "",
    ]
    for role, names in contributions.items():
        lines.append(f"- {role}: {inline_list(as_list(names))}")
    lines.extend(
        [
            "",
            "## Cover Letter Skeleton",
            "",
            str(cover["editor_salutation"]),
            "",
            (
                f"We submit `{cover['manuscript_title']}` for consideration as "
                f"{cover['article_type']} in {cover['journal']}."
            ),
            "",
            "The manuscript presents a feature-first industrial anomaly localization pipeline centered on label-free policy/threshold selection, HN-SEV false-positive suppression, and LC-RDS budgeted local repair scheduling. We explicitly do not claim universal SOTA, do not use real anomaly labels or masks for policy selection, and downgrade CRV to repair visualization/post-hoc audit based on negative alignment evidence.",
            "",
            str(cover["availability_sentence"]),
            "",
            "Sincerely,",
            "",
            str(cover["closing_name"]),
            "",
            "## Final Pre-Upload Checks",
            "",
            "- Confirm title, abstract, contribution list, figures, supplement, and cover letter all follow the feature-first mainline.",
            "- Verify CRV is never described as a main contribution.",
            "- Verify DiffusionAD full is either absent from claims or clearly marked as optional/pending official reproduction.",
            "- Re-run `python tools/export_submission_package_readiness.py`, `python tools/export_completion_gap_matrix.py`, and `pytest -q`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("submission_metadata.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/submission_statement_placeholders.md"))
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow rendering preview files from the template even if placeholders remain.",
    )
    args = parser.parse_args()

    payload = read_json(args.input)
    errors = validate_metadata(payload, allow_placeholders=args.allow_placeholders)
    if errors:
        raise SystemExit("Submission metadata is not final:\n- " + "\n- ".join(errors))
    text = render_submission_statements(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote submission statements to {args.out}")


if __name__ == "__main__":
    main()
