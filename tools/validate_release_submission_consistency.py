"""Validate consistency between release and submission metadata.

Both metadata files are filled outside the repository. This validator checks
that they point to the same public identifiers before final release and
submission artifacts are rendered.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.render_release_metadata import build_release_links, strip_doi_url, validate_release_metadata
from tools.render_submission_statements import validate_metadata as validate_submission_metadata


SCHEMA = "lite-seer-ad-release-submission-consistency-v1"
DEFAULT_OUT_DIR = Path("tables/final_external_handoff/final_metadata_consistency")
CONSISTENCY_REQUIREMENTS = [
    "github_release_url",
    "zenodo_doi",
    "hf_model_url",
    "hf_dataset_url",
    "repository_code",
    "manuscript_title",
    "citation_authors",
    "zenodo_creators",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def normalize_url(value: str) -> str:
    return str(value).strip().rstrip("/")


def normalize_person_name(value: str) -> str:
    text = " ".join(str(value).replace("\u00a0", " ").strip().split())
    if "," in text:
        family, given = [part.strip() for part in text.split(",", 1)]
        text = f"{given} {family}".strip()
    return " ".join(text.lower().split())


def citation_author_name(author: Any) -> str:
    if not isinstance(author, dict):
        return ""
    name = str(author.get("name", "")).strip()
    if name:
        return name
    given = str(author.get("given_names", "")).strip()
    family = str(author.get("family_names", "")).strip()
    return f"{given} {family}".strip()


def normalized_name_set(values: list[Any]) -> set[str]:
    names = {normalize_person_name(str(value)) for value in values}
    return {name for name in names if name}


def text_contains_identifier(text: str, identifier: str) -> bool:
    normalized_text = text.replace("\\/", "/")
    value = normalize_url(identifier)
    return value in normalized_text or f"{value}/" in normalized_text


def text_contains_doi(text: str, doi_or_url: str) -> bool:
    doi = strip_doi_url(str(doi_or_url))
    candidates = [
        doi,
        f"https://doi.org/{doi}",
        f"https://zenodo.org/doi/{doi}",
    ]
    return any(candidate in text for candidate in candidates)


def submission_availability_text(payload: dict[str, Any]) -> str:
    cover = payload.get("cover_letter") or {}
    return "\n".join(
        [
            str(payload.get("data_availability_statement", "")),
            str(payload.get("code_availability_statement", "")),
            str(cover.get("availability_sentence", "")),
        ]
    )


def status_row(requirement: str, status: str, evidence: str, detail: str) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "detail": detail,
    }


def build_rows(release_metadata: Path, submission_metadata: Path) -> list[dict[str, Any]]:
    release = read_json(release_metadata)
    submission = read_json(submission_metadata)
    release_errors = validate_release_metadata(release) if release else ["missing release metadata"]
    submission_errors = validate_submission_metadata(submission) if submission else ["missing submission metadata"]

    rows = [
        status_row(
            "metadata:release_valid",
            "pass" if not release_errors else "fail",
            str(release_metadata),
            "valid" if not release_errors else "; ".join(release_errors[:5]),
        ),
        status_row(
            "metadata:submission_valid",
            "pass" if not submission_errors else "fail",
            str(submission_metadata),
            "valid" if not submission_errors else "; ".join(submission_errors[:5]),
        ),
    ]
    if release_errors or submission_errors:
        for name in CONSISTENCY_REQUIREMENTS:
            rows.append(
                status_row(
                    f"consistency:{name}",
                    "fail",
                    f"{release_metadata}; {submission_metadata}",
                    "metadata validation failed",
                )
            )
        return rows

    links = build_release_links(release)
    availability = submission_availability_text(submission)
    repository = str((release.get("citation") or {}).get("repository_code", "")).strip()
    citation_title = str((release.get("citation") or {}).get("title", "")).strip()
    submission_title = str(((submission.get("cover_letter") or {}).get("manuscript_title", ""))).strip()
    submission_authors = normalized_name_set((submission.get("authors") or {}).get("author_list") or [])
    citation_authors = normalized_name_set(
        [citation_author_name(author) for author in (release.get("citation") or {}).get("authors") or []]
    )
    zenodo_creators = normalized_name_set(
        [creator.get("name", "") for creator in (release.get("zenodo") or {}).get("creators") or [] if isinstance(creator, dict)]
    )
    checks = [
        (
            "github_release_url",
            text_contains_identifier(availability, links["github_release_url"]),
            links["github_release_url"],
        ),
        (
            "zenodo_doi",
            text_contains_doi(availability, links["zenodo_doi"]),
            links["zenodo_doi"],
        ),
        (
            "hf_model_url",
            text_contains_identifier(availability, links["hf_model_url"]),
            links["hf_model_url"],
        ),
        (
            "hf_dataset_url",
            text_contains_identifier(availability, links["hf_dataset_url"]),
            links["hf_dataset_url"],
        ),
        (
            "repository_code",
            text_contains_identifier(availability, repository),
            repository,
        ),
        (
            "manuscript_title",
            citation_title == submission_title,
            f"release={citation_title!r}; submission={submission_title!r}",
        ),
        (
            "citation_authors",
            citation_authors == submission_authors,
            f"submission={sorted(submission_authors)}; citation={sorted(citation_authors)}",
        ),
        (
            "zenodo_creators",
            zenodo_creators == submission_authors,
            f"submission={sorted(submission_authors)}; zenodo={sorted(zenodo_creators)}",
        ),
    ]
    for name, ok, expected in checks:
        rows.append(
            status_row(
                f"consistency:{name}",
                "pass" if ok else "fail",
                f"{release_metadata}; {submission_metadata}",
                f"matched {expected}"
                if ok
                else (
                    f"author set mismatch: {expected}"
                    if name in {"citation_authors", "zenodo_creators"}
                    else f"title mismatch: {expected}"
                    if name == "manuscript_title"
                    else f"missing {expected} in submission availability text"
                ),
            )
        )
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    blockers = [row["requirement"] for row in rows if row["status"] != "pass"]
    return {
        "schema": SCHEMA,
        "final_metadata_consistent": not blockers,
        "counts": counts,
        "blocking_requirements": blockers,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["requirement", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Release/Submission Metadata Consistency",
        "",
        f"- Final metadata consistent: `{summary['final_metadata_consistent']}`",
        "",
        "| Requirement | Status | Detail |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['requirement']}` | `{row['status']}` | {row['detail']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    release_metadata: Path,
    submission_metadata: Path,
    out_dir: Path,
) -> dict[str, Any]:
    rows = build_rows(release_metadata, submission_metadata)
    summary = build_summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_final_metadata_consistency.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "final_metadata_consistency.md", summary, rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-metadata", type=Path, default=Path("release_metadata.json"))
    parser.add_argument("--submission-metadata", type=Path, default=Path("submission_metadata.json"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--fail-on-inconsistent",
        action="store_true",
        help="Exit non-zero when metadata is invalid or public identifiers disagree.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_outputs(args.release_metadata, args.submission_metadata, args.out_dir)
    print(
        f"Wrote release/submission consistency to {args.out_dir} "
        f"(final_metadata_consistent={summary['final_metadata_consistent']})"
    )
    if args.fail_on_inconsistent and not summary["final_metadata_consistent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
