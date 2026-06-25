"""Render final public-release metadata from a checked JSON file.

The release metadata file is filled only after public identifiers exist. This
renderer validates the identifiers, then writes the files consumed by the
release-readiness gate: ``release_links.json``, ``CITATION.cff``, and
``.zenodo.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.export_release_readiness import EXTERNAL_LINK_FORMATS, EXTERNAL_LINK_KEYS


SCHEMA = "lite-seer-ad-release-metadata-v1"
PLACEHOLDER_RE = re.compile(r"<[^>]+>|\[[^\]]+\]|placeholder|TBD|TODO|to be", re.IGNORECASE)
REQUIRED_TOP_LEVEL = ["schema", "release", "citation", "zenodo"]
CITATION_FIELDS = ["message", "title", "version", "date_released", "repository_code", "license", "authors"]
ZENODO_FIELDS = ["title", "upload_type", "description", "creators", "license", "keywords", "version"]
GITHUB_REPO_RE = re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+/?$")
GITHUB_RELEASE_RE = re.compile(
    r"^(?P<repo>https://github\.com/[^/\s]+/[^/\s]+)/releases/tag/(?P<tag>[^/\s]+)/?$"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def as_nonempty_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [item for item in value if str(item).strip()]


def quote_yaml(value: Any) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def strip_doi_url(value: str) -> str:
    value = value.strip()
    if value.startswith("https://doi.org/"):
        return value.removeprefix("https://doi.org/")
    if value.startswith("https://zenodo.org/doi/"):
        return value.removeprefix("https://zenodo.org/doi/")
    return value


def github_release_repo_url(value: str) -> str:
    match = GITHUB_RELEASE_RE.match(str(value).strip())
    return match.group("repo") if match else ""


def github_release_tag(value: str) -> str:
    match = GITHUB_RELEASE_RE.match(str(value).strip())
    return match.group("tag") if match else ""


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


def release_tag_matches_version(tag: str, version: str) -> bool:
    tag = str(tag).strip()
    version = str(version).strip()
    return bool(tag and version) and tag in {version, f"v{version}"}


def validate_release_metadata(payload: dict[str, Any], *, allow_placeholders: bool = False) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in payload:
            errors.append(f"missing top-level field: {key}")
    if payload.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")

    release = payload.get("release") or {}
    for key in EXTERNAL_LINK_KEYS:
        value = str(release.get(key, "")).strip()
        if not value:
            errors.append(f"missing release.{key}")
            continue
        if allow_placeholders and PLACEHOLDER_RE.search(value):
            continue
        if not EXTERNAL_LINK_FORMATS[key].match(value):
            errors.append(f"release.{key} has invalid format: {value}")

    citation = payload.get("citation") or {}
    for key in CITATION_FIELDS:
        if key not in citation or not str(citation.get(key, "")).strip():
            errors.append(f"missing citation.{key}")
    if citation.get("date_released"):
        try:
            date.fromisoformat(str(citation["date_released"]))
        except ValueError:
            errors.append("citation.date_released must use YYYY-MM-DD")
    repository = str(citation.get("repository_code", "")).strip()
    if repository and not (allow_placeholders and PLACEHOLDER_RE.search(repository)):
        if not GITHUB_REPO_RE.match(repository):
            errors.append(f"citation.repository_code has invalid GitHub URL: {repository}")
    github_release_url = str(release.get("github_release_url", "")).strip()
    if github_release_url and repository and not allow_placeholders:
        release_repo = github_release_repo_url(github_release_url)
        if release_repo and release_repo.rstrip("/") != repository.rstrip("/"):
            errors.append(
                "release.github_release_url repository must match citation.repository_code: "
                f"{release_repo} != {repository}"
            )
        release_tag = github_release_tag(github_release_url)
        version = str(citation.get("version", "")).strip()
        if release_tag and version and not release_tag_matches_version(release_tag, version):
            errors.append(
                "release.github_release_url tag must match citation.version "
                f"or v-prefixed version: {release_tag} != {version}"
            )
    authors = as_nonempty_list(citation.get("authors"))
    if not authors:
        errors.append("citation.authors must contain at least one author")
    for index, author in enumerate(authors):
        if not isinstance(author, dict):
            errors.append(f"citation.authors[{index}] must be an object")
            continue
        if not str(author.get("name") or author.get("family_names") or "").strip():
            errors.append(f"citation.authors[{index}] needs name or family_names")

    zenodo = payload.get("zenodo") or {}
    for key in ZENODO_FIELDS:
        if key not in zenodo or not str(zenodo.get(key, "")).strip():
            errors.append(f"missing zenodo.{key}")
    creators = as_nonempty_list(zenodo.get("creators"))
    if not creators:
        errors.append("zenodo.creators must contain at least one creator")
    for index, creator in enumerate(creators):
        if not isinstance(creator, dict):
            errors.append(f"zenodo.creators[{index}] must be an object")
            continue
        if not str(creator.get("name", "")).strip():
            errors.append(f"zenodo.creators[{index}].name is required")
    if not as_nonempty_list(zenodo.get("keywords")):
        errors.append("zenodo.keywords must contain at least one keyword")

    citation_author_names = normalized_name_set([citation_author_name(author) for author in authors])
    zenodo_creator_names = normalized_name_set(
        [creator.get("name", "") for creator in creators if isinstance(creator, dict)]
    )
    if citation_author_names and zenodo_creator_names and citation_author_names != zenodo_creator_names:
        errors.append(
            "citation.authors and zenodo.creators must list the same people: "
            f"citation={sorted(citation_author_names)}; zenodo={sorted(zenodo_creator_names)}"
        )

    if citation.get("title") and zenodo.get("title") and citation["title"] != zenodo["title"]:
        errors.append("citation.title and zenodo.title must match")
    if citation.get("version") and zenodo.get("version") and citation["version"] != zenodo["version"]:
        errors.append("citation.version and zenodo.version must match")

    if not allow_placeholders:
        for key, value in walk_strings(payload):
            if PLACEHOLDER_RE.search(value):
                errors.append(f"placeholder remains in {key}: {value}")
    return errors


def build_release_links(payload: dict[str, Any]) -> dict[str, str]:
    release = payload["release"]
    return {key: str(release[key]).strip() for key in EXTERNAL_LINK_KEYS}


def render_citation_cff(payload: dict[str, Any]) -> str:
    citation = payload["citation"]
    links = build_release_links(payload)
    lines = [
        "cff-version: 1.2.0",
        f"message: {quote_yaml(citation['message'])}",
        f"title: {quote_yaml(citation['title'])}",
        f"version: {quote_yaml(citation['version'])}",
        f"date-released: {quote_yaml(citation['date_released'])}",
        "authors:",
    ]
    for author in citation["authors"]:
        lines.append("  -")
        if author.get("name"):
            lines.append(f"    name: {quote_yaml(author['name'])}")
        if author.get("family_names"):
            lines.append(f"    family-names: {quote_yaml(author['family_names'])}")
        if author.get("given_names"):
            lines.append(f"    given-names: {quote_yaml(author['given_names'])}")
        if author.get("orcid"):
            lines.append(f"    orcid: {quote_yaml(author['orcid'])}")
    lines.extend(
        [
            f"repository-code: {quote_yaml(citation['repository_code'])}",
            f"url: {quote_yaml(links['github_release_url'])}",
            f"doi: {quote_yaml(strip_doi_url(links['zenodo_doi']))}",
            f"license: {quote_yaml(citation['license'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_zenodo_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    zenodo = dict(payload["zenodo"])
    links = build_release_links(payload)
    related = [
        {
            "identifier": links["github_release_url"],
            "relation": "isSupplementTo",
            "scheme": "url",
        },
        {
            "identifier": links["hf_model_url"],
            "relation": "isSupplementTo",
            "scheme": "url",
        },
        {
            "identifier": links["hf_dataset_url"],
            "relation": "isSupplementTo",
            "scheme": "url",
        },
    ]
    existing = zenodo.get("related_identifiers") or []
    if isinstance(existing, list):
        related = existing + related
    zenodo["related_identifiers"] = related
    return zenodo


def render_identifier_note(payload: dict[str, Any]) -> str:
    links = build_release_links(payload)
    lines = [
        "# Public Release Identifiers",
        "",
        "This file was rendered from `release_metadata.json` by `tools/render_release_metadata.py`.",
        "",
        f"- GitHub Release: {links['github_release_url']}",
        f"- Zenodo DOI: {links['zenodo_doi']}",
        f"- Hugging Face model repository: {links['hf_model_url']}",
        f"- Hugging Face dataset repository: {links['hf_dataset_url']}",
    ]
    return "\n".join(lines) + "\n"


def write_release_metadata_outputs(payload: dict[str, Any], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    links = build_release_links(payload)
    (root / "release_links.json").write_text(
        json.dumps(links, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(render_citation_cff(payload), encoding="utf-8")
    (root / ".zenodo.json").write_text(
        json.dumps(build_zenodo_metadata(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    note = root / "docs/public_release_identifiers.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(render_identifier_note(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("release_metadata.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow validating a preview/template file without writing outputs.",
    )
    args = parser.parse_args()

    payload = read_json(args.input)
    errors = validate_release_metadata(payload, allow_placeholders=args.allow_placeholders)
    if errors:
        raise SystemExit("Release metadata is not final:\n- " + "\n- ".join(errors))
    if args.allow_placeholders:
        print(f"Validated release metadata preview from {args.input}")
        return
    write_release_metadata_outputs(payload, args.root)
    print(f"Wrote release metadata outputs under {args.root}")


if __name__ == "__main__":
    main()
