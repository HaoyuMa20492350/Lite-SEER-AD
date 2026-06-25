from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "baselines" / "official_sources.json"
REQUIRED_PROVENANCE_FIELDS = (
    "method",
    "dataset",
    "category",
    "source_kind",
    "source_repository",
    "source_commit",
    "official_implementation",
    "execution_command",
    "environment",
    "checkpoint_source",
)


def load_official_source_manifest(
    path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"{manifest_path} contains no official baseline sources")
    for method, source in sources.items():
        for key in (
            "display_name",
            "source_kind",
            "repository",
            "commit",
            "reference_key",
            "declared_datasets",
            "entrypoint",
        ):
            if not source.get(key):
                raise ValueError(f"{method} source is missing '{key}'")
        commit = str(source["commit"])
        if len(commit) != 40 or any(
            char not in "0123456789abcdef" for char in commit.lower()
        ):
            raise ValueError(f"{method} has an invalid pinned commit: {commit}")
    return payload


def validate_official_provenance(
    provenance: dict[str, Any],
    source: dict[str, Any],
    *,
    method: str,
    dataset: str,
    category: str,
) -> list[str]:
    errors = []
    for field in REQUIRED_PROVENANCE_FIELDS:
        value = provenance.get(field)
        if value is None or value == "":
            errors.append(f"missing provenance field: {field}")
    expected = {
        "method": method,
        "dataset": dataset,
        "category": category,
        "source_kind": source["source_kind"],
        "source_repository": source["repository"],
        "source_commit": source["commit"],
    }
    for key, value in expected.items():
        if str(provenance.get(key, "")) != str(value):
            errors.append(
                f"{key} mismatch: expected {value!r}, got "
                f"{provenance.get(key)!r}"
            )
    expected_official = source["source_kind"] == "author_official"
    if bool(provenance.get("official_implementation")) != expected_official:
        errors.append(
            "official_implementation does not match source_kind "
            f"({source['source_kind']})"
        )
    if provenance.get("paper_eligible_full_training") is False:
        errors.append(
            "artifact is explicitly marked as incomplete for paper use"
        )
    return errors
