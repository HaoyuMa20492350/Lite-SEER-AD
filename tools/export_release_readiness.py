"""Export public reproduction release-readiness artifacts.

The release gate separates local reproducibility readiness from external
publication. Local manifests can be complete before GitHub Release, Zenodo DOI,
and Hugging Face URLs exist; the final public-reproduction gate remains closed
until those external identifiers are filled in.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PLACEHOLDER_RE = re.compile(
    r"placeholder|Lite-SEER-AD Authors|<[^>]+>|\[[^\]\n]{1,80}\]|to be|todo|TBD",
    re.IGNORECASE,
)
REQUIRED_LOCAL_FILES = [
    "README.md",
    "REPRODUCE.md",
    "MODEL_CARD.md",
    "DATASETS.md",
    "CITATION.cff",
    ".zenodo.json",
    "release_links.template.json",
    "release_metadata.template.json",
    "submission_metadata.template.json",
    "artifacts/manifest.json",
    "artifacts/predictions_manifest.json",
    "artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json",
    "tables/claim_traceability/summary.json",
    "tables/claim_traceability/table_claim_traceability.csv",
    "paper/manuscript.md",
    "tools/export_strict_threshold_paper_artifacts.py",
    "tools/export_external_baseline_comparison.py",
    "scripts/release/build_artifact_manifest.py",
    "scripts/release/build_prediction_array_manifest.py",
    "scripts/release/export_fixed_threshold_bundle.py",
]
EXTERNAL_LINK_KEYS = [
    "github_release_url",
    "zenodo_doi",
    "hf_model_url",
    "hf_dataset_url",
]
EXTERNAL_LINK_FORMATS = {
    "github_release_url": re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+/releases/tag/[^/\s]+/?$"),
    "zenodo_doi": re.compile(
        r"^(?:https://doi\.org/)?10\.5281/zenodo\.\d+$|^https://zenodo\.org/doi/10\.5281/zenodo\.\d+$"
    ),
    "hf_model_url": re.compile(r"^https://huggingface\.co/(?!datasets/)[^/\s]+/[^/\s]+/?$"),
    "hf_dataset_url": re.compile(r"^https://huggingface\.co/datasets/[^/\s]+/[^/\s]+/?$"),
}
EXTERNAL_REQUIREMENT_PREFIXES = ("external_link:", "no_placeholder:", "external_consistency:")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def has_valid_sha(entry: dict[str, Any]) -> bool:
    sha = str(entry.get("sha256", ""))
    return len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha.lower())


def check_local_files(root: Path) -> list[dict[str, Any]]:
    rows = []
    manifest = read_json(root / "artifacts/manifest.json")
    manifest_paths = {entry.get("path") for entry in manifest.get("files", [])}
    for relative in REQUIRED_LOCAL_FILES:
        path = root / relative
        rows.append(
            {
                "requirement": f"local_file:{relative}",
                "status": "pass" if path.is_file() else "fail",
                "evidence": relative,
                "detail": "present" if path.is_file() else "missing",
                "in_manifest": relative in manifest_paths,
            }
        )
    return rows


def check_manifests(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    artifact_manifest = read_json(root / "artifacts/manifest.json")
    artifact_files = artifact_manifest.get("files", []) or []
    artifact_ok = (
        artifact_manifest.get("schema") == "lite-seer-ad-artifact-manifest-v1"
        and len(artifact_files) > 0
        and all(has_valid_sha(entry) and int(entry.get("bytes", 0)) > 0 for entry in artifact_files)
    )
    rows.append(
        {
            "requirement": "sha256_artifact_manifest",
            "status": "pass" if artifact_ok else "fail",
            "evidence": "artifacts/manifest.json",
            "detail": f"files={len(artifact_files)}",
            "in_manifest": True,
        }
    )

    predictions = read_json(root / "artifacts/predictions_manifest.json")
    entries = predictions.get("entries", []) or []
    prediction_ok = (
        predictions.get("all_predictions_present") is True
        and predictions.get("all_threshold_policies_present") is True
        and len(entries) >= 99
        and all(str(entry.get("prediction_sha256", "")).strip() for entry in entries)
    )
    rows.append(
        {
            "requirement": "selected_prediction_arrays_and_thresholds",
            "status": "pass" if prediction_ok else "fail",
            "evidence": "artifacts/predictions_manifest.json",
            "detail": f"entries={len(entries)}",
            "in_manifest": True,
        }
    )

    thresholds = read_json(root / "artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json")
    threshold_ok = (
        thresholds.get("schema") == "synthetic_normal_fixed_threshold_v1_bundle"
        and thresholds.get("policy_count", 0) >= 33
        and thresholds.get("uses_real_anomaly_labels") is False
        and thresholds.get("uses_real_anomaly_masks") is False
    )
    rows.append(
        {
            "requirement": "fixed_threshold_bundle_label_free",
            "status": "pass" if threshold_ok else "fail",
            "evidence": "artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json",
            "detail": f"policies={thresholds.get('policy_count', 0)}",
            "in_manifest": True,
        }
    )
    return rows


def load_external_links(root: Path) -> dict[str, str]:
    path = root / "release_links.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return {key: str(payload.get(key, "")).strip() for key in EXTERNAL_LINK_KEYS}


def strip_doi_url(value: str) -> str:
    text = str(value).strip()
    for prefix in ("https://doi.org/", "https://zenodo.org/doi/"):
        if text.startswith(prefix):
            return text.removeprefix(prefix)
    return text


def text_contains_identifier(text: str, identifier: str) -> bool:
    value = str(identifier).strip().rstrip("/")
    if not value:
        return False
    normalized_text = text.replace("\\/", "/")
    return value in normalized_text or f"{value}/" in normalized_text


def text_contains_doi(text: str, doi_or_url: str) -> bool:
    doi = strip_doi_url(doi_or_url)
    candidates = [doi, f"https://doi.org/{doi}", f"https://zenodo.org/doi/{doi}"]
    return any(candidate in text for candidate in candidates)


def links_are_final(links: dict[str, str]) -> bool:
    return all(
        (value := links.get(key, ""))
        and EXTERNAL_LINK_FORMATS[key].match(value)
        and not PLACEHOLDER_RE.search(value)
        for key in EXTERNAL_LINK_KEYS
    )


def external_consistency_row(requirement: str, ok: bool, evidence: str, detail: str) -> dict[str, Any]:
    return {
        "requirement": f"external_consistency:{requirement}",
        "status": "pass" if ok else "pending_external",
        "evidence": evidence,
        "detail": detail,
        "in_manifest": False,
    }


def check_rendered_identifier_consistency(root: Path, links: dict[str, str]) -> list[dict[str, Any]]:
    citation = read_text(root / "CITATION.cff")
    zenodo = read_text(root / ".zenodo.json")
    if not links_are_final(links):
        return [
            external_consistency_row(
                "citation_matches_release_links",
                False,
                "CITATION.cff; release_links.json",
                "release links are missing or not final",
            ),
            external_consistency_row(
                "zenodo_related_identifiers_match_release_links",
                False,
                ".zenodo.json; release_links.json",
                "release links are missing or not final",
            ),
        ]

    citation_ok = text_contains_identifier(citation, links["github_release_url"]) and text_contains_doi(
        citation,
        links["zenodo_doi"],
    )
    zenodo_ok = all(
        text_contains_identifier(zenodo, links[key])
        for key in ["github_release_url", "hf_model_url", "hf_dataset_url"]
    )
    return [
        external_consistency_row(
            "citation_matches_release_links",
            citation_ok,
            "CITATION.cff; release_links.json",
            "CITATION.cff contains GitHub release URL and Zenodo DOI"
            if citation_ok
            else "CITATION.cff does not match release_links.json",
        ),
        external_consistency_row(
            "zenodo_related_identifiers_match_release_links",
            zenodo_ok,
            ".zenodo.json; release_links.json",
            ".zenodo.json contains GitHub/Hugging Face related identifiers"
            if zenodo_ok
            else ".zenodo.json does not match release_links.json",
        ),
    ]


def check_external_links(root: Path) -> list[dict[str, Any]]:
    links = load_external_links(root)
    rows = []
    citation = read_text(root / "CITATION.cff")
    model_card = read_text(root / "MODEL_CARD.md")
    dataset_card = read_text(root / "DATASETS.md")
    zenodo = read_text(root / ".zenodo.json")
    for key in EXTERNAL_LINK_KEYS:
        value = links.get(key, "")
        format_ok = bool(value) and bool(EXTERNAL_LINK_FORMATS[key].match(value))
        ok = format_ok and not PLACEHOLDER_RE.search(value)
        detail = value or "missing"
        if value and not ok:
            detail = f"invalid format: {value}"
        rows.append(
            {
                "requirement": f"external_link:{key}",
                "status": "pass" if ok else "pending_external",
                "evidence": "release_links.json",
                "detail": detail,
                "in_manifest": False,
            }
        )
    placeholder_sources = {
        "citation_repository": citation,
        "model_card_links": model_card,
        "dataset_card_links": dataset_card,
        "zenodo_metadata": zenodo,
    }
    for name, text in placeholder_sources.items():
        rows.append(
            {
                "requirement": f"no_placeholder:{name}",
                "status": "pass" if not PLACEHOLDER_RE.search(text) else "pending_external",
                "evidence": name,
                "detail": "placeholder-free" if not PLACEHOLDER_RE.search(text) else "contains placeholder or missing final link",
                "in_manifest": name != "citation_repository",
            }
        )
    rows.extend(check_rendered_identifier_consistency(root, links))
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    local_rows = [row for row in rows if not row["requirement"].startswith(EXTERNAL_REQUIREMENT_PREFIXES)]
    local_ready = all(row["status"] == "pass" for row in local_rows)
    external_rows = [row for row in rows if row not in local_rows]
    external_ready = all(row["status"] == "pass" for row in external_rows)
    return {
        "schema": "lite-seer-ad-release-readiness-v1",
        "local_artifact_ready": local_ready,
        "external_publication_ready": external_ready,
        "release_gate_passed": local_ready and external_ready,
        "release_gate_reason": (
            "Local artifacts are ready, but external GitHub/Zenodo/Hugging Face identifiers are missing."
            if local_ready and not external_ready
            else "Release gate passed."
            if local_ready and external_ready
            else "Local release artifacts are incomplete."
        ),
        "counts": counts,
        "required_external_links": EXTERNAL_LINK_KEYS,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["requirement", "status", "evidence", "detail", "in_manifest"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_release_notes(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Lite-SEER-AD v1.0-paper Release Notes",
        "",
        "This release is the claim-bounded paper package for Feature-first Lite-SEER-AD.",
        "",
        "Included local artifacts:",
        "",
        "- SHA256 artifact manifest: `artifacts/manifest.json`",
        "- Selected prediction-array manifest: `artifacts/predictions_manifest.json`",
        "- Fixed threshold bundle: `artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json`",
        "- Main paper tables and claim-boundary documents",
        "- Reproduction scripts for strict threshold tables, external baselines, manifests, and statistics",
        "",
        "Claim boundaries:",
        "",
        "- No universal SOTA claim.",
        "- No diffusion-first detector claim.",
        "- CRV is visualization/post-hoc audit only.",
        "- DiffusionAD full official reproduction is optional and not replaced by smoke runs.",
        "",
        "External identifiers to fill before public release:",
        "",
    ]
    for key in summary["required_external_links"]:
        lines.append(f"- `{key}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_zenodo_checklist(path: Path) -> None:
    lines = [
        "# Zenodo Deposition Checklist",
        "",
        "- Upload source archive and generated artifact manifest.",
        "- Confirm `.zenodo.json` has real creators, license, keywords, and version.",
        "- Reserve or publish DOI.",
        "- Add DOI and public repository URLs to `release_metadata.json`.",
        "- Run `python tools/render_release_metadata.py --input release_metadata.json`.",
        "- Re-run `python tools/export_release_readiness.py`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hf_card_patch(path: Path) -> None:
    lines = [
        "# Hugging Face Card Patch Notes",
        "",
        "Add final URLs before release:",
        "",
        "- `github_release_url`",
        "- `zenodo_doi`",
        "- `hf_model_url`",
        "- `hf_dataset_url`",
        "",
        "Required card boundary text:",
        "",
        "Lite-SEER-AD is feature-first and label-free for policy/threshold selection. "
        "Regional HN-SEV/LC-RDS/CRV artifacts are audited separately; CRV is not "
        "semantic repair evidence and diffusion is not the main detector.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(root: Path, out_dir: Path) -> dict[str, Any]:
    rows = check_local_files(root) + check_manifests(root) + check_external_links(root)
    summary = build_summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_release_readiness.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_release_notes(out_dir / "github_release_notes.md", summary)
    write_zenodo_checklist(out_dir / "zenodo_deposition_checklist.md")
    write_hf_card_patch(out_dir / "hf_card_patch_notes.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("tables/release_readiness"))
    args = parser.parse_args()
    summary = write_outputs(args.root, args.out_dir)
    print(
        f"Wrote release readiness to {args.out_dir} "
        f"(release_gate_passed={summary['release_gate_passed']})"
    )


if __name__ == "__main__":
    main()
