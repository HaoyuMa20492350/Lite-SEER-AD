from __future__ import annotations

import json
from pathlib import Path

from tools.export_release_readiness import build_summary, check_external_links, write_outputs


def touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_release_repo(root: Path) -> None:
    for path in [
        "README.md",
        "REPRODUCE.md",
        "MODEL_CARD.md",
        "DATASETS.md",
        "CITATION.cff",
        ".zenodo.json",
        "release_links.template.json",
        "release_metadata.template.json",
        "submission_metadata.template.json",
        "paper/manuscript.md",
        "tools/export_strict_threshold_paper_artifacts.py",
        "tools/export_external_baseline_comparison.py",
        "scripts/release/build_artifact_manifest.py",
        "scripts/release/build_prediction_array_manifest.py",
        "scripts/release/export_fixed_threshold_bundle.py",
        "tables/claim_traceability/summary.json",
        "tables/claim_traceability/table_claim_traceability.csv",
    ]:
        touch(root / path, "final link text\n")
    manifest_files = [
        {"path": "README.md", "sha256": "a" * 64, "bytes": 12},
        {"path": "artifacts/predictions_manifest.json", "sha256": "b" * 64, "bytes": 12},
        {"path": "tables/claim_traceability/summary.json", "sha256": "d" * 64, "bytes": 12},
        {
            "path": "tables/claim_traceability/table_claim_traceability.csv",
            "sha256": "e" * 64,
            "bytes": 12,
        },
        {
            "path": "artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json",
            "sha256": "c" * 64,
            "bytes": 12,
        },
    ]
    touch(
        root / "artifacts/manifest.json",
        json.dumps(
            {
                "schema": "lite-seer-ad-artifact-manifest-v1",
                "files": manifest_files,
            }
        ),
    )
    entries = [
        {
            "prediction_exists": True,
            "threshold_policy_exists": True,
            "prediction_sha256": "d" * 64,
        }
        for _ in range(99)
    ]
    touch(
        root / "artifacts/predictions_manifest.json",
        json.dumps(
            {
                "all_predictions_present": True,
                "all_threshold_policies_present": True,
                "entries": entries,
            }
        ),
    )
    touch(
        root / "artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json",
        json.dumps(
            {
                "schema": "synthetic_normal_fixed_threshold_v1_bundle",
                "policy_count": 33,
                "uses_real_anomaly_labels": False,
                "uses_real_anomaly_masks": False,
            }
        ),
    )


def test_external_links_are_pending_when_release_links_missing(tmp_path: Path) -> None:
    rows = check_external_links(tmp_path)
    link_rows = [row for row in rows if row["requirement"].startswith("external_link:")]

    assert len(link_rows) == 4
    assert all(row["status"] == "pending_external" for row in link_rows)


def test_external_links_reject_invalid_non_placeholder_values(tmp_path: Path) -> None:
    touch(
        tmp_path / "release_links.json",
        json.dumps(
            {
                "github_release_url": "https://example.com/release",
                "zenodo_doi": "https://doi.org/10.0000/not-zenodo",
                "hf_model_url": "https://huggingface.co/datasets/org/data",
                "hf_dataset_url": "https://huggingface.co/org/model",
            }
        ),
    )

    rows = check_external_links(tmp_path)
    link_rows = [row for row in rows if row["requirement"].startswith("external_link:")]

    assert all(row["status"] == "pending_external" for row in link_rows)
    assert all(row["detail"].startswith("invalid format:") for row in link_rows)


def test_external_links_accept_public_identifier_shapes(tmp_path: Path) -> None:
    touch(
        tmp_path / "release_links.json",
        json.dumps(
            {
                "github_release_url": "https://github.com/org/lite-seer-ad/releases/tag/v1.0-paper",
                "zenodo_doi": "https://doi.org/10.5281/zenodo.1234567",
                "hf_model_url": "https://huggingface.co/org/lite-seer-ad-models",
                "hf_dataset_url": "https://huggingface.co/datasets/org/lite-seer-ad-artifacts",
            }
        ),
    )

    rows = check_external_links(tmp_path)
    link_rows = [row for row in rows if row["requirement"].startswith("external_link:")]

    assert all(row["status"] == "pass" for row in link_rows)


def test_rendered_identifier_consistency_passes_when_outputs_match_release_links(tmp_path: Path) -> None:
    touch(
        tmp_path / "release_links.json",
        json.dumps(
            {
                "github_release_url": "https://github.com/org/lite-seer-ad/releases/tag/v1.0-paper",
                "zenodo_doi": "https://doi.org/10.5281/zenodo.1234567",
                "hf_model_url": "https://huggingface.co/org/lite-seer-ad-models",
                "hf_dataset_url": "https://huggingface.co/datasets/org/lite-seer-ad-artifacts",
            }
        ),
    )
    touch(
        tmp_path / "CITATION.cff",
        "\n".join(
            [
                'url: "https://github.com/org/lite-seer-ad/releases/tag/v1.0-paper"',
                'doi: "10.5281/zenodo.1234567"',
            ]
        ),
    )
    touch(
        tmp_path / ".zenodo.json",
        json.dumps(
            {
                "related_identifiers": [
                    {"identifier": "https://github.com/org/lite-seer-ad/releases/tag/v1.0-paper"},
                    {"identifier": "https://huggingface.co/org/lite-seer-ad-models"},
                    {"identifier": "https://huggingface.co/datasets/org/lite-seer-ad-artifacts"},
                ]
            }
        ),
    )

    rows = check_external_links(tmp_path)
    by_requirement = {row["requirement"]: row for row in rows}

    assert by_requirement["external_consistency:citation_matches_release_links"]["status"] == "pass"
    assert (
        by_requirement["external_consistency:zenodo_related_identifiers_match_release_links"]["status"]
        == "pass"
    )


def test_rendered_identifier_consistency_rejects_mismatched_citation(tmp_path: Path) -> None:
    touch(
        tmp_path / "release_links.json",
        json.dumps(
            {
                "github_release_url": "https://github.com/org/lite-seer-ad/releases/tag/v1.0-paper",
                "zenodo_doi": "https://doi.org/10.5281/zenodo.1234567",
                "hf_model_url": "https://huggingface.co/org/lite-seer-ad-models",
                "hf_dataset_url": "https://huggingface.co/datasets/org/lite-seer-ad-artifacts",
            }
        ),
    )
    touch(tmp_path / "CITATION.cff", 'url: "https://github.com/other/repo/releases/tag/v1.0-paper"\n')
    touch(
        tmp_path / ".zenodo.json",
        json.dumps(
            {
                "related_identifiers": [
                    {"identifier": "https://github.com/org/lite-seer-ad/releases/tag/v1.0-paper"},
                    {"identifier": "https://huggingface.co/org/lite-seer-ad-models"},
                    {"identifier": "https://huggingface.co/datasets/org/lite-seer-ad-artifacts"},
                ]
            }
        ),
    )

    rows = check_external_links(tmp_path)
    by_requirement = {row["requirement"]: row for row in rows}

    assert by_requirement["external_consistency:citation_matches_release_links"]["status"] == "pending_external"
    assert (
        by_requirement["external_consistency:zenodo_related_identifiers_match_release_links"]["status"]
        == "pass"
    )


def test_summary_separates_local_ready_from_external_publication(tmp_path: Path) -> None:
    make_release_repo(tmp_path)

    summary = write_outputs(tmp_path, tmp_path / "tables/release_readiness")

    assert summary["local_artifact_ready"] is True
    assert summary["external_publication_ready"] is False
    assert summary["release_gate_passed"] is False
    assert (tmp_path / "tables/release_readiness/summary.json").is_file()
    assert (tmp_path / "tables/release_readiness/table_release_readiness.csv").is_file()
    assert (tmp_path / "tables/release_readiness/github_release_notes.md").is_file()


def test_build_summary_passes_when_everything_is_ready() -> None:
    rows = [
        {"requirement": "local_file:README.md", "status": "pass"},
        {"requirement": "external_link:zenodo_doi", "status": "pass"},
        {"requirement": "no_placeholder:model_card_links", "status": "pass"},
    ]

    summary = build_summary(rows)

    assert summary["local_artifact_ready"] is True
    assert summary["external_publication_ready"] is True
    assert summary["release_gate_passed"] is True
