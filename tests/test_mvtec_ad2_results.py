from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from seer_ad_v2.evaluation.mvtec_ad2_results import (
    aggregate_public_rows,
    evaluate_leaderboard,
    load_result_rows,
    load_leaderboard_rows,
    render_leaderboard_claim,
    render_results_section,
    replace_marked_section,
    sha256_file,
    summarize_rows,
    validate_evaluation_metadata,
    validate_local_submission_evidence,
)


def test_load_aliases_and_render(tmp_path: Path) -> None:
    export = tmp_path / "result.json"
    export.write_text(
        json.dumps(
            {
                "source_url": "https://benchmark.mvtec.com/results/123",
                "submission_id": "submission-123",
                "evaluated_at": "2026-06-13T12:00:00Z",
                "results": [
                    {
                        "split": "private_overall",
                        "category": "all",
                        "I-AUROC": 0.91,
                        "P-AUROC": 0.95,
                        "AU-PRO": 0.88,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata, rows = load_result_rows(export)
    identity = validate_evaluation_metadata(metadata)
    assert identity["submission_id"] == "submission-123"
    assert rows[0]["image_auroc"] == 0.91
    evaluation = {
        **identity,
        "summary": summarize_rows(rows),
    }
    section = render_results_section(evaluation)
    assert "0.9100" in section
    text = (
        "before\n<!-- MVTEC_AD2_RESULTS_START -->\nold\n"
        "<!-- MVTEC_AD2_RESULTS_END -->\nafter\n"
    )
    updated = replace_marked_section(text, section)
    assert "0.9100" in updated
    assert "old" not in updated


def test_rejects_unofficial_or_invalid_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="benchmark.mvtec.com"):
        validate_evaluation_metadata(
            {
                "source_url": "https://example.com/result",
                "submission_id": "x",
                "evaluated_at": "2026-06-13",
            }
        )
    export = tmp_path / "bad.csv"
    export.write_text(
        "split,category,image_auroc,pixel_auroc,aupro\n"
        "private_overall,all,1.2,0.9,0.8\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_result_rows(export)


def test_local_submission_evidence_required(tmp_path: Path) -> None:
    archive = tmp_path / "submission.tar.gz"
    archive.write_bytes(b"archive")
    protocol = {
        "files": 4090,
        "full_official_submission": True,
        "official_checker": {"status": "passed"},
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
    }
    validate_local_submission_evidence(protocol, archive)
    protocol["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        validate_local_submission_evidence(protocol, archive)
    protocol["archive_sha256"] = sha256_file(archive)
    protocol["files"] = 1
    with pytest.raises(ValueError, match="4090"):
        validate_local_submission_evidence(protocol, archive)


def public_rows() -> list[dict[str, object]]:
    categories = [
        "can",
        "fabric",
        "fruit_jelly",
        "rice",
        "sheet_metal",
        "vial",
        "wallplugs",
        "walnuts",
    ]
    return [
        {
            "category": category,
            "seed": seed,
            "image_auroc": 0.9,
            "pixel_auroc": 0.91,
            "aupro": 0.8,
            "pixel_ap": 0.7,
            "dice": 0.6,
        }
        for category in categories
        for seed in (7, 13, 23)
    ]


def test_public_aggregation_requires_exact_coverage() -> None:
    rows = public_rows()
    aggregate = aggregate_public_rows(rows)
    assert len(aggregate) == 3
    assert aggregate[0]["categories"] == 8
    with pytest.raises(ValueError, match="8 categories x 3 seeds"):
        aggregate_public_rows(rows[:-1])


def test_import_cli_updates_outputs_and_draft(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    export = tmp_path / "server-result.json"
    export.write_text(
        json.dumps(
            {
                "source_url": "https://benchmark.mvtec.com/results/abc",
                "submission_id": "abc",
                "evaluated_at": "2026-06-13T12:00:00Z",
                "results": [
                    {
                        "split": "private_overall",
                        "category": "all",
                        "image_auroc": 0.9,
                        "pixel_auroc": 0.91,
                        "aupro": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "submission_protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "files": 4090,
                "full_official_submission": True,
                "official_checker": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "submission.tar.gz"
    archive.write_bytes(b"archive")
    draft = tmp_path / "draft.md"
    draft.write_text(
        "<!-- MVTEC_AD2_RESULTS_START -->\npending\n"
        "<!-- MVTEC_AD2_RESULTS_END -->\n",
        encoding="utf-8",
    )
    output = tmp_path / "official_evaluation.json"
    table = tmp_path / "official_private.csv"
    public = tmp_path / "public.csv"
    public.write_text(
        "dataset,category,seed,image_auroc,pixel_auroc,aupro,pixel_ap,dice\n"
        + "\n".join(
            (
                f"mvtec_ad2,{row['category']},{row['seed']},"
                f"{row['image_auroc']},{row['pixel_auroc']},{row['aupro']},"
                f"{row['pixel_ap']},{row['dice']}"
            )
            for row in public_rows()
        )
        + "\n",
        encoding="utf-8",
    )
    existing = tmp_path / "existing.csv"
    existing.write_text(
        "dataset,seed,categories,image_auroc,pixel_auroc,aupro,pixel_ap,dice\n"
        "mvtec15,7,15,0.9,0.9,0.8,0.7,0.6\n",
        encoding="utf-8",
    )
    combined = tmp_path / "combined.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "import_mvtec_ad2_official_results.py"),
            "--input",
            str(export),
            "--submission-protocol",
            str(protocol),
            "--archive",
            str(archive),
            "--out",
            str(output),
            "--table-out",
            str(table),
            "--public-runs",
            str(public),
            "--existing-main",
            str(existing),
            "--combined-table-out",
            str(combined),
            "--draft",
            str(draft),
            "--update-draft",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    evaluation = json.loads(output.read_text(encoding="utf-8"))
    assert evaluation["complete"] is True
    assert evaluation["sota_claim_supported"] is False
    assert "0.9000" in draft.read_text(encoding="utf-8")
    assert table.is_file()
    assert len(combined.read_text(encoding="utf-8").splitlines()) == 6


def test_leaderboard_requires_matching_strict_best(tmp_path: Path) -> None:
    snapshot = tmp_path / "leaderboard.json"
    snapshot.write_text(
        json.dumps(
            {
                "source_url": "https://benchmark.mvtec.com/leaderboard",
                "captured_at": "2026-06-13T12:00:00Z",
                "leaderboard": [
                    {
                        "method": "Lite-SEER-AD",
                        "submission_id": "ours",
                        "image_auroc": 0.9,
                        "pixel_auroc": 0.91,
                        "aupro": 0.8,
                    },
                    {
                        "method": "Baseline",
                        "submission_id": "baseline",
                        "image_auroc": 0.8,
                        "pixel_auroc": 0.81,
                        "aupro": 0.7,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata, rows = load_leaderboard_rows(snapshot)
    evaluation = {
        "submission_id": "ours",
        "summary": {
            "means": {
                "image_auroc": 0.9,
                "pixel_auroc": 0.91,
                "aupro": 0.8,
            }
        },
    }
    result = evaluate_leaderboard(
        evaluation,
        rows,
        source_url=metadata["source_url"],
        captured_at=metadata["captured_at"],
    )
    assert result["sota_claim_supported"] is True
    assert "ranks first" in render_leaderboard_claim(result)
    rows[1]["aupro"] = 0.85
    result = evaluate_leaderboard(
        evaluation,
        rows,
        source_url=metadata["source_url"],
        captured_at=metadata["captured_at"],
    )
    assert result["sota_claim_supported"] is False
    assert "does not support" in render_leaderboard_claim(result)


def test_leaderboard_rejects_score_mismatch(tmp_path: Path) -> None:
    evaluation = {
        "submission_id": "ours",
        "summary": {
            "means": {
                "image_auroc": 0.9,
                "pixel_auroc": 0.91,
                "aupro": 0.8,
            }
        },
    }
    rows = [
        {
            "method": "Lite-SEER-AD",
            "submission_id": "ours",
            "image_auroc": 0.89,
            "pixel_auroc": 0.91,
            "aupro": 0.8,
        },
        {
            "method": "Baseline",
            "submission_id": "baseline",
            "image_auroc": 0.8,
            "pixel_auroc": 0.8,
            "aupro": 0.7,
        },
    ]
    with pytest.raises(ValueError, match="does not match"):
        evaluate_leaderboard(
            evaluation,
            rows,
            source_url="https://benchmark.mvtec.com/leaderboard",
            captured_at="2026-06-13T12:00:00Z",
        )


def test_leaderboard_cli_can_update_evaluation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    evaluation_path = tmp_path / "official_evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "complete": True,
                "submission_id": "ours",
                "summary": {
                    "means": {
                        "image_auroc": 0.9,
                        "pixel_auroc": 0.91,
                        "aupro": 0.8,
                    }
                },
                "sota_claim_supported": False,
            }
        ),
        encoding="utf-8",
    )
    snapshot = tmp_path / "leaderboard.json"
    snapshot.write_text(
        json.dumps(
            {
                "source_url": "https://benchmark.mvtec.com/leaderboard",
                "captured_at": "2026-06-13T12:00:00Z",
                "leaderboard": [
                    {
                        "method": "Lite-SEER-AD",
                        "submission_id": "ours",
                        "image_auroc": 0.9,
                        "pixel_auroc": 0.91,
                        "aupro": 0.8,
                    },
                    {
                        "method": "Baseline",
                        "submission_id": "baseline",
                        "image_auroc": 0.8,
                        "pixel_auroc": 0.81,
                        "aupro": 0.7,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "leaderboard_evidence.json"
    claim = tmp_path / "leaderboard_claim.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "audit_mvtec_ad2_leaderboard.py"),
            "--input",
            str(snapshot),
            "--evaluation",
            str(evaluation_path),
            "--out",
            str(evidence),
            "--claim-out",
            str(claim),
            "--update-evaluation",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    updated = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert updated["sota_claim_supported"] is True
    assert evidence.is_file()
    assert "ranks first" in claim.read_text(encoding="utf-8")
