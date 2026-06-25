from __future__ import annotations

import json
from pathlib import Path

from tools.export_submission_package_readiness import (
    build_summary,
    check_claim_alignment,
    check_final_placeholders,
    write_outputs,
)


def touch(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def final_statement_text() -> str:
    return """# Submission Statements

## Target Journal

- Journal: Journal of Industrial Anomaly Detection
- Article type: Original research
- Template status: Converted
- Required word/page limit: 8000 words

## Authors And Affiliations

- Corresponding author: A. Researcher, a@example.com, Example University
- Author list: A. Researcher
- Affiliations: Example University, Department, City, Country

## Funding Statement

No funding was received.

## Conflict Of Interest Statement

The authors declare no competing interests.

## Data Availability Statement

GitHub Release URL: https://github.com/example/lite-seer-ad/releases/tag/v1
Zenodo DOI: https://doi.org/10.5281/zenodo.123
Hugging Face URL: https://huggingface.co/example/lite-seer-ad
repository URL: https://github.com/example/lite-seer-ad

## Cover Letter Skeleton

Dear Editor,

We submit this manuscript for consideration.
"""


def make_submission_repo(root: Path, *, final_statements: bool = False) -> None:
    touch(
        root / "paper/manuscript.md",
        (
            "Lite-SEER-AD is feature-first and label-free. We do not claim "
            "universal SOTA. CRV is visualization and not evidence of semantic "
            "repair. Diffusion is not the main detector.\n"
        ),
    )
    touch(root / "paper/references.bib", "@article{x,title={x}}\n")
    touch(
        root / "paper/supplement.md",
        (
            "Supplement. Lite-SEER-AD is label-free. CRV is visualization and "
            "not evidence of semantic repair. Diffusion is not the main detector.\n"
        ),
    )
    touch(root / "docs/reviewer_faq.md", "FAQ\n")
    touch(root / "docs/results_limitations_draft.md", "Limitations\n")
    touch(root / "docs/submission_reproducibility_checklist.md", "Checklist\n")
    touch(
        root / "docs/submission_statement_placeholders.md",
        final_statement_text()
        if final_statements
        else "## Target Journal\n- Journal: [to be selected]\n## Cover Letter Skeleton\nDear Editor,\n[journal]\n",
    )
    touch(
        root / "tables/completion_gap_matrix/summary.json",
        json.dumps({"default_100_ready": final_statements}),
    )
    touch(
        root / "tables/release_readiness/summary.json",
        json.dumps({"release_gate_passed": final_statements}),
    )
    touch(
        root / "tables/final_external_handoff/final_metadata_consistency/summary.json",
        json.dumps({"final_metadata_consistent": final_statements}),
    )


def test_claim_alignment_detects_feature_first_boundaries(tmp_path: Path) -> None:
    make_submission_repo(tmp_path)

    rows = check_claim_alignment(tmp_path)

    assert rows
    assert all(row["status"] == "pass" for row in rows)


def test_final_placeholders_remain_pending_for_draft_statements(tmp_path: Path) -> None:
    make_submission_repo(tmp_path, final_statements=False)

    rows = check_final_placeholders(tmp_path)

    assert any(row["status"] == "pending_journal" for row in rows)


def test_final_placeholders_are_scoped_by_statement_section(tmp_path: Path) -> None:
    make_submission_repo(tmp_path, final_statements=False)
    statements = """# Submission Statements

## Target Journal

- Journal: Pattern Recognition
- Article type: Full Length Article
- Template status: Elsevier article template selected
- Required word/page limit: Follow Pattern Recognition Guide for Authors

## Authors And Affiliations

- Corresponding author: [name, email, affiliation]
- Author list: [ordered names]
- Affiliations: [institution, department, city, country]

## Funding Statement

[funding]

## Conflict Of Interest Statement

[conflicts]

## Data Availability Statement

GitHub Release URL: [GitHub Release URL]
Zenodo DOI: [Zenodo DOI]
Hugging Face URL: [Hugging Face URL]
repository URL: [repository URL]

## Cover Letter Skeleton

Dear Editor,
[journal]
"""
    (tmp_path / "docs/submission_statement_placeholders.md").write_text(
        statements,
        encoding="utf-8",
    )

    rows = check_final_placeholders(tmp_path)
    by_requirement = {row["requirement"]: row["status"] for row in rows}

    assert by_requirement["final_upload:target_journal"] == "pass"
    assert by_requirement["final_upload:authors_affiliations"] == "pending_journal"


def test_final_placeholders_detect_angle_bracket_template_values(tmp_path: Path) -> None:
    make_submission_repo(tmp_path, final_statements=True)
    statements = final_statement_text().replace("A. Researcher", "<ordered author names>")
    (tmp_path / "docs/submission_statement_placeholders.md").write_text(
        statements,
        encoding="utf-8",
    )

    rows = check_final_placeholders(tmp_path)
    by_requirement = {row["requirement"]: row["status"] for row in rows}

    assert by_requirement["final_upload:authors_affiliations"] == "pending_journal"


def test_write_outputs_separates_local_draft_from_final_upload(tmp_path: Path) -> None:
    make_submission_repo(tmp_path, final_statements=False)
    out_dir = tmp_path / "tables/submission_package_readiness"

    summary = write_outputs(tmp_path, out_dir)

    assert summary["local_submission_draft_ready"] is True
    assert summary["final_upload_ready"] is False
    assert "final_upload:completion_matrix_default_100" in summary["blocking_requirements"]
    assert "final_upload:release_submission_consistency" in summary["blocking_requirements"]
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "table_submission_readiness.csv").is_file()
    assert (out_dir / "submission_upload_todo.md").is_file()


def test_build_summary_passes_when_local_and_final_rows_pass() -> None:
    rows = [
        {"requirement": "structural_file:manuscript", "status": "pass", "gate": "local_draft"},
        {"requirement": "final_upload:target_journal", "status": "pass", "gate": "final_upload"},
    ]

    summary = build_summary(rows)

    assert summary["local_submission_draft_ready"] is True
    assert summary["final_upload_ready"] is True
    assert summary["release_gate_passed"] is True


def test_write_outputs_can_pass_for_fully_finalized_fixture(tmp_path: Path) -> None:
    make_submission_repo(tmp_path, final_statements=True)

    summary = write_outputs(tmp_path, tmp_path / "tables/submission_package_readiness")

    assert summary["local_submission_draft_ready"] is True
    assert summary["final_upload_ready"] is True
