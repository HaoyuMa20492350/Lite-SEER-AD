"""Export submission-package readiness artifacts.

This audit keeps the manuscript submission gate separate from local draft
readiness. A draft package can be structurally complete while final upload is
still blocked by target-journal choice, real author/funding statements, public
release links, or unresolved completion-matrix blockers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PLACEHOLDER_RE = re.compile(r"<[^>]+>|\[[^\]]+\]|to be selected|to be checked|TBD|TODO", re.IGNORECASE)
FEATURE_FIRST_RE = re.compile(r"feature-first|label-free", re.IGNORECASE)
REQUIRED_STRUCTURAL_FILES = {
    "manuscript": "paper/manuscript.md",
    "references": "paper/references.bib",
    "supplement": "paper/supplement.md",
    "reviewer_faq": "docs/reviewer_faq.md",
    "statements_and_cover": "docs/submission_statement_placeholders.md",
    "reproducibility_checklist": "docs/submission_reproducibility_checklist.md",
    "limitations": "docs/results_limitations_draft.md",
}
FINAL_PLACEHOLDER_SECTIONS = {
    "target_journal": {
        "headings": ["Target Journal"],
        "markers": ["Journal:", "Template status:", "Required word/page limit:"],
    },
    "authors_affiliations": {
        "headings": ["Authors And Affiliations"],
        "markers": ["Corresponding author:", "Author list:", "Affiliations:"],
    },
    "funding_conflicts": {
        "headings": ["Funding Statement", "Conflict Of Interest Statement"],
        "markers": ["Funding Statement", "Conflict Of Interest Statement"],
    },
    "availability_links": {
        "headings": ["Data Availability Statement", "Code Availability Statement"],
        "markers": ["GitHub Release URL", "Zenodo DOI", "Hugging Face URL", "repository URL"],
    },
    "cover_letter": {
        "headings": ["Cover Letter Skeleton"],
        "markers": ["Dear Editor", "[journal]"],
    },
}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def status_row(
    requirement: str,
    status: str,
    evidence: str,
    detail: str,
    gate: str = "local_draft",
) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "status": status,
        "gate": gate,
        "evidence": evidence,
        "detail": detail,
    }


def check_structural_files(root: Path) -> list[dict[str, Any]]:
    rows = []
    for name, relative in REQUIRED_STRUCTURAL_FILES.items():
        path = root / relative
        rows.append(
            status_row(
                f"structural_file:{name}",
                "pass" if path.is_file() and path.stat().st_size > 0 else "fail",
                relative,
                "present" if path.is_file() else "missing",
            )
        )
    return rows


def check_claim_alignment(root: Path) -> list[dict[str, Any]]:
    manuscript = read_text(root / "paper/manuscript.md")
    supplement = read_text(root / "paper/supplement.md")
    combined = f"{manuscript}\n{supplement}".lower()
    checks = [
        (
            "claim_alignment:feature_first_label_free",
            bool(FEATURE_FIRST_RE.search(manuscript)) and "label-free" in combined,
            "feature-first/label-free mainline present",
        ),
        (
            "claim_alignment:no_universal_sota",
            "universal sota" in combined and "do not claim" in combined,
            "universal SOTA is explicitly negated",
        ),
        (
            "claim_alignment:crv_downgraded",
            "crv" in combined and "visualization" in combined and "not evidence" in combined,
            "CRV is bounded as visualization/audit rather than main contribution",
        ),
        (
            "claim_alignment:diffusion_not_main_detector",
            "diffusion" in combined
            and (
                "not the main detector" in combined
                or "not the main anomaly detector" in combined
                or "confines diffusion reasoning to optional regional evidence" in combined
                or "diffusion repair is treated as a replaceable executor" in combined
            ),
            "diffusion is excluded from the main detector claim",
        ),
    ]
    return [
        status_row(name, "pass" if ok else "fail", "paper/manuscript.md; paper/supplement.md", detail)
        for name, ok, detail in checks
    ]


def markdown_section(text: str, heading: str) -> str:
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        return ""
    next_match = re.search(r"^##\s+", text[match.end() :], re.MULTILINE)
    end = match.end() + next_match.start() if next_match else len(text)
    return text[match.start() : end]


def section_has_placeholder(text: str, headings: list[str], markers: list[str]) -> bool:
    section = "\n".join(markdown_section(text, heading) for heading in headings)
    if not section.strip():
        return True
    required_markers = [marker for marker in markers if "[" not in marker and "]" not in marker]
    if any(marker not in section for marker in required_markers):
        return True
    return bool(PLACEHOLDER_RE.search(section))


def check_final_placeholders(root: Path) -> list[dict[str, Any]]:
    statements = read_text(root / "docs/submission_statement_placeholders.md")
    rows = []
    for name, spec in FINAL_PLACEHOLDER_SECTIONS.items():
        pending = section_has_placeholder(statements, spec["headings"], spec["markers"])
        rows.append(
            status_row(
                f"final_upload:{name}",
                "pending_journal" if pending else "pass",
                "docs/submission_statement_placeholders.md",
                "contains target-journal or author placeholder"
                if pending
                else "final values present",
                gate="final_upload",
            )
        )
    return rows


def check_external_gates(root: Path) -> list[dict[str, Any]]:
    completion = read_json(root / "tables/completion_gap_matrix/summary.json")
    release = read_json(root / "tables/release_readiness/summary.json")
    consistency = read_json(root / "tables/final_external_handoff/final_metadata_consistency/summary.json")
    rows = [
        status_row(
            "final_upload:release_submission_consistency",
            "pass" if consistency.get("final_metadata_consistent") is True else "pending_external",
            "tables/final_external_handoff/final_metadata_consistency/summary.json",
            "release and submission metadata are consistent"
            if consistency.get("final_metadata_consistent") is True
            else "release_metadata.json and submission_metadata.json are missing or inconsistent",
            gate="final_upload",
        ),
        status_row(
            "final_upload:completion_matrix_default_100",
            "pass" if completion.get("default_100_ready") is True else "blocked",
            "tables/completion_gap_matrix/summary.json",
            "default_100_ready=true"
            if completion.get("default_100_ready") is True
            else "completion matrix still has blocking P0 dimensions",
            gate="final_upload",
        ),
        status_row(
            "final_upload:public_release_identifiers",
            "pass" if release.get("release_gate_passed") is True else "pending_external",
            "tables/release_readiness/summary.json",
            "public release gate passed"
            if release.get("release_gate_passed") is True
            else "GitHub/Zenodo/Hugging Face identifiers are not finalized",
            gate="final_upload",
        ),
    ]
    return rows


def build_rows(root: Path) -> list[dict[str, Any]]:
    return (
        check_structural_files(root)
        + check_claim_alignment(root)
        + check_final_placeholders(root)
        + check_external_gates(root)
    )


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    local_rows = [row for row in rows if row["gate"] == "local_draft"]
    final_rows = [row for row in rows if row["gate"] == "final_upload"]
    local_ready = all(row["status"] == "pass" for row in local_rows)
    final_ready = local_ready and all(row["status"] == "pass" for row in final_rows)
    return {
        "schema": "lite-seer-ad-submission-package-readiness-v1",
        "local_submission_draft_ready": local_ready,
        "final_upload_ready": final_ready,
        "release_gate_passed": final_ready,
        "release_gate_reason": (
            "Submission package is structurally ready, but final upload is gated by journal placeholders, public-release identifiers, or remaining completion blockers."
            if local_ready and not final_ready
            else "Submission package is ready for upload."
            if final_ready
            else "Submission package draft is structurally incomplete."
        ),
        "counts": counts,
        "blocking_requirements": [
            row["requirement"] for row in rows if row["status"] != "pass"
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["requirement", "status", "gate", "evidence", "detail"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_upload_todo(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Submission Upload TODO",
        "",
        f"- Local draft ready: `{summary['local_submission_draft_ready']}`",
        f"- Final upload ready: `{summary['final_upload_ready']}`",
        "",
        "Blocking items:",
        "",
    ]
    blockers = [row for row in rows if row["status"] != "pass"]
    if not blockers:
        lines.append("- None")
    else:
        for row in blockers:
            lines.append(f"- `{row['requirement']}`: {row['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(root: Path, out_dir: Path) -> dict[str, Any]:
    rows = build_rows(root)
    summary = build_summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_submission_readiness.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_upload_todo(out_dir / "submission_upload_todo.md", summary, rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("tables/submission_package_readiness"))
    args = parser.parse_args()
    summary = write_outputs(args.root, args.out_dir)
    print(
        f"Wrote submission package readiness to {args.out_dir} "
        f"(final_upload_ready={summary['final_upload_ready']})"
    )


if __name__ == "__main__":
    main()
