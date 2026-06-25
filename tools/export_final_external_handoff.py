"""Export the final external-action handoff for Lite-SEER-AD.

The remaining 100% blockers depend on external state: a second hardware run,
public release identifiers, and final author/journal metadata. This exporter
turns those blockers into a single auditable checklist without marking any of
them complete prematurely.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXTERNAL_LINK_REQUIREMENTS = {
    "external_link:github_release_url": {
        "gate": "public_release",
        "owner_input": "Create the v1.0-paper GitHub Release and record its tag URL in release_metadata.json.",
        "completion_command": "python tools/render_release_metadata.py --input release_metadata.json; python tools/export_release_readiness.py",
        "unblocks_dimension": "公开复现",
    },
    "external_link:zenodo_doi": {
        "gate": "public_release",
        "owner_input": "Publish or reserve the Zenodo deposition and record the final DOI in release_metadata.json.",
        "completion_command": "python tools/render_release_metadata.py --input release_metadata.json; python tools/export_release_readiness.py",
        "unblocks_dimension": "公开复现",
    },
    "external_link:hf_model_url": {
        "gate": "public_release",
        "owner_input": "Publish the Hugging Face model repository and record its URL in release_metadata.json.",
        "completion_command": "python tools/render_release_metadata.py --input release_metadata.json; python tools/export_release_readiness.py",
        "unblocks_dimension": "公开复现",
    },
    "external_link:hf_dataset_url": {
        "gate": "public_release",
        "owner_input": "Publish the Hugging Face dataset/artifact repository and record its URL in release_metadata.json.",
        "completion_command": "python tools/render_release_metadata.py --input release_metadata.json; python tools/export_release_readiness.py",
        "unblocks_dimension": "公开复现",
    },
    "no_placeholder:citation_repository": {
        "gate": "public_release",
        "owner_input": "Render final CITATION.cff from release_metadata.json after release identifiers are known.",
        "completion_command": "python tools/render_release_metadata.py --input release_metadata.json; python tools/export_release_readiness.py",
        "unblocks_dimension": "公开复现",
    },
    "no_placeholder:zenodo_metadata": {
        "gate": "public_release",
        "owner_input": "Render final .zenodo.json from release_metadata.json after creators and release identifiers are known.",
        "completion_command": "python tools/render_release_metadata.py --input release_metadata.json; python tools/export_release_readiness.py",
        "unblocks_dimension": "公开复现",
    },
    "external_consistency:citation_matches_release_links": {
        "gate": "public_release",
        "owner_input": "Render final CITATION.cff and release_links.json from one consistent release_metadata.json.",
        "completion_command": "python tools/render_release_metadata.py --input release_metadata.json; python tools/export_release_readiness.py",
        "unblocks_dimension": "公开复现",
    },
    "external_consistency:zenodo_related_identifiers_match_release_links": {
        "gate": "public_release",
        "owner_input": "Render final .zenodo.json and release_links.json from one consistent release_metadata.json.",
        "completion_command": "python tools/render_release_metadata.py --input release_metadata.json; python tools/export_release_readiness.py",
        "unblocks_dimension": "公开复现",
    },
}

SUBMISSION_REQUIREMENTS = {
    "final_upload:release_submission_consistency": {
        "owner_input": "Fill release_metadata.json and submission_metadata.json with matching public identifiers.",
        "completion_command": "python tools/validate_release_submission_consistency.py --fail-on-inconsistent; python tools/export_submission_package_readiness.py; python tools/export_completion_gap_matrix.py",
        "unblocks_dimension": "论文与投稿",
    },
    "final_upload:authors_affiliations": {
        "owner_input": "Fill corresponding author, ordered author list, affiliations, and ORCID fields.",
        "completion_command": "python tools/render_submission_statements.py --input submission_metadata.json --out docs/submission_statement_placeholders.md; python tools/export_submission_package_readiness.py; python tools/export_completion_gap_matrix.py",
        "unblocks_dimension": "论文与投稿",
    },
    "final_upload:funding_conflicts": {
        "owner_input": "Fill the exact funding statement and conflict-of-interest statement required by the target journal.",
        "completion_command": "python tools/render_submission_statements.py --input submission_metadata.json --out docs/submission_statement_placeholders.md; python tools/export_submission_package_readiness.py; python tools/export_completion_gap_matrix.py",
        "unblocks_dimension": "论文与投稿",
    },
    "final_upload:availability_links": {
        "owner_input": "Replace availability placeholders with the final GitHub Release, Zenodo DOI, and Hugging Face URLs.",
        "completion_command": "python tools/render_submission_statements.py --input submission_metadata.json --out docs/submission_statement_placeholders.md; python tools/export_submission_package_readiness.py; python tools/export_completion_gap_matrix.py",
        "unblocks_dimension": "论文与投稿",
    },
    "final_upload:cover_letter": {
        "owner_input": "Replace journal/article-type/corresponding-author placeholders in the cover letter.",
        "completion_command": "python tools/render_submission_statements.py --input submission_metadata.json --out docs/submission_statement_placeholders.md; python tools/export_submission_package_readiness.py; python tools/export_completion_gap_matrix.py",
        "unblocks_dimension": "论文与投稿",
    },
    "final_upload:completion_matrix_default_100": {
        "owner_input": "Clear the remaining P0 completion blockers and regenerate the completion matrix.",
        "completion_command": "python tools/export_completion_gap_matrix.py; python tools/export_submission_package_readiness.py",
        "unblocks_dimension": "论文与投稿",
    },
    "final_upload:public_release_identifiers": {
        "owner_input": "Complete the public-release gate first; submission availability statements depend on those identifiers.",
        "completion_command": "python tools/render_release_metadata.py --input release_metadata.json; python tools/export_release_readiness.py; python tools/render_submission_statements.py --input submission_metadata.json --out docs/submission_statement_placeholders.md; python tools/export_submission_package_readiness.py; python tools/export_completion_gap_matrix.py",
        "unblocks_dimension": "论文与投稿",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def handoff_row(
    *,
    gate: str,
    requirement: str,
    status: str,
    evidence: str,
    detail: str,
    owner_input: str,
    completion_command: str,
    unblocks_dimension: str,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "detail": detail,
        "owner_input": owner_input,
        "completion_command": completion_command,
        "unblocks_dimension": unblocks_dimension,
    }


def build_deployment_rows(root: Path) -> list[dict[str, Any]]:
    summary = read_json(root / "tables/deployment_readiness/summary.json")
    production = read_json(root / "tables/deployment_production_latency/summary.json")
    ready = summary.get("release_gate_passed") is True
    blocked = "production:cross_hardware" in (summary.get("blocking_requirements") or [])
    status = "pass" if ready else "pending_external" if blocked else "blocked"
    return [
        handoff_row(
            gate="deployment",
            requirement="production:cross_hardware",
            status=status,
            evidence="tables/deployment_production_latency/summary.json",
            detail=f"hardware_profiles={production.get('hardware_profiles')}; cross_hardware_ready={production.get('cross_hardware_ready')}",
            owner_input=(
                "Run tables/deployment_production_latency/second_hardware_run_package/run_second_hardware_probe.ps1 "
                "on another machine, then return the generated second_hardware_return_package energy/profile JSON files."
            ),
            completion_command=(
                "python tools/export_second_hardware_run_package.py; "
                "python tools/validate_second_hardware_package.py "
                "--hardware-profile <returned-second-hardware-profile.json> "
                "--energy-measurement <returned-second-hardware-energy.json> --stage; "
                "python tools/export_production_deployment_latency.py --command-run-prefix production_lc_rds_budget_guarded; "
                "python tools/export_deployment_readiness.py"
            ),
            unblocks_dimension="效率与部署",
        )
    ]


def build_release_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for release_row in read_csv(root / "tables/release_readiness/table_release_readiness.csv"):
        requirement = release_row.get("requirement", "")
        spec = EXTERNAL_LINK_REQUIREMENTS.get(requirement)
        if spec is None:
            continue
        rows.append(
            handoff_row(
                gate=spec["gate"],
                requirement=requirement,
                status=release_row.get("status", "pending_external"),
                evidence=release_row.get("evidence", ""),
                detail=release_row.get("detail", ""),
                owner_input=spec["owner_input"],
                completion_command=spec["completion_command"],
                unblocks_dimension=spec["unblocks_dimension"],
            )
        )
    return rows


def build_submission_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for submission_row in read_csv(root / "tables/submission_package_readiness/table_submission_readiness.csv"):
        requirement = submission_row.get("requirement", "")
        spec = SUBMISSION_REQUIREMENTS.get(requirement)
        if spec is None:
            continue
        rows.append(
            handoff_row(
                gate="submission",
                requirement=requirement,
                status=submission_row.get("status", "pending_journal"),
                evidence=submission_row.get("evidence", ""),
                detail=submission_row.get("detail", ""),
                owner_input=spec["owner_input"],
                completion_command=spec["completion_command"],
                unblocks_dimension=spec["unblocks_dimension"],
            )
        )
    return rows


def build_rows(root: Path) -> list[dict[str, Any]]:
    return build_deployment_rows(root) + build_release_rows(root) + build_submission_rows(root)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    gate_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        status = str(row["status"])
        gate = str(row["gate"])
        counts[status] = counts.get(status, 0) + 1
        gate_counts.setdefault(gate, {})
        gate_counts[gate][status] = gate_counts[gate].get(status, 0) + 1
    unresolved = [row for row in rows if row["status"] != "pass"]
    return {
        "schema": "lite-seer-ad-final-external-handoff-v1",
        "rows": len(rows),
        "all_external_actions_complete": len(unresolved) == 0,
        "counts": counts,
        "gate_counts": gate_counts,
        "unresolved_requirements": [row["requirement"] for row in unresolved],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "gate",
        "requirement",
        "status",
        "evidence",
        "detail",
        "owner_input",
        "completion_command",
        "unblocks_dimension",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Final External Handoff",
        "",
        f"- All external actions complete: `{summary['all_external_actions_complete']}`",
        f"- Rows: `{summary['rows']}`",
        "",
        "| Gate | Requirement | Status | Owner input | Completion command |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['gate']} | `{row['requirement']}` | `{row['status']}` | "
            f"{row['owner_input']} | `{row['completion_command']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(root: Path, out_dir: Path) -> dict[str, Any]:
    rows = build_rows(root)
    summary = build_summary(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_final_external_handoff.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "final_external_handoff.md", summary, rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("tables/final_external_handoff"))
    args = parser.parse_args()
    summary = write_outputs(args.root, args.out_dir)
    print(
        f"Wrote final external handoff to {args.out_dir} "
        f"(all_external_actions_complete={summary['all_external_actions_complete']})"
    )


if __name__ == "__main__":
    main()
