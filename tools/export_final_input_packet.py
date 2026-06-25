"""Export one packet for the remaining final external inputs.

The plan-defined 100% state still depends on values that cannot be invented
inside the repository: public release identifiers, author/journal metadata,
and a second hardware run. This exporter collects those inputs into one
machine-readable packet so the final closeout can be completed without hunting
through several readiness tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.export_final_external_handoff import build_rows as build_handoff_rows


SCHEMA = "lite-seer-ad-final-input-packet-v1"
DEFAULT_OUT_DIR = Path("tables/final_external_handoff/final_input_packet")
PLACEHOLDER_RE = re.compile(r"<[^>]+>|\[[^\]]+\]|placeholder|TBD|TODO|to be", re.IGNORECASE)
GITHUB_HTTPS_RE = re.compile(r"^https://github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?/?$")
GITHUB_SSH_RE = re.compile(r"^git@github\.com:(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?$")
SECOND_HARDWARE_PACKAGE = Path("tables/deployment_production_latency/second_hardware_run_package")
DEFAULT_SECOND_HARDWARE_RETURN_FILES = [
    "second_hardware_return_package/second_hardware_probe_energy.json",
    "second_hardware_return_package/second_hardware_probe_hardware_profile.json",
]
TEMPLATE_SPECS = [
    {
        "source": "release",
        "template": "release_metadata.template.json",
        "draft": "release_metadata.draft.json",
        "final": "release_metadata.json",
        "validator": "python tools/render_release_metadata.py --input release_metadata.json",
    },
    {
        "source": "submission",
        "template": "submission_metadata.template.json",
        "draft": "submission_metadata.draft.json",
        "final": "submission_metadata.json",
        "validator": (
            "python tools/render_submission_statements.py --input submission_metadata.json "
            "--out docs/submission_statement_placeholders.md"
        ),
    },
]
SECOND_HARDWARE_INPUTS = [
    {
        "source": "second_hardware",
        "json_path": "$SecondHardwareProfile",
        "destination_file": "second_hardware_profile.json",
        "prompt": "Return the hardware profile JSON generated on a second machine.",
        "validator": (
            "python tools/validate_second_hardware_package.py "
            "--hardware-profile <profile.json> --energy-measurement <energy.json> --stage"
        ),
    },
    {
        "source": "second_hardware",
        "json_path": "$SecondHardwareEnergy",
        "destination_file": "second_hardware_energy.json",
        "prompt": "Return the energy measurement JSON generated on a second machine.",
        "validator": (
            "python tools/validate_second_hardware_package.py "
            "--hardware-profile <profile.json> --energy-measurement <energy.json> --stage"
        ),
    },
]
REQUIREMENT_INPUT_HINTS = {
    "production:cross_hardware": {
        "input_files": ["second_hardware_profile.json", "second_hardware_energy.json"],
        "field_hints": ["$SecondHardwareProfile", "$SecondHardwareEnergy"],
    },
    "external_link:github_release_url": {
        "input_files": ["release_metadata.json"],
        "field_hints": ["release.github_release_url"],
    },
    "external_link:zenodo_doi": {
        "input_files": ["release_metadata.json"],
        "field_hints": ["release.zenodo_doi"],
    },
    "external_link:hf_model_url": {
        "input_files": ["release_metadata.json"],
        "field_hints": ["release.hf_model_url"],
    },
    "external_link:hf_dataset_url": {
        "input_files": ["release_metadata.json"],
        "field_hints": ["release.hf_dataset_url"],
    },
    "no_placeholder:citation_repository": {
        "input_files": ["release_metadata.json"],
        "field_hints": ["citation.repository_code"],
    },
    "no_placeholder:zenodo_metadata": {
        "input_files": ["release_metadata.json"],
        "field_hints": ["zenodo.creators", "zenodo.related_identifiers"],
    },
    "external_consistency:citation_matches_release_links": {
        "input_files": ["release_metadata.json"],
        "field_hints": [
            "release.github_release_url",
            "release.zenodo_doi",
            "citation.repository_code",
            "citation.version",
        ],
    },
    "external_consistency:zenodo_related_identifiers_match_release_links": {
        "input_files": ["release_metadata.json"],
        "field_hints": [
            "release.github_release_url",
            "release.hf_model_url",
            "release.hf_dataset_url",
            "zenodo.related_identifiers",
        ],
    },
    "final_upload:authors_affiliations": {
        "input_files": ["submission_metadata.json", "release_metadata.json"],
        "field_hints": [
            "submission.authors",
            "submission.author_contributions",
            "release.citation.authors",
            "release.zenodo.creators",
        ],
    },
    "final_upload:funding_conflicts": {
        "input_files": ["submission_metadata.json"],
        "field_hints": ["funding_statement", "conflict_of_interest_statement"],
    },
    "final_upload:availability_links": {
        "input_files": ["submission_metadata.json", "release_metadata.json"],
        "field_hints": [
            "data_availability_statement",
            "code_availability_statement",
            "cover_letter.availability_sentence",
            "release.*",
            "citation.repository_code",
        ],
    },
    "final_upload:cover_letter": {
        "input_files": ["submission_metadata.json"],
        "field_hints": ["cover_letter.*"],
    },
    "final_upload:release_submission_consistency": {
        "input_files": ["release_metadata.json", "submission_metadata.json"],
        "field_hints": [
            "release.github_release_url",
            "release.zenodo_doi",
            "release.hf_model_url",
            "release.hf_dataset_url",
            "citation.repository_code",
            "submission availability statements",
        ],
    },
    "final_upload:completion_matrix_default_100": {
        "input_files": [
            "second_hardware_profile.json",
            "second_hardware_energy.json",
            "release_metadata.json",
            "submission_metadata.json",
        ],
        "field_hints": ["all final external gates"],
    },
    "final_upload:public_release_identifiers": {
        "input_files": ["release_metadata.json", "submission_metadata.json"],
        "field_hints": [
            "release.github_release_url",
            "release.zenodo_doi",
            "release.hf_model_url",
            "release.hf_dataset_url",
            "submission availability statements",
        ],
    },
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def git_dir(root: Path) -> Path | None:
    dot_git = root / ".git"
    if dot_git.is_dir():
        return dot_git
    if not dot_git.is_file():
        return None
    text = dot_git.read_text(encoding="utf-8", errors="ignore").strip()
    prefix = "gitdir:"
    if not text.lower().startswith(prefix):
        return None
    target = Path(text[len(prefix) :].strip())
    return target if target.is_absolute() else (root / target).resolve()


def read_origin_remote(root: Path) -> str:
    directory = git_dir(root)
    config_path = directory / "config" if directory else root / ".git" / "config"
    if not config_path.exists():
        return ""
    in_origin = False
    for line in config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_origin = stripped in {'[remote "origin"]', "[remote 'origin']"}
            continue
        if in_origin and stripped.startswith("url") and "=" in stripped:
            return stripped.split("=", 1)[1].strip()
    return ""


def github_repository_url(root: Path) -> str:
    remote = read_origin_remote(root)
    match = GITHUB_HTTPS_RE.match(remote) or GITHUB_SSH_RE.match(remote)
    if not match:
        return ""
    repo = match.group("repo")
    if repo.endswith(".git"):
        repo = repo[:-4]
    return f"https://github.com/{match.group('owner')}/{repo}"


def read_head_commit(root: Path) -> str:
    directory = git_dir(root)
    if directory is None:
        return ""
    head_path = directory / "HEAD"
    if not head_path.exists():
        return ""
    head = head_path.read_text(encoding="utf-8", errors="ignore").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", head):
        return head
    if head.startswith("ref:"):
        ref_path = directory / head.removeprefix("ref:").strip()
        if ref_path.exists():
            ref = ref_path.read_text(encoding="utf-8", errors="ignore").strip()
            if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
                return ref
    return ""


def release_tag_from_template(value: str) -> str:
    match = re.search(r"/releases/tag/([^/\s]+)", value)
    if not match:
        return "v1.0-paper"
    tag = match.group(1)
    return "v1.0-paper" if PLACEHOLDER_RE.search(tag) else tag


def suggested_value_or_hint(root: Path, source: str, json_path: str, template_value: str) -> str:
    repo_url = github_repository_url(root)
    commit = read_head_commit(root)
    if source == "release" and json_path == "release.github_release_url" and repo_url:
        tag = release_tag_from_template(template_value)
        return f"{repo_url}/releases/tag/{tag} (create this release before marking the gate complete)"
    if source == "release" and json_path == "citation.repository_code" and repo_url:
        return repo_url
    if source == "submission" and json_path == "code_availability_statement" and repo_url:
        commit_hint = (
            f" Current HEAD is {commit}; replace it with the final tagged release commit after committing release changes."
            if commit
            else " Fill the exact tagged release commit after committing release changes."
        )
        return f"Use repository URL {repo_url}.{commit_hint}"
    return ""


def second_hardware_return_files(root: Path) -> list[str]:
    manifest = read_json(root / SECOND_HARDWARE_PACKAGE / "manifest.json")
    files = manifest.get("expected_return_files")
    if isinstance(files, list) and all(str(item).strip() for item in files):
        return [str(item) for item in files]
    return DEFAULT_SECOND_HARDWARE_RETURN_FILES


def second_hardware_hint(root: Path, destination_file: str) -> str:
    files = second_hardware_return_files(root)
    if "profile" in destination_file:
        match = next((item for item in files if item.endswith("_hardware_profile.json")), "")
        if match:
            return f"Use `{match}` from the generated second-hardware run package; do not reuse the primary-machine profile."
    if "energy" in destination_file:
        match = next((item for item in files if item.endswith("_energy.json")), "")
        if match:
            return f"Use `{match}` from the generated second-hardware run package; do not reuse primary-machine energy evidence."
    return "Use the generated second-hardware run package; do not reuse primary-machine evidence."


def prefill_release_draft(root: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    repo_url = github_repository_url(root)
    if not repo_url:
        return payload, []
    citation = payload.get("citation")
    if not isinstance(citation, dict):
        return payload, []
    citation["repository_code"] = repo_url
    return payload, ["citation.repository_code"]


def prefill_submission_draft(root: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    repo_url = github_repository_url(root)
    if not repo_url:
        return payload, []
    statement = str(payload.get("code_availability_statement", ""))
    if "<repository url>" not in statement:
        return payload, []
    payload["code_availability_statement"] = statement.replace("<repository url>", repo_url)
    return payload, ["code_availability_statement.repository_url"]


def prefill_draft_payload(root: Path, spec: dict[str, str], payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    source = spec["source"]
    if source == "release":
        return prefill_release_draft(root, payload)
    if source == "submission":
        return prefill_submission_draft(root, payload)
    return payload, []


def walk_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(walk_strings(item, child))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(walk_strings(item, f"{prefix}[{index}]"))
        return rows
    return [(prefix, "" if value is None else str(value))]


def placeholder_rows(root: Path, spec: dict[str, str]) -> list[dict[str, Any]]:
    template_path = root / spec["template"]
    payload = read_json(template_path)
    rows: list[dict[str, Any]] = []
    for json_path, value in walk_strings(payload):
        if not PLACEHOLDER_RE.search(value):
            continue
        rows.append(
            {
                "source": spec["source"],
                "requirement": f"metadata_field:{spec['final']}::{json_path}",
                "destination_file": spec["final"],
                "json_path": json_path,
                "prompt": f"Replace template value `{value}` with the final value.",
                "suggested_value_or_hint": suggested_value_or_hint(root, spec["source"], json_path, value),
                "validator": spec["validator"],
            }
        )
    return rows


def build_field_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in SECOND_HARDWARE_INPUTS:
        rows.append(
            {
                "source": item["source"],
                "requirement": f"external_input:{item['destination_file']}",
                "destination_file": item["destination_file"],
                "json_path": item["json_path"],
                "prompt": item["prompt"],
                "suggested_value_or_hint": second_hardware_hint(root, item["destination_file"]),
                "validator": item["validator"],
            }
        )
    for spec in TEMPLATE_SPECS:
        rows.extend(placeholder_rows(root, spec))
    return rows


def unique_commands(commands: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for command in commands:
        if command and command not in seen:
            seen.add(command)
            out.append(command)
    return out


def metadata_input_sources(root: Path, field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in TEMPLATE_SPECS:
        template_path = root / spec["template"]
        final_path = root / spec["final"]
        placeholder_count = sum(
            1
            for row in field_rows
            if row["source"] == spec["source"] and str(row["requirement"]).startswith("metadata_field:")
        )
        rows.append(
            {
                "source": spec["source"],
                "template": spec["template"],
                "final": spec["final"],
                "template_exists": template_path.is_file(),
                "final_exists": final_path.is_file(),
                "placeholder_fields": placeholder_count,
            }
        )
    return rows


def build_handoff_input_coverage_rows(handoff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in handoff_rows:
        if row.get("status") == "pass":
            continue
        requirement = str(row.get("requirement", ""))
        hints = REQUIREMENT_INPUT_HINTS.get(requirement, {})
        input_files = hints.get("input_files", [])
        field_hints = hints.get("field_hints", [])
        covered = bool(input_files and field_hints)
        rows.append(
            {
                "requirement": requirement,
                "status": row.get("status", ""),
                "covered": "true" if covered else "false",
                "input_files": ";".join(str(item) for item in input_files),
                "field_hints": ";".join(str(item) for item in field_hints),
                "completion_command": row.get("completion_command", ""),
            }
        )
    return rows


def packet_blockers(root: Path, field_rows: list[dict[str, Any]], handoff_rows: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for metadata_source in metadata_input_sources(root, field_rows):
        source = metadata_source["source"]
        template_exists = bool(metadata_source["template_exists"])
        final_exists = bool(metadata_source["final_exists"])
        placeholder_count = int(metadata_source["placeholder_fields"])
        if not template_exists and not final_exists:
            blockers.append(f"missing_metadata_input_source:{source}")
        if template_exists and not final_exists and placeholder_count == 0:
            blockers.append(f"missing_metadata_field_prompts:{source}")

    second_hardware_rows = [row for row in field_rows if row["source"] == "second_hardware"]
    if len(second_hardware_rows) != len(SECOND_HARDWARE_INPUTS):
        blockers.append("missing_second_hardware_inputs")

    metadata_fields = [row for row in field_rows if row["source"] in {"release", "submission"}]
    if not metadata_fields and not all(source["final_exists"] for source in metadata_input_sources(root, field_rows)):
        blockers.append("missing_metadata_placeholder_fields")

    if not handoff_rows:
        blockers.append("missing_external_handoff_rows")

    for coverage_row in build_handoff_input_coverage_rows(handoff_rows):
        if coverage_row["covered"] != "true":
            blockers.append(f"missing_input_coverage:{coverage_row['requirement']}")

    for row in handoff_rows:
        if row.get("status") == "pass":
            continue
        if not str(row.get("completion_command", "")).strip():
            blockers.append(f"missing_completion_command:{row.get('requirement')}")
        if not str(row.get("owner_input", "")).strip():
            blockers.append(f"missing_owner_input:{row.get('requirement')}")
    return blockers


def build_summary(root: Path, field_rows: list[dict[str, Any]], handoff_rows: list[dict[str, Any]]) -> dict[str, Any]:
    unresolved = [row for row in handoff_rows if row.get("status") != "pass"]
    status_counts: dict[str, int] = {}
    for row in unresolved:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    metadata_fields = [row for row in field_rows if row["source"] in {"release", "submission"}]
    blockers = packet_blockers(root, field_rows, handoff_rows)
    metadata_sources = metadata_input_sources(root, field_rows)
    coverage_rows = build_handoff_input_coverage_rows(handoff_rows)
    draft_prefills = draft_prefill_fields(root)
    return {
        "schema": SCHEMA,
        "packet_ready": not blockers,
        "packet_blocking_requirements": blockers,
        "external_handoff_rows": len(handoff_rows),
        "unresolved_handoff_rows": len(unresolved),
        "handoff_input_coverage_rows": len(coverage_rows),
        "handoff_input_coverage_ready": all(row["covered"] == "true" for row in coverage_rows),
        "unresolved_status_counts": status_counts,
        "field_rows": len(field_rows),
        "metadata_placeholder_fields": len(metadata_fields),
        "metadata_input_sources": metadata_sources,
        "draft_prefill_fields": draft_prefills,
        "second_hardware_inputs": len([row for row in field_rows if row["source"] == "second_hardware"]),
        "metadata_templates": [str(root / spec["template"]) for spec in TEMPLATE_SPECS],
        "draft_files": [spec["draft"] for spec in TEMPLATE_SPECS],
        "final_files_to_create": [spec["final"] for spec in TEMPLATE_SPECS]
        + ["second_hardware_profile.json", "second_hardware_energy.json"],
        "unresolved_requirements": [str(row.get("requirement")) for row in unresolved],
        "completion_commands": unique_commands([str(row.get("completion_command", "")) for row in unresolved]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "requirement",
        "destination_file",
        "json_path",
        "prompt",
        "suggested_value_or_hint",
        "validator",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_coverage_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["requirement", "status", "covered", "input_files", "field_hints", "completion_command"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_second_hardware_manifest(path: Path, root: Path) -> None:
    payload = {
        "schema": "lite-seer-ad-second-hardware-inputs-v1",
        "required_inputs": SECOND_HARDWARE_INPUTS,
        "package": SECOND_HARDWARE_PACKAGE.as_posix(),
        "expected_return_files": second_hardware_return_files(root),
        "stage_command": (
            "python tools/validate_second_hardware_package.py "
            "--hardware-profile <profile.json> --energy-measurement <energy.json> --stage"
        ),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(summary: dict[str, Any], field_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Final Input Packet",
        "",
        "This packet lists the remaining external values needed before the 100% closeout can pass.",
        "It does not mark any external requirement complete; it only centralizes the inputs.",
        "",
        f"- Unresolved handoff rows: `{summary['unresolved_handoff_rows']}`",
        f"- Handoff input coverage ready: `{summary['handoff_input_coverage_ready']}`",
        f"- Metadata placeholder fields: `{summary['metadata_placeholder_fields']}`",
        f"- Second hardware inputs: `{summary['second_hardware_inputs']}`",
        f"- Packet ready: `{summary['packet_ready']}`",
        "",
        "Fill these final files at the repository root:",
        "",
    ]
    if summary["packet_blocking_requirements"]:
        lines.extend(
            [
                "Packet blockers:",
                "",
            ]
        )
        for blocker in summary["packet_blocking_requirements"]:
            lines.append(f"- `{blocker}`")
        lines.append("")
    lines.extend(
        [
            "Metadata input sources:",
            "",
            "| Source | Template exists | Final file exists | Placeholder fields |",
            "|---|---:|---:|---:|",
        ]
    )
    for source in summary["metadata_input_sources"]:
        lines.append(
            f"| `{source['source']}` | `{source['template_exists']}` | "
            f"`{source['final_exists']}` | `{source['placeholder_fields']}` |"
        )
    lines.append("")
    lines.extend(
        [
            "Draft prefilled fields:",
            "",
        ]
    )
    if summary["draft_prefill_fields"]:
        for field in summary["draft_prefill_fields"]:
            lines.append(f"- `{field}`")
    else:
        lines.append("- None")
    lines.append("")
    lines.extend(
        [
            "Handoff-to-input coverage:",
            "",
            "See `table_handoff_input_coverage.csv` for the machine-readable mapping from each unresolved requirement to its required input files and field hints.",
            "",
        ]
    )
    for item in summary["final_files_to_create"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "Draft metadata files are included in this packet:",
            "",
        ]
    )
    for item in summary["draft_files"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "Field checklist:",
            "",
            "| Source | Destination | JSON path | Prompt | Suggested value or hint |",
            "|---|---|---|---|---|",
        ]
    )
    for row in field_rows:
        lines.append(
            f"| `{row['source']}` | `{row['destination_file']}` | `{row['json_path']}` | "
            f"{row['prompt']} | {row.get('suggested_value_or_hint', '')} |"
        )
    lines.extend(
        [
            "",
            "After filling the final inputs, run:",
            "",
        ]
    )
    for command in summary["completion_commands"]:
        lines.append(f"- `{command}`")
    return "\n".join(lines) + "\n"


def draft_prefill_fields(root: Path) -> list[str]:
    fields: list[str] = []
    for spec in TEMPLATE_SPECS:
        template = root / spec["template"]
        if not template.exists():
            continue
        payload, prefilled = prefill_draft_payload(root, spec, read_json(template))
        if not payload:
            continue
        fields.extend(f"{spec['draft']}::{field}" for field in prefilled)
    return fields


def copy_drafts(root: Path, out_dir: Path) -> None:
    for spec in TEMPLATE_SPECS:
        source = root / spec["template"]
        destination = out_dir / spec["draft"]
        if not source.exists():
            continue
        payload = read_json(source)
        if payload:
            payload, _ = prefill_draft_payload(root, spec, payload)
            destination.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def write_outputs(root: Path, out_dir: Path) -> dict[str, Any]:
    field_rows = build_field_rows(root)
    handoff_rows = build_handoff_rows(root)
    coverage_rows = build_handoff_input_coverage_rows(handoff_rows)
    summary = build_summary(root, field_rows, handoff_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_drafts(root, out_dir)
    write_second_hardware_manifest(out_dir / "second_hardware_inputs.json", root)
    write_csv(out_dir / "table_final_input_fields.csv", field_rows)
    write_coverage_csv(out_dir / "table_handoff_input_coverage.csv", coverage_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(render_markdown(summary, field_rows), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_outputs(args.root, args.out_dir)
    print(
        f"Wrote final input packet to {args.out_dir} "
        f"(field_rows={summary['field_rows']}, unresolved={summary['unresolved_handoff_rows']})"
    )


if __name__ == "__main__":
    main()
