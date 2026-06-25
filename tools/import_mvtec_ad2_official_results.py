from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.mvtec_ad2_results import (
    combined_results_rows,
    load_result_rows,
    read_csv,
    render_results_section,
    replace_marked_section,
    sha256_file,
    summarize_rows,
    validate_evaluation_metadata,
    validate_local_submission_evidence,
    write_combined_csv,
    write_rows_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import an official MVTec AD 2 benchmark result export and "
            "update the paper-facing evidence."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--submission-id", default="")
    parser.add_argument("--evaluated-at", default="")
    parser.add_argument(
        "--submission-protocol",
        default=(
            "submissions/mvtec_ad2_seed7_model256_metadata/"
            "submission_protocol.json"
        ),
    )
    parser.add_argument(
        "--archive",
        default="submissions/mvtec_ad2_seed7_model256.tar.gz",
    )
    parser.add_argument(
        "--out",
        default="tables/mvtec_ad2_feature_first/official_evaluation.json",
    )
    parser.add_argument(
        "--table-out",
        default="tables/mvtec_ad2_feature_first/table_official_private.csv",
    )
    parser.add_argument(
        "--public-runs",
        default="tables/mvtec_ad2_feature_first/table_public_runs.csv",
    )
    parser.add_argument(
        "--existing-main",
        default=(
            "tables/feature_first_fusion_aggregate_paper_package/"
            "table_main_seed_metrics.csv"
        ),
    )
    parser.add_argument(
        "--combined-table-out",
        default=(
            "tables/feature_first_fusion_aggregate_paper_package/"
            "table_main_seed_metrics_with_ad2.csv"
        ),
    )
    parser.add_argument(
        "--draft",
        default="docs/results_limitations_draft.md",
    )
    parser.add_argument("--update-draft", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    metadata, rows = load_result_rows(input_path)
    identity = validate_evaluation_metadata(
        metadata,
        source_url=args.source_url or None,
        submission_id=args.submission_id or None,
        evaluated_at=args.evaluated_at or None,
    )
    protocol_path = Path(args.submission_protocol)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    archive_path = Path(args.archive)
    validate_local_submission_evidence(protocol, archive_path)
    evaluation = {
        "complete": True,
        "dataset": "mvtec_ad2",
        "evaluation": "official_private_server",
        **identity,
        "raw_export": str(input_path.resolve()),
        "raw_export_sha256": sha256_file(input_path),
        "local_submission_protocol": str(protocol_path.resolve()),
        "local_archive": str(archive_path.resolve()),
        "local_archive_sha256": sha256_file(archive_path),
        "rows": rows,
        "summary": summarize_rows(rows),
        "sota_claim_supported": False,
        "claim_boundary": (
            "Official evaluation is complete. Universal SOTA still requires "
            "a dated, like-for-like leaderboard comparison."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    write_rows_csv(Path(args.table_out), rows)
    public_rows = read_csv(Path(args.public_runs))
    existing_rows = read_csv(Path(args.existing_main))
    combined = combined_results_rows(existing_rows, public_rows, evaluation)
    write_combined_csv(Path(args.combined_table_out), combined)
    section = render_results_section(evaluation)
    (output.parent / "official_results_section.md").write_text(
        section + "\n",
        encoding="utf-8",
    )
    if args.update_draft:
        draft = Path(args.draft)
        updated = replace_marked_section(
            draft.read_text(encoding="utf-8"),
            section,
        )
        draft.write_text(updated, encoding="utf-8")
    print(
        json.dumps(
            {
                "complete": True,
                "rows": len(rows),
                "combined_rows": len(combined),
                "out": str(output),
                "draft_updated": bool(args.update_draft),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
