from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.mvtec_ad2_results import (
    evaluate_leaderboard,
    load_leaderboard_rows,
    render_leaderboard_claim,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a dated official MVTec AD 2 leaderboard snapshot before "
            "making any SOTA claim."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--evaluation",
        default="tables/mvtec_ad2_feature_first/official_evaluation.json",
    )
    parser.add_argument("--source-url", default="")
    parser.add_argument("--captured-at", default="")
    parser.add_argument(
        "--out",
        default="tables/mvtec_ad2_feature_first/leaderboard_evidence.json",
    )
    parser.add_argument(
        "--claim-out",
        default="tables/mvtec_ad2_feature_first/leaderboard_claim.md",
    )
    parser.add_argument(
        "--update-evaluation",
        action="store_true",
        help=(
            "Write the audited leaderboard decision back into the official "
            "evaluation JSON."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_path = Path(args.evaluation)
    if not evaluation_path.is_file():
        raise FileNotFoundError(evaluation_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("complete") is not True:
        raise ValueError("Official evaluation must be complete first.")
    snapshot = Path(args.input)
    metadata, rows = load_leaderboard_rows(snapshot)
    source_url = str(args.source_url or metadata.get("source_url", "")).strip()
    captured_at = str(
        args.captured_at or metadata.get("captured_at", "")
    ).strip()
    result = evaluate_leaderboard(
        evaluation,
        rows,
        source_url=source_url,
        captured_at=captured_at,
    )
    result.update(
        {
            "complete": True,
            "snapshot": str(snapshot.resolve()),
            "snapshot_sha256": sha256_file(snapshot),
            "official_evaluation": str(evaluation_path.resolve()),
            "official_evaluation_sha256": sha256_file(evaluation_path),
        }
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    claim_output = Path(args.claim_out)
    claim_output.parent.mkdir(parents=True, exist_ok=True)
    claim_output.write_text(
        render_leaderboard_claim(result) + "\n",
        encoding="utf-8",
    )
    if args.update_evaluation:
        evaluation["leaderboard_evidence"] = str(output.resolve())
        evaluation["leaderboard_evidence_sha256"] = sha256_file(output)
        evaluation["sota_claim_supported"] = result[
            "sota_claim_supported"
        ]
        evaluation["claim_boundary"] = result["supported_claim"]
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
