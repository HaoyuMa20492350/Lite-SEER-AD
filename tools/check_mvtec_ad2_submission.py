from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.mvtec_ad2_submission import (
    assert_submission_root,
    create_submission_archive,
    run_official_checker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and optionally archive an MVTec AD 2 submission."
    )
    parser.add_argument("submission")
    parser.add_argument(
        "--official-checker",
        default=(
            "official_mvtec_ad2_utils/MVTecAD2_public_code_utils/"
            "check_and_prepare_data_for_upload.py"
        ),
    )
    parser.add_argument("--archive-out", default=None)
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission = Path(args.submission)
    assert_submission_root(submission)
    result = run_official_checker(
        submission,
        Path(args.official_checker),
    )
    if args.archive_out:
        result["archive"] = str(
            create_submission_archive(submission, Path(args.archive_out))
        )
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
