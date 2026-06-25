from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.data.mvtec_ad2_install import (
    install_archive,
    install_category_archives,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely install an official MVTec AD 2 ZIP/TAR archive and "
            "validate all required dataset paths."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive")
    source.add_argument("--category-archives-dir")
    parser.add_argument(
        "--out",
        default="SEER-AD-dataset/MVTec-AD2",
    )
    parser.add_argument(
        "--report",
        default="tables/mvtec_ad2_feature_first_readiness/install.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.category_archives_dir:
        report = install_category_archives(
            Path(args.category_archives_dir),
            Path(args.out),
        )
    else:
        report = install_archive(Path(args.archive), Path(args.out))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
