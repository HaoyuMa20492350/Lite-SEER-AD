from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.data.mvtec_ad2_discovery import discover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find an MVTec AD 2 directory or archive without extracting it."
    )
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument(
        "--out",
        default="tables/mvtec_ad2_feature_first_readiness/discovery.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [
        Path(value)
        for value in (
            args.root
            or [
                REPO_ROOT / "SEER-AD-dataset",
                Path.home() / "Downloads",
                Path.home() / "Desktop",
                Path.home() / "Documents",
            ]
        )
    ]
    report = discover(roots, max_depth=args.max_depth)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ready": report["ready"],
                "results": len(report["results"]),
                "out": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
