from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.prediction_schema import (
    CANONICAL_HEATMAP_KEYS,
    prediction_heatmap_payload,
    resolve_prediction_heatmaps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add explicit detection/verification/image-score heatmap keys."
    )
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def upgrade(path: Path) -> str:
    with np.load(path, allow_pickle=False) as source:
        if all(key in source.files for key in CANONICAL_HEATMAP_KEYS):
            return "already_current"
        arrays = {key: source[key] for key in source.files}
    detection, verification, image_score = resolve_prediction_heatmaps(arrays)
    arrays.update(
        prediction_heatmap_payload(detection, verification, image_score)
    )
    temporary = path.with_name(f".{path.stem}.schema_tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    return "upgraded"


def main() -> None:
    args = parse_args()
    paths: set[Path] = set()
    for value in args.roots:
        root = Path(value)
        if root.is_file() and root.name == "predictions.npz":
            paths.add(root)
        elif root.exists():
            paths.update(root.rglob("predictions.npz"))
    rows = [
        {"path": str(path), "status": upgrade(path)}
        for path in sorted(paths)
    ]
    payload = {
        "files": len(rows),
        "upgraded": sum(row["status"] == "upgraded" for row in rows),
        "already_current": sum(
            row["status"] == "already_current" for row in rows
        ),
        "canonical_keys": list(CANONICAL_HEATMAP_KEYS),
        "rows": rows,
    }
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
