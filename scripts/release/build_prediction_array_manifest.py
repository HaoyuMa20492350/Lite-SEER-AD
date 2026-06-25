"""Build a manifest for selected prediction arrays without copying them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_prediction_manifest(csv_path: Path, root: Path, limit: int | None = None) -> dict[str, object]:
    rows = _read_rows(csv_path)
    selected = []
    for row in rows:
        run_dir = root / row["heldout_run"]
        prediction_path = run_dir / "predictions.npz"
        threshold_path = run_dir / "pixel_threshold_policy.json"
        entry = {
            "dataset": row["dataset"],
            "category": row["category"],
            "split_seed": row["split_seed"],
            "selected_candidate": row["selected_candidate"],
            "heldout_run": row["heldout_run"],
            "prediction_path": prediction_path.as_posix(),
            "prediction_exists": prediction_path.is_file(),
            "threshold_policy_path": threshold_path.as_posix(),
            "threshold_policy_exists": threshold_path.is_file(),
        }
        if prediction_path.is_file():
            entry["prediction_sha256"] = sha256_file(prediction_path)
            entry["prediction_bytes"] = prediction_path.stat().st_size
        if threshold_path.is_file():
            entry["threshold_policy_sha256"] = sha256_file(threshold_path)
            entry["threshold_policy_bytes"] = threshold_path.stat().st_size
        selected.append(entry)
        if limit is not None and len(selected) >= limit:
            break

    return {
        "schema": "lite-seer-ad-selected-prediction-manifest-v1",
        "source_csv": csv_path.as_posix(),
        "root": root.resolve().as_posix(),
        "entry_count": len(selected),
        "all_predictions_present": all(entry["prediction_exists"] for entry in selected),
        "all_threshold_policies_present": all(entry["threshold_policy_exists"] for entry in selected),
        "entries": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("tables/strict_fixed_threshold/strict_selected_metrics.csv"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("artifacts/predictions_manifest.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest = build_prediction_manifest(args.csv, args.root, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest['entry_count']} entries to {args.out}")


if __name__ == "__main__":
    main()
