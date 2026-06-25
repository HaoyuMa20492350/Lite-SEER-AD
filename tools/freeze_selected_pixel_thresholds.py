from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.pixel_threshold_policy import (
    load_pixel_threshold_policy,
    save_pixel_threshold_policy,
)
from tools.freeze_pixel_threshold_policy import freeze_run_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze synthetic-normal pixel thresholds for selected held-out "
            "runs and evaluate them without test-GT threshold selection."
        )
    )
    parser.add_argument(
        "--selection-root",
        action="append",
        default=[],
        help="Directory recursively containing selection.csv files.",
    )
    parser.add_argument(
        "--selection",
        action="append",
        default=[],
        help="Explicit selection.csv path.",
    )
    parser.add_argument("--out-dir", default="tables/strict_fixed_threshold")
    parser.add_argument("--max-normal-fpr", type=float, default=0.005)
    parser.add_argument("--max-candidates", type=int, default=2048)
    parser.add_argument("--refresh-policies", action="store_true")
    return parser.parse_args()


def _selection_paths(args: argparse.Namespace) -> list[Path]:
    paths = {Path(value) for value in args.selection}
    for root_value in args.selection_root:
        root = Path(root_value)
        paths.update(root.rglob("selection.csv"))
    existing = sorted(path for path in paths if path.exists())
    if not existing:
        raise FileNotFoundError("No selection.csv files were found")
    return existing


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _parse_seeds(row: dict[str, str]) -> list[int]:
    raw = row.get("selection_evidence_seeds", "")
    seeds = [int(value) for value in raw.replace(",", " ").split()]
    if not seeds:
        raise ValueError(
            f"{row.get('category', '<unknown>')} has no threshold evidence seeds"
        )
    return seeds


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _heldout_run(selection_path: Path, row: dict[str, str]) -> Path:
    return (
        selection_path.parent
        / "selected_runs"
        / f"{row['category']}_{row['selected_candidate']}_heldout"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cached: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for selection_path in _selection_paths(args):
        for row in _read_rows(selection_path):
            source_run = _repo_path(row["selected_run"])
            heldout_run = _heldout_run(selection_path, row)
            if not (heldout_run / "predictions.npz").exists():
                raise FileNotFoundError(
                    f"Missing held-out predictions: {heldout_run / 'predictions.npz'}"
                )
            seeds = _parse_seeds(row)
            cache_key = (str(source_run.resolve()), tuple(seeds))
            source_policy_path = source_run / "pixel_threshold_policy.json"
            if cache_key not in cached:
                if source_policy_path.exists() and not args.refresh_policies:
                    policy = load_pixel_threshold_policy(source_policy_path)
                else:
                    policy = freeze_run_policy(
                        source_run,
                        seeds,
                        max_normal_fpr=args.max_normal_fpr,
                        max_candidates=args.max_candidates,
                    )
                    save_pixel_threshold_policy(policy, source_policy_path)
                cached[cache_key] = policy
            policy = dict(cached[cache_key])
            policy["selected_heldout_run"] = str(heldout_run)
            heldout_policy_path = heldout_run / "pixel_threshold_policy.json"
            save_pixel_threshold_policy(policy, heldout_policy_path)

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "evaluate.py"),
                    "--pred_dir",
                    str(heldout_run),
                    "--pixel-threshold-policy",
                    str(heldout_policy_path),
                    "--require-fixed-threshold",
                ],
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            metrics = json.loads(
                (heldout_run / "metrics.json").read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "dataset": row.get("dataset", ""),
                    "split_seed": selection_path.parent.name,
                    "category": row["category"],
                    "selected_candidate": row["selected_candidate"],
                    "source_run": str(source_run),
                    "heldout_run": str(heldout_run),
                    "threshold": metrics.get("threshold"),
                    "threshold_protocol": metrics.get("threshold_protocol"),
                    "normal_pixel_fpr": policy.get(
                        "observed_normal_pixel_fpr"
                    ),
                    "image_auroc": metrics.get("image_auroc"),
                    "pixel_auroc": metrics.get("pixel_auroc"),
                    "aupro": metrics.get("aupro"),
                    "pixel_ap": metrics.get("pixel_ap"),
                    "f1": metrics.get("f1"),
                    "iou": metrics.get("iou"),
                    "dice": metrics.get("dice"),
                    "oracle_f1": metrics.get("oracle_f1"),
                    "oracle_iou": metrics.get("oracle_iou"),
                    "oracle_dice": metrics.get("oracle_dice"),
                    "uses_real_anomaly_labels_for_threshold": policy.get(
                        "uses_real_anomaly_labels"
                    ),
                    "uses_real_anomaly_masks_for_threshold": policy.get(
                        "uses_real_anomaly_masks"
                    ),
                }
            )

    _write_csv(out_dir / "strict_selected_metrics.csv", rows)
    summary = {
        "selection_files": len(_selection_paths(args)),
        "evaluated_runs": len(rows),
        "unique_threshold_policies": len(cached),
        "max_normal_pixel_fpr": args.max_normal_fpr,
        "all_fixed_threshold": all(
            row["threshold_protocol"] == "synthetic_normal_fixed_threshold_v1"
            for row in rows
        ),
        "uses_real_anomaly_labels_for_threshold": any(
            bool(row["uses_real_anomaly_labels_for_threshold"]) for row in rows
        ),
        "uses_real_anomaly_masks_for_threshold": any(
            bool(row["uses_real_anomaly_masks_for_threshold"]) for row in rows
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
