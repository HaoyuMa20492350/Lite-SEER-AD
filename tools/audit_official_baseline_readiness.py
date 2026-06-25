from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.official_sources import (
    load_official_source_manifest,
    validate_official_provenance,
)


DATASET_CATEGORIES = {
    "mvtec15": (
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ),
    "visa": (
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ),
    "mpdd": (
        "bracket_black",
        "bracket_brown",
        "bracket_white",
        "connector",
        "metal_plate",
        "tubes",
    ),
}
PREDICTION_KEYS = ("labels", "image_scores", "masks", "heatmaps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit pinned official baseline sources and exported prediction "
            "artifacts before paper-table import."
        )
    )
    parser.add_argument(
        "--manifest",
        default="baselines/official_sources.json",
    )
    parser.add_argument(
        "--external-root",
        default="baselines/external_outputs",
    )
    parser.add_argument(
        "--datasets",
        default="mvtec15,visa,mpdd",
    )
    parser.add_argument(
        "--out",
        default="tables/official_baseline_readiness",
    )
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _artifact_dir(
    root: Path,
    dataset: str,
    method: str,
    category: str,
) -> Path:
    return root / dataset / method / category


def _validate_prediction(path: Path) -> list[str]:
    if not path.exists():
        return ["predictions.npz missing"]
    try:
        data = np.load(path)
    except Exception as exc:
        return [f"predictions.npz unreadable: {exc}"]
    missing = [key for key in PREDICTION_KEYS if key not in data.files]
    if missing:
        return [f"prediction arrays missing: {', '.join(missing)}"]
    labels = np.asarray(data["labels"]).reshape(-1)
    scores = np.asarray(data["image_scores"]).reshape(-1)
    masks = np.asarray(data["masks"])
    heatmaps = np.asarray(data["heatmaps"])
    errors = []
    if not (len(labels) == len(scores) == len(masks) == len(heatmaps)):
        errors.append("prediction image counts do not align")
    if masks.shape != heatmaps.shape:
        errors.append("masks and heatmaps shapes do not align")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(heatmaps)):
        errors.append("prediction scores contain non-finite values")
    return errors


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "method",
        "category",
        "source_kind",
        "source_repository",
        "source_commit",
        "declared_dataset_support",
        "prediction_exists",
        "provenance_exists",
        "ready",
        "errors",
        "artifact_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = load_official_source_manifest(args.manifest)
    sources = manifest["sources"]
    external_root = Path(args.external_root)
    datasets = split_csv(args.datasets)
    unknown = [dataset for dataset in datasets if dataset not in DATASET_CATEGORIES]
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for method, source in sorted(sources.items()):
            declared = dataset in source["declared_datasets"]
            for category in DATASET_CATEGORIES[dataset]:
                artifact_dir = _artifact_dir(
                    external_root,
                    dataset,
                    method,
                    category,
                )
                prediction_path = artifact_dir / "predictions.npz"
                provenance_path = artifact_dir / "provenance.json"
                errors = []
                if not declared:
                    errors.append(
                        "dataset not declared by source; custom adapter required"
                    )
                errors.extend(_validate_prediction(prediction_path))
                provenance = _read_json(provenance_path)
                if not provenance:
                    errors.append("provenance.json missing or unreadable")
                else:
                    errors.extend(
                        validate_official_provenance(
                            provenance,
                            source,
                            method=method,
                            dataset=dataset,
                            category=category,
                        )
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "category": category,
                        "source_kind": source["source_kind"],
                        "source_repository": source["repository"],
                        "source_commit": source["commit"],
                        "declared_dataset_support": declared,
                        "prediction_exists": prediction_path.exists(),
                        "provenance_exists": provenance_path.exists(),
                        "ready": not errors,
                        "errors": "; ".join(errors),
                        "artifact_dir": str(artifact_dir),
                    }
                )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "table_official_baseline_readiness.csv", rows)
    ready = [row for row in rows if row["ready"]]
    declared_rows = [row for row in rows if row["declared_dataset_support"]]
    summary = {
        "manifest": str(Path(args.manifest)),
        "manifest_verified_on": manifest.get("verified_on"),
        "external_root": str(external_root),
        "datasets": datasets,
        "methods": sorted(sources),
        "records": len(rows),
        "declared_support_records": len(declared_rows),
        "ready_records": len(ready),
        "missing_records": len(rows) - len(ready),
        "declared_ready_records": sum(
            bool(row["ready"]) for row in declared_rows
        ),
        "complete": len(ready) == len(rows),
        "paper_eligible_complete": all(
            row["ready"]
            for row in declared_rows
            if sources[row["method"]].get(
                "paper_table_eligible_when_ready",
                False,
            )
        ),
        "by_dataset": {
            dataset: {
                "records": sum(row["dataset"] == dataset for row in rows),
                "ready": sum(
                    row["dataset"] == dataset and bool(row["ready"])
                    for row in rows
                ),
                "declared_support": sum(
                    row["dataset"] == dataset
                    and bool(row["declared_dataset_support"])
                    for row in rows
                ),
            }
            for dataset in datasets
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if args.fail_on_incomplete and not summary["paper_eligible_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
