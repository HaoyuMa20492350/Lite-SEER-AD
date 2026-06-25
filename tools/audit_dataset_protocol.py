from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.data.datasets import build_dataset


DATASETS = {
    "mvtec": {
        "config": Path("configs/mvtec.yaml"),
        "expected_test": 1725,
    },
    "visa": {
        "config": Path("configs/visa.yaml"),
        "expected_train": 8659,
        "expected_test": 2162,
    },
    "mpdd": {
        "config": Path("configs/mpdd.yaml"),
        "expected_test": 458,
    },
}
FULL_DTD_ROOT = Path(
    "SEER-AD-dataset/DTD/Describable-Textures-Dataset-DTD/images"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit official dataset splits before full-test experiments."
    )
    parser.add_argument(
        "--out",
        default="tables/full_test_protocol_audit.json",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def categories(cfg: dict[str, Any]) -> list[str]:
    dataset = cfg.get("dataset", {}) or {}
    root = REPO_ROOT / str(dataset["root"])
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in {"archives", "split_csv"}
    )


def norm_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").lower()


def audit_dataset(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    config_path = REPO_ROOT / spec["config"]
    cfg = load_config(config_path)
    dataset_cfg = cfg.get("dataset", {}) or {}
    dataset_name = str(dataset_cfg["name"])
    root = str(dataset_cfg["root"])
    category_rows: list[dict[str, Any]] = []
    all_train: set[str] = set()
    all_test: set[str] = set()
    missing_images: list[str] = []
    missing_masks: list[str] = []

    for category in categories(cfg):
        train = build_dataset(dataset_name, root, category, "train", 256)
        test = build_dataset(dataset_name, root, category, "test", 256)
        train_paths = {norm_path(row.image_path) for row in train.records}
        test_paths = {norm_path(row.image_path) for row in test.records}
        overlap = sorted(train_paths & test_paths)
        all_train.update(train_paths)
        all_test.update(test_paths)
        for row in [*train.records, *test.records]:
            if not row.image_path.exists():
                missing_images.append(str(row.image_path))
        for row in test.records:
            if row.label and (row.mask_path is None or not row.mask_path.exists()):
                missing_masks.append(str(row.image_path))
        category_rows.append(
            {
                "category": category,
                "train_images": len(train.records),
                "test_images": len(test.records),
                "test_normal": sum(row.label == 0 for row in test.records),
                "test_anomaly": sum(row.label == 1 for row in test.records),
                "train_test_overlap": len(overlap),
                "overlap_paths": overlap,
            }
        )

    expected_train = spec.get("expected_train")
    expected_test = int(spec["expected_test"])
    return {
        "dataset": name,
        "config": str(spec["config"]),
        "categories": len(category_rows),
        "train_images": len(all_train),
        "test_images": len(all_test),
        "expected_train_images": expected_train,
        "expected_test_images": expected_test,
        "train_count_matches": (
            expected_train is None or len(all_train) == int(expected_train)
        ),
        "test_count_matches": len(all_test) == expected_test,
        "train_test_overlap": len(all_train & all_test),
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "category_rows": category_rows,
        "protocol_pass": (
            (expected_train is None or len(all_train) == int(expected_train))
            and len(all_test) == expected_test
            and not (all_train & all_test)
            and not missing_images
            and not missing_masks
        ),
    }


def main() -> None:
    args = parse_args()
    dataset_rows = [
        audit_dataset(name, spec) for name, spec in DATASETS.items()
    ]
    dtd_root = REPO_ROOT / FULL_DTD_ROOT
    dtd_images = (
        sum(
            path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            for path in dtd_root.rglob("*")
            if path.is_file()
        )
        if dtd_root.exists()
        else 0
    )
    payload = {
        "datasets": dataset_rows,
        "total_categories": sum(row["categories"] for row in dataset_rows),
        "total_test_images": sum(row["test_images"] for row in dataset_rows),
        "expected_total_categories": 33,
        "expected_total_test_images": 4345,
        "dtd": {
            "role": "synthetic_anomaly_texture_bank",
            "root": str(FULL_DTD_ROOT),
            "images": dtd_images,
            "expected_images": 5640,
            "complete": dtd_images == 5640,
        },
    }
    payload["protocol_pass"] = (
        all(row["protocol_pass"] for row in dataset_rows)
        and payload["total_categories"] == payload["expected_total_categories"]
        and payload["total_test_images"] == payload["expected_total_test_images"]
        and payload["dtd"]["complete"]
    )
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["protocol_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
