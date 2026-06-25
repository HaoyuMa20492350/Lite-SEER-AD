from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.config import cfg_first, dataset_category, image_size, load_config
from seer_ad_v2.data.datasets import build_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index anomaly datasets and report train/test/mask counts.")
    p.add_argument("--config", required=True)
    p.add_argument("--category", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_name = cfg_first(cfg, ("dataset.name",), "mvtec")
    root = cfg_first(cfg, ("dataset.root",), "")
    category = dataset_category(cfg, args.category)
    size = image_size(cfg)
    rows = []
    for split in ["train", "test"]:
        ds = build_dataset(dataset_name, root, category, split, size)
        labels = [int(r.label) for r in ds.records]
        masks = [1 for r in ds.records if r.mask_path is not None and r.mask_path.exists()]
        rows.append(
            {
                "dataset": dataset_name,
                "root": root,
                "category": category,
                "split": split,
                "images": len(ds.records),
                "normal": labels.count(0),
                "anomaly": labels.count(1),
                "masks": len(masks),
            }
        )
    out = Path(args.out) if args.out else Path("dataset_index.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "root", "category", "split", "images", "normal", "anomaly", "masks"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
