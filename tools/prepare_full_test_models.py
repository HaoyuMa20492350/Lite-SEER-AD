from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy reusable diffusion and feature-prior checkpoints into a new "
            "full-test model namespace. HN-SEV is intentionally not copied."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--target-prefix", required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def discover_categories(cfg: dict[str, Any]) -> list[str]:
    dataset = cfg.get("dataset", {}) or {}
    root = REPO_ROOT / str(dataset["root"])
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name not in {"archives", "split_csv"}
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    rows: list[dict[str, Any]] = []
    for category in discover_categories(cfg):
        source = REPO_ROOT / "runs" / f"{args.source_prefix}_{category}_models"
        target = REPO_ROOT / "runs" / f"{args.target_prefix}_{category}_models"
        target.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for name in ("diffusion.pt", "feature_prior.pt"):
            src = source / name
            dst = target / name
            if not src.exists():
                raise FileNotFoundError(f"Missing reusable checkpoint: {src}")
            if not dst.exists():
                shutil.copy2(src, dst)
                copied.append(name)
        rows.append(
            {
                "category": category,
                "source": str(source),
                "target": str(target),
                "copied": copied,
                "hn_sev_reused": False,
                "reason": "HN-SEV must be rebuilt with full official train data and full DTD.",
            }
        )
    manifest = (
        REPO_ROOT
        / "runs"
        / f"{args.target_prefix}_reused_model_manifest.json"
    )
    manifest.write_text(json.dumps({"models": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "categories": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
