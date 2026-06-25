from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from tools.run_mvtec_mini import (
    FULL_ABLATIONS,
    MVTEC15_ABLATIONS,
    REPO_ROOT,
    SMOKE_ABLATIONS,
    check_schema,
    py,
    run_category,
    run_command,
    split_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a Lite-SEER-AD mini experiment for a dataset config.")
    p.add_argument("--config", required=True)
    p.add_argument("--categories", default=None, help="Comma-separated categories. Defaults to dataset.categories or discovered dataset folders.")
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--tables-out", required=True)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--max-samples", default="64")
    p.add_argument("--diffusion-epochs", type=int, default=3)
    p.add_argument("--sev-epochs", type=int, default=2)
    p.add_argument("--scheduler-epochs", type=int, default=10)
    p.add_argument("--scheduler-samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sev-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--crv-weight", type=float, default=0.35)
    p.add_argument("--ablations", default=None)
    p.add_argument("--qualitative-limit", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def discover_categories(cfg: dict[str, Any]) -> list[str]:
    dataset = cfg.get("dataset", {}) or {}
    categories = dataset.get("categories")
    if isinstance(categories, list) and categories:
        return [str(item) for item in categories]
    if isinstance(categories, str) and categories != "all":
        return split_csv(categories)

    root = REPO_ROOT / str(dataset.get("root", ""))
    if not root.exists():
        category = dataset.get("category")
        if isinstance(category, str) and category:
            return [category]
        raise FileNotFoundError(f"Cannot discover categories because dataset root does not exist: {root}")
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    return [name for name in names if name not in {"split_csv", "archives"}]


def write_dataset_table_aliases(tables_dir: Path, dataset_name: str) -> None:
    aliases = {
        "table_main_mvtec5.csv": f"table_main_{dataset_name}.csv",
        "table_efficiency_mvtec5.csv": f"table_efficiency_{dataset_name}.csv",
    }
    for src_name, dst_name in aliases.items():
        src = tables_dir / src_name
        dst = tables_dir / dst_name
        if src.exists():
            shutil.copyfile(src, dst)


def main() -> None:
    args = parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    dataset_name = str((cfg.get("dataset", {}) or {}).get("name", "dataset"))

    if args.categories:
        categories = split_csv(args.categories)
    else:
        categories = discover_categories(cfg)

    if args.smoke:
        categories = categories[:1]
        if not args.run_prefix.endswith("_smoke"):
            args.run_prefix = f"{args.run_prefix}_smoke"
        args.image_size = 64
        args.max_samples = "8"
        args.diffusion_epochs = 1
        args.sev_epochs = 1
        args.scheduler_epochs = min(args.scheduler_epochs, 3)
        args.scheduler_samples = min(args.scheduler_samples, 128)

    if args.ablations:
        ablations = split_csv(args.ablations)
    elif args.smoke:
        ablations = SMOKE_ABLATIONS
    elif len(categories) > 5:
        ablations = MVTEC15_ABLATIONS
    else:
        ablations = FULL_ABLATIONS

    report: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for category in categories:
        try:
            index_out = f"runs/index_{dataset_name}_{category}.csv"
            run_command(py("tools/index_datasets.py", "--config", args.config, "--category", category, "--out", index_out), dry_run=args.dry_run)
            run_category(category, args, ablations, report)
        except subprocess.CalledProcessError as exc:
            failures.append({"category": category, "error": str(exc)})
            if args.fail_fast:
                raise

    report_path = REPO_ROOT / "runs" / f"{args.run_prefix}_report.json"
    if not args.dry_run:
        report_path.write_text(json.dumps({"schema": report, "failures": failures}, indent=2), encoding="utf-8")

    summary_cmd = py("tools/summarize_evidence.py", "--runs", "runs", "--prefix", args.run_prefix, "--out", args.tables_out)
    if not args.smoke:
        summary_cmd.extend(["--exclude-prefix", f"{args.run_prefix}_smoke"])
    summary_cmd.extend(["--include-ablations", ",".join(["full", *ablations])])
    run_command(summary_cmd, dry_run=args.dry_run)
    if not args.dry_run:
        write_dataset_table_aliases(REPO_ROOT / args.tables_out, dataset_name)
    print(f"Finished {dataset_name} mini experiment. Report: {report_path}. Failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
