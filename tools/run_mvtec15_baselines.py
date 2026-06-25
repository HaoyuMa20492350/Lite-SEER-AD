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

from baselines.registry import BASELINES, LOCAL_BASELINES, REQUIRED_MVTEC15_BASELINES

MVTEC15_CATEGORIES = [
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
]
PROVENANCE_FIELDS = [
    "method",
    "method_id",
    "display_method",
    "implementation_variant",
    "official_implementation",
    "source_path",
    "source_url",
    "source_commit",
    "reference_key",
]
MAIN_FIELDS = [
    *PROVENANCE_FIELDS,
    "run",
    "dataset",
    "category",
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "aupro_proxy",
    "pixel_ap",
    "f1",
    "iou",
    "dice",
    "threshold_protocol",
]
EFF_FIELDS = [
    *PROVENANCE_FIELDS,
    "run",
    "category",
    "latency_ms_mean",
    "latency_ms_std",
    "latency_ms_p50",
    "latency_ms_p95",
    "fps",
    "latency_protocol",
    "latency_batch_size",
    "latency_warmups",
    "latency_repeats",
    "nfe_mean",
    "repaired_area_ratio_mean",
    "local_region_ratio_mean",
    "gpu_memory_mb",
]
REQUIRED_FILES = ["predictions.npz", "metrics.json", "metrics.csv", "efficiency.csv", "scores.csv"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run or import baseline adapters on MVTec15.")
    p.add_argument("--methods", default="patchcore,padim")
    p.add_argument("--required-methods", default=",".join(REQUIRED_MVTEC15_BASELINES))
    p.add_argument("--config", default="configs/mvtec.yaml")
    p.add_argument("--categories", default=",".join(MVTEC15_CATEGORIES))
    p.add_argument("--run-prefix", default="mvtec15_baseline")
    p.add_argument("--tables-out", default="tables/mvtec15_baselines")
    p.add_argument("--external-root", default="baselines/external_outputs")
    p.add_argument(
        "--prefer-external",
        action="store_true",
        help=(
            "Use a matching external prediction artifact even when a local "
            "runner exists for the same method."
        ),
    )
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--train-max-samples", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--simplenet-epochs", type=int, default=2)
    p.add_argument("--simplenet-hidden-dim", type=int, default=384)
    p.add_argument("--simplenet-lr", type=float, default=1e-3)
    p.add_argument("--simplenet-noise-std", type=float, default=0.05)
    p.add_argument("--simplenet-max-patches", type=int, default=4096)
    p.add_argument("--draem-epochs", type=int, default=2)
    p.add_argument("--draem-lr", type=float, default=2e-4)
    p.add_argument("--draem-base-channels", type=int, default=16)
    p.add_argument("--draem-recon-weight", type=float, default=1.0)
    p.add_argument("--draem-seg-weight", type=float, default=1.0)
    p.add_argument("--uniad-epochs", type=int, default=2)
    p.add_argument("--uniad-lr", type=float, default=2e-4)
    p.add_argument("--uniad-patch-size", type=int, default=16)
    p.add_argument("--uniad-embed-dim", type=int, default=64)
    p.add_argument("--uniad-depth", type=int, default=2)
    p.add_argument("--uniad-heads", type=int, default=4)
    p.add_argument("--diffusionad-epochs", type=int, default=2)
    p.add_argument("--diffusionad-lr", type=float, default=2e-4)
    p.add_argument("--diffusionad-base-channels", type=int, default=16)
    p.add_argument("--diffusionad-timesteps", type=int, default=50)
    p.add_argument("--diffusionad-score-timestep", type=int, default=25)
    p.add_argument("--ddad-epochs", type=int, default=2)
    p.add_argument("--ddad-lr", type=float, default=2e-4)
    p.add_argument("--ddad-base-channels", type=int, default=16)
    p.add_argument("--ddad-timesteps", type=int, default=50)
    p.add_argument("--ddad-low-timestep", type=int, default=12)
    p.add_argument("--ddad-high-timestep", type=int, default=38)
    p.add_argument("--rd4ad-epochs", type=int, default=2)
    p.add_argument("--rd4ad-lr", type=float, default=1e-4)
    p.add_argument("--latency-warmups", type=int, default=50)
    p.add_argument("--latency-repeats", type=int, default=200)
    p.add_argument("--latency-batch-size", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--allow-random-weights", action="store_true")
    return p.parse_args()


def split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def dataset_name(config: str) -> str:
    cfg = _load_yaml(REPO_ROOT / config)
    name = str((cfg.get("dataset", {}) or {}).get("name", "dataset"))
    return {"mvtec": "mvtec15", "mvtec_ad": "mvtec15"}.get(name, name)


def discover_categories(config: str, requested: str) -> list[str]:
    if requested != "all":
        return split_csv(requested)
    cfg = _load_yaml(REPO_ROOT / config)
    dataset = cfg.get("dataset", {}) or {}
    categories = dataset.get("categories")
    if isinstance(categories, list):
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


def py(script: str, *args: Any) -> list[str]:
    return [sys.executable, script, *[str(a) for a in args]]


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_if_needed(cmd: list[str], output: Path, *, resume: bool, dry_run: bool) -> None:
    if resume and output.exists():
        print(f"SKIP existing {output}", flush=True)
        return
    run_command(cmd, dry_run=dry_run)


def external_prediction_path(
    root: Path,
    dataset: str,
    method: str,
    category: str,
) -> Path | None:
    candidates = [
        root / dataset / method / category / "predictions.npz",
        root / dataset / method / f"{category}.npz",
        root / method / category / "predictions.npz",
        root / method / f"{category}.npz",
        root / f"{method}_{category}" / "predictions.npz",
        root / f"{method}_{category}.npz",
    ]
    return next((path for path in candidates if path.exists()), None)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metric_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row.get("metric", ""): row.get("value", "") for row in csv.DictReader(f) if row.get("metric")}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def check_schema(run_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    heatmap_count = len(list((run_dir / "heatmaps").glob("*.png"))) if (run_dir / "heatmaps").exists() else 0
    return {"run": run_dir.name, "ok": not missing and heatmap_count > 0, "missing": missing, "heatmaps": heatmap_count}


def _row(run_dir: Path) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    eff = _load_metric_csv(run_dir / "efficiency.csv")
    args = _load_json(run_dir / "run_args.json")
    cfg = _load_yaml(run_dir / "config.yaml")
    arg_payload = args.get("args", {}) if isinstance(args, dict) else {}
    method = arg_payload.get("method", metrics.get("method", ""))
    spec = BASELINES.get(str(method))
    row = {
        "method": method,
        "method_id": metrics.get("method_id", method),
        "display_method": metrics.get(
            "display_method",
            spec.display_name if spec else method,
        ),
        "implementation_variant": metrics.get(
            "implementation_variant",
            spec.implementation_variant if spec else "",
        ),
        "official_implementation": metrics.get(
            "official_implementation",
            spec.official_implementation if spec else False,
        ),
        "source_path": metrics.get(
            "source_path",
            (
                metrics.get("provenance_path", "")
                if metrics.get("external_baseline")
                else (spec.source_path if spec else "")
            ),
        ),
        "source_url": metrics.get("source_url", ""),
        "source_commit": metrics.get("source_commit", ""),
        "reference_key": metrics.get(
            "reference_key",
            spec.reference_key if spec else "",
        ),
        "run": run_dir.name,
        "dataset": (cfg.get("dataset", {}) or {}).get("name", ""),
        "category": arg_payload.get("category", metrics.get("category", "")),
    }
    row.update(metrics)
    row.update(eff)
    return row


def _write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_tables(run_dirs: list[Path], out_dir: Path, dataset: str) -> None:
    rows = [_row(run_dir) for run_dir in run_dirs if (run_dir / "metrics.json").exists()]
    _write(out_dir / "table_main_mvtec15.csv", rows, MAIN_FIELDS)
    _write(out_dir / "table_efficiency_mvtec15.csv", rows, EFF_FIELDS)
    if dataset and dataset != "mvtec":
        _write(out_dir / f"table_main_{dataset}.csv", rows, MAIN_FIELDS)
        _write(out_dir / f"table_efficiency_{dataset}.csv", rows, EFF_FIELDS)


def write_coverage(tables_dir: Path, required_methods: list[str], categories: list[str]) -> None:
    rows = _read_csv(tables_dir / "table_main_mvtec15.csv")
    present = {(str(row.get("method", "")), str(row.get("category", ""))): str(row.get("run", "")) for row in rows}
    coverage = []
    by_method: dict[str, dict[str, int]] = {}
    for method in required_methods:
        count = 0
        for category in categories:
            key = (method, category)
            found = key in present
            count += int(found)
            coverage.append({"method": method, "category": category, "present": found, "run": present.get(key, "")})
        by_method[method] = {"present": count, "total": len(categories), "missing": len(categories) - count}
    missing = [row for row in coverage if not bool(row["present"])]
    _write(tables_dir / "baseline_coverage.csv", coverage, ["method", "category", "present", "run"])
    (tables_dir / "baseline_coverage.json").write_text(
        json.dumps(
            {
                "required_methods": required_methods,
                "categories": categories,
                "total_required": len(required_methods) * len(categories),
                "present": len(coverage) - len(missing),
                "missing": len(missing),
                "complete": not missing,
                "by_method": by_method,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    methods = split_csv(args.methods)
    unknown = [method for method in methods if method not in BASELINES]
    if unknown:
        raise SystemExit(f"Unknown baseline method(s): {', '.join(unknown)}")
    required_methods = split_csv(args.required_methods)
    categories = discover_categories(args.config, args.categories)
    dataset = dataset_name(args.config)
    report: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    run_dirs: list[Path] = []
    for method in methods:
        for category in categories:
            run_name = f"{args.run_prefix}_{method}_{category}"
            run_dir = REPO_ROOT / "runs" / run_name
            try:
                external_source = external_prediction_path(
                    REPO_ROOT / args.external_root,
                    dataset,
                    method,
                    category,
                )
                use_external = bool(
                    external_source is not None
                    and (args.prefer_external or method not in LOCAL_BASELINES)
                )
                if method in LOCAL_BASELINES and not use_external:
                    cmd = py(
                        "baselines/run_baseline.py",
                        "--method",
                        method,
                        "--config",
                        args.config,
                        "--category",
                        category,
                        "--image-size",
                        args.image_size,
                        "--batch-size",
                        args.batch_size,
                        "--seed",
                        args.seed,
                        "--device",
                        args.device,
                        "--run-name",
                        run_name,
                        "--latency-warmups",
                        args.latency_warmups,
                        "--latency-repeats",
                        args.latency_repeats,
                        "--latency-batch-size",
                        args.latency_batch_size,
                    )
                    if args.train_max_samples is not None:
                        cmd.extend(["--train-max-samples", str(args.train_max_samples)])
                    if args.max_samples is not None:
                        cmd.extend(["--max-samples", str(args.max_samples)])
                    if method == "simplenet":
                        cmd.extend(
                            [
                                "--simplenet-epochs",
                                str(args.simplenet_epochs),
                                "--simplenet-hidden-dim",
                                str(args.simplenet_hidden_dim),
                                "--simplenet-lr",
                                str(args.simplenet_lr),
                                "--simplenet-noise-std",
                                str(args.simplenet_noise_std),
                                "--simplenet-max-patches",
                                str(args.simplenet_max_patches),
                            ]
                        )
                    if method == "draem":
                        cmd.extend(
                            [
                                "--draem-epochs",
                                str(args.draem_epochs),
                                "--draem-lr",
                                str(args.draem_lr),
                                "--draem-base-channels",
                                str(args.draem_base_channels),
                                "--draem-recon-weight",
                                str(args.draem_recon_weight),
                                "--draem-seg-weight",
                                str(args.draem_seg_weight),
                            ]
                        )
                    if method == "uniad":
                        cmd.extend(
                            [
                                "--uniad-epochs",
                                str(args.uniad_epochs),
                                "--uniad-lr",
                                str(args.uniad_lr),
                                "--uniad-patch-size",
                                str(args.uniad_patch_size),
                                "--uniad-embed-dim",
                                str(args.uniad_embed_dim),
                                "--uniad-depth",
                                str(args.uniad_depth),
                                "--uniad-heads",
                                str(args.uniad_heads),
                            ]
                        )
                    if method == "diffusionad":
                        cmd.extend(
                            [
                                "--diffusionad-epochs",
                                str(args.diffusionad_epochs),
                                "--diffusionad-lr",
                                str(args.diffusionad_lr),
                                "--diffusionad-base-channels",
                                str(args.diffusionad_base_channels),
                                "--diffusionad-timesteps",
                                str(args.diffusionad_timesteps),
                                "--diffusionad-score-timestep",
                                str(args.diffusionad_score_timestep),
                            ]
                        )
                    if method == "ddad":
                        cmd.extend(
                            [
                                "--ddad-epochs",
                                str(args.ddad_epochs),
                                "--ddad-lr",
                                str(args.ddad_lr),
                                "--ddad-base-channels",
                                str(args.ddad_base_channels),
                                "--ddad-timesteps",
                                str(args.ddad_timesteps),
                                "--ddad-low-timestep",
                                str(args.ddad_low_timestep),
                                "--ddad-high-timestep",
                                str(args.ddad_high_timestep),
                            ]
                        )
                    if method == "rd4ad":
                        cmd.extend(
                            [
                                "--rd4ad-epochs",
                                str(args.rd4ad_epochs),
                                "--rd4ad-lr",
                                str(args.rd4ad_lr),
                            ]
                        )
                    if args.allow_random_weights:
                        cmd.append("--allow-random-weights")
                else:
                    source = external_source
                    if source is None:
                        raise FileNotFoundError(
                            f"Missing external predictions for {method}/{category}. Expected one of "
                            f"{args.external_root}/{method}/{category}/predictions.npz, "
                            f"{args.external_root}/{method}/{category}.npz, "
                            f"{args.external_root}/{method}_{category}/predictions.npz, or "
                            f"{args.external_root}/{method}_{category}.npz"
                        )
                    cmd = py(
                        "tools/import_external_baseline.py",
                        "--method",
                        method,
                        "--predictions",
                        source,
                        "--config",
                        args.config,
                        "--category",
                        category,
                        "--image-size",
                        args.image_size,
                        "--device",
                        args.device,
                        "--run-name",
                        run_name,
                        "--dataset-id",
                        dataset,
                    )
                    source_metrics = _load_json(source.parent / "metrics.json")
                    if source_metrics.get("latency_ms_mean") is not None:
                        cmd.extend(
                            [
                                "--latency-ms-mean",
                                str(source_metrics["latency_ms_mean"]),
                            ]
                        )
                    provenance = source.parent / "provenance.json"
                    if provenance.exists():
                        cmd.extend(["--provenance", str(provenance)])
                run_if_needed(cmd, run_dir / "predictions.npz", resume=args.resume, dry_run=args.dry_run)
                run_command(py("evaluate.py", "--pred_dir", run_dir, "--out", run_dir / "eval_metrics.json"), dry_run=args.dry_run)
                report.append(check_schema(run_dir))
                run_dirs.append(run_dir)
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                failures.append({"method": method, "category": category, "error": str(exc)})
                if args.fail_fast:
                    raise
    report_path = REPO_ROOT / "runs" / f"{args.run_prefix}_report.json"
    if not args.dry_run:
        report_path.write_text(json.dumps({"schema": report, "failures": failures}, indent=2), encoding="utf-8")
        write_tables(run_dirs, REPO_ROOT / args.tables_out, dataset)
        write_coverage(REPO_ROOT / args.tables_out, required_methods, categories)
    print(f"Finished baselines. Report: {report_path}. Failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
