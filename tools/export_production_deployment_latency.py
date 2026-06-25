"""Export production inference latency evidence from saved inference runs.

The regular deployment latency audit measures synchronized smoke callables.
This exporter instead reads real ``infer.py`` outputs: per-image component
latency CSVs plus ROI budget logs. It is still conservative: historical
single-budget inference logs can prove production component timing, but they
do not become the final six-budget, multi-action LC-RDS sweep unless the logs
cover the required action space and budget protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BUDGETS_MS = [10, 25, 50, 75, 100, 150]
REQUIRED_ACTIONS = ["skip", "repair-5", "repair-10", "repair-25", "native-refine"]
COMPONENT_FIELDS = [
    ("detector", "detector_latency_ms"),
    ("verifier", "hn_sev_latency_ms"),
    ("repair", "repair_latency_ms"),
    ("end_to_end", "end_to_end_latency_ms"),
]
HARDWARE_PROFILE_DIR = Path("tables/deployment_production_latency/hardware_profiles")
ENERGY_MEASUREMENT_DIR = Path("tables/deployment_production_latency/energy_measurements")


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json_records(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values) if values else 0.0,
        "median": median(values) if values else 0.0,
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
    }


def json_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.glob("*.json") if item.is_file())


def collect_hardware_profiles(root: Path) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    latency = read_json(root / "tables/deployment_latency/summary.json")
    if isinstance(latency, dict) and isinstance(latency.get("hardware"), dict):
        profile = dict(latency["hardware"])
        profile["source"] = "tables/deployment_latency/summary.json"
        profiles.append(profile)
    for path in json_files(root / HARDWARE_PROFILE_DIR):
        for record in read_json_records(path):
            record = dict(record)
            record.setdefault("source", path.relative_to(root).as_posix())
            profiles.append(record)
    return profiles


def hardware_profile_key(profile: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(profile.get("platform") or ""),
        str(profile.get("processor") or ""),
        str(profile.get("cpu_count") or ""),
        str(profile.get("device") or ""),
        str(profile.get("gpu_name") or ""),
        str(profile.get("python") or ""),
        str(profile.get("torch") or ""),
    )


def collect_energy_measurements(root: Path) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    for path in json_files(root / ENERGY_MEASUREMENT_DIR):
        for record in read_json_records(path):
            record = dict(record)
            record.setdefault("source", path.relative_to(root).as_posix())
            measurements.append(record)
    return measurements


def energy_joules(record: dict[str, Any]) -> float:
    for key in ("energy_joules", "joules", "total_energy_joules"):
        value = as_float(record.get(key), default=-1.0)
        if value > 0.0:
            return value
    return -1.0


def dataset_from_config(config: str | None) -> str:
    stem = Path(config or "").stem.lower()
    if stem == "mvtec":
        return "mvtec15"
    return stem or "unknown"


def configured_budget_ms(
    run_dir: Path,
    root: Path,
    config_arg: str | None,
    args: dict[str, Any] | None = None,
) -> float:
    if args and args.get("latency_budget_ms") is not None:
        return as_float(args.get("latency_budget_ms"))
    candidates = [run_dir / "config.yaml"]
    if config_arg:
        candidates.append(root / config_arg)
    for path in candidates:
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            budget = ((data.get("lc_rds") or {}).get("latency_budget_ms"))
            if budget is not None:
                return as_float(budget)
    return 0.0


def normalized_action(row: dict[str, Any]) -> str:
    value = str(row.get("scheduler_action") or row.get("action") or "skip").strip()
    return value.replace("_", "-") or "skip"


def spent_by_image(roi_rows: list[dict[str, Any]]) -> dict[int, float]:
    spent: dict[int, float] = {}
    fallback_sum: dict[int, float] = {}
    for row in roi_rows:
        idx = int(as_float(row.get("image_index"), 0.0))
        cumulative = row.get("cumulative_spent_ms")
        if cumulative is not None:
            spent[idx] = max(spent.get(idx, 0.0), as_float(cumulative))
        fallback_sum[idx] = fallback_sum.get(idx, 0.0) + as_float(row.get("action_latency_ms"))
    for idx, value in fallback_sum.items():
        if idx not in spent:
            spent[idx] = value
    return spent


def collect_runs(root: Path, run_prefix: str, ablation: str) -> list[dict[str, Any]]:
    runs_root = root / "runs"
    records: list[dict[str, Any]] = []
    if not runs_root.exists():
        return records
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if run_prefix and not run_dir.name.startswith(run_prefix):
            continue
        run_args = read_json(run_dir / "run_args.json")
        if not isinstance(run_args, dict):
            continue
        args = run_args.get("args") or {}
        if args.get("ablation") != ablation:
            continue
        component_rows = read_csv(run_dir / "component_latency.csv")
        roi_rows = read_json(run_dir / "roi_budget.json")
        if not component_rows or not isinstance(roi_rows, list):
            continue
        category = str(args.get("category") or "unknown")
        config = str(args.get("config") or "")
        dataset = dataset_from_config(config)
        budget = configured_budget_ms(run_dir, root, config, args)
        records.append(
            {
                "run": run_dir.name,
                "run_dir": run_dir,
                "dataset": dataset,
                "category": category,
                "config": config,
                "args": args,
                "configured_budget_ms": budget,
                "component_rows": component_rows,
                "roi_rows": roi_rows,
            }
        )
    return records


def component_summary_row(record: dict[str, Any]) -> dict[str, Any]:
    component_rows = record["component_rows"]
    roi_rows = record["roi_rows"]
    images = len(component_rows)
    image_spent = spent_by_image(roi_rows)
    budget = float(record["configured_budget_ms"])
    observed_actions = sorted({normalized_action(row) for row in roi_rows})
    row: dict[str, Any] = {
        "dataset": record["dataset"],
        "category": record["category"],
        "run": record["run"],
        "images": images,
        "configured_budget_ms": budget,
        "observed_actions": " ".join(observed_actions),
        "configured_budget_violation_rate": mean(
            [1.0 if image_spent.get(idx, 0.0) > budget + 1e-9 else 0.0 for idx in range(images)]
        )
        if images and budget > 0.0
        else 0.0,
    }
    for prefix, field in COMPONENT_FIELDS:
        values = [as_float(item.get(field)) for item in component_rows]
        summary = stats(values)
        row[f"{prefix}_latency_mean_ms"] = summary["mean"]
        row[f"{prefix}_latency_median_ms"] = summary["median"]
        row[f"{prefix}_latency_p95_ms"] = summary["p95"]
        row[f"{prefix}_latency_p99_ms"] = summary["p99"]
    return row


def budget_replay_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in records:
        record["configured_budget_key"] = int(round(float(record["configured_budget_ms"])))
    rows = []
    for budget in BUDGETS_MS:
        budget_records = [
            record for record in records if int(record.get("configured_budget_key", -1)) == budget
        ]
        all_spent: list[float] = []
        all_end_to_end: list[float] = []
        for record in budget_records:
            image_spent = spent_by_image(record["roi_rows"])
            images = len(record["component_rows"])
            all_spent.extend(image_spent.get(idx, 0.0) for idx in range(images))
            all_end_to_end.extend(
                as_float(row.get("end_to_end_latency_ms")) for row in record["component_rows"]
            )
        spent_stats = stats(all_spent)
        end_stats = stats(all_end_to_end)
        violations = [1.0 if value > budget + 1e-9 else 0.0 for value in all_spent]
        rows.append(
            {
                "budget_ms": budget,
                "runs": len(budget_records),
                "images": len(all_spent),
                "repair_spent_mean_ms": spent_stats["mean"],
                "repair_spent_median_ms": spent_stats["median"],
                "repair_spent_p95_ms": spent_stats["p95"],
                "repair_spent_p99_ms": spent_stats["p99"],
                "end_to_end_latency_mean_ms": end_stats["mean"],
                "end_to_end_latency_p95_ms": end_stats["p95"],
                "end_to_end_latency_p99_ms": end_stats["p99"],
                "budget_violation_rate": mean(violations) if violations else 1.0,
            }
        )
    return rows


def action_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {action: 0 for action in REQUIRED_ACTIONS}
    for record in records:
        for row in record["roi_rows"]:
            action = normalized_action(row)
            counts[action] = counts.get(action, 0) + 1
    return [
        {
            "action": action,
            "required": action in REQUIRED_ACTIONS,
            "observed": counts.get(action, 0) > 0,
            "roi_rows": counts.get(action, 0),
        }
        for action in sorted(counts, key=lambda item: (item not in REQUIRED_ACTIONS, item))
    ]


def rel_arg(root: Path, value: Any) -> str | None:
    if value in {None, ""}:
        return None
    text = str(value)
    path = Path(text)
    try:
        resolved_root = root.resolve()
        if path.is_absolute() and path.exists() and path.resolve().is_relative_to(resolved_root):
            return str(path.resolve().relative_to(resolved_root))
    except OSError:
        pass
    return text


def add_optional(cmd: list[str], flag: str, value: Any, root: Path) -> None:
    value = rel_arg(root, value)
    if value is not None:
        cmd.extend([flag, value])


def infer_command(record: dict[str, Any], root: Path, budget_ms: int, run_name: str) -> str:
    args = record["args"]
    cmd = [
        sys.executable,
        "infer.py",
        "--config",
        str(args.get("config") or record["config"] or "configs/mvtec.yaml"),
        "--category",
        str(args.get("category") or record["category"]),
        "--checkpoint",
        str(rel_arg(root, args.get("checkpoint")) or "<diffusion.pt>"),
        "--run-name",
        run_name,
        "--ablation",
        str(args.get("ablation") or "utility_lc_rds"),
        "--latency-budget-ms",
        str(budget_ms),
    ]
    add_optional(cmd, "--sev-checkpoint", args.get("sev_checkpoint"), root)
    add_optional(cmd, "--scheduler-checkpoint", args.get("scheduler_checkpoint"), root)
    add_optional(cmd, "--feature-prior-checkpoint", args.get("feature_prior_checkpoint"), root)
    for flag, key in [
        ("--image-size", "image_size"),
        ("--max-samples", "max_samples"),
        ("--seed", "seed"),
        ("--device", "device"),
        ("--crv-weight", "crv_weight"),
        ("--image-score-mode", "image_score_mode"),
        ("--image-score-source", "image_score_source"),
        ("--pixel-heatmap-source", "pixel_heatmap_source"),
        ("--reconstruction-steps", "reconstruction_steps"),
        ("--pixel-threshold-policy", "pixel_threshold_policy"),
    ]:
        add_optional(cmd, flag, args.get(key), root)
    if args.get("require_fixed_threshold"):
        cmd.append("--require-fixed-threshold")
    if args.get("allow_random_feature_weights"):
        cmd.append("--allow-random-feature-weights")
    if args.get("enable_retrieval_repair"):
        cmd.append("--enable-retrieval-repair")
    if args.get("disable_retrieval_repair"):
        cmd.append("--disable-retrieval-repair")
    return subprocess.list2cmdline([str(part) for part in cmd])


def checkpoint_ready(record: dict[str, Any], root: Path) -> bool:
    args = record["args"]
    required = [
        args.get("checkpoint"),
        args.get("sev_checkpoint"),
        args.get("feature_prior_checkpoint"),
    ]
    for value in required:
        if value in {None, ""}:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            return False
    return True


def missing_budget_command_rows(
    template_records: list[dict[str, Any]],
    present_records: list[dict[str, Any]],
    root: Path,
    *,
    command_run_prefix: str,
) -> list[dict[str, Any]]:
    templates: dict[tuple[str, str], dict[str, Any]] = {}
    present: set[tuple[str, str, int]] = set()
    for record in template_records:
        key = (str(record["dataset"]), str(record["category"]))
        templates.setdefault(key, record)
    for record in present_records:
        key = (str(record["dataset"]), str(record["category"]))
        budget = int(round(float(record["configured_budget_ms"])))
        present.add((key[0], key[1], budget))

    rows: list[dict[str, Any]] = []
    for dataset, category in sorted(templates):
        record = templates[(dataset, category)]
        for budget in BUDGETS_MS:
            has_run = (dataset, category, budget) in present
            if has_run:
                continue
            run_name = f"{command_run_prefix}_{dataset}_{category}_budget{budget}_utility_lc_rds"
            rows.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "budget_ms": budget,
                    "has_production_run": False,
                    "checkpoint_ready": checkpoint_ready(record, root),
                    "run_name": run_name,
                    "infer_command": infer_command(record, root, budget, run_name),
                    "evaluate_command": subprocess.list2cmdline(
                        [
                            sys.executable,
                            "evaluate.py",
                            "--pred_dir",
                            str(Path("runs") / run_name),
                            "--out",
                            str(Path("runs") / run_name / "eval_metrics.json"),
                        ]
                    ),
                    "next_step": "run infer/evaluate, rerun production deployment latency export, then rerun deployment readiness",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_outputs(
    root: Path,
    *,
    run_prefix: str = "fulltest",
    ablation: str = "utility_lc_rds",
    min_categories: int = 33,
    command_run_prefix: str = "production_lc_rds_budget",
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    source_records = collect_runs(root, run_prefix=run_prefix, ablation=ablation)
    budget_records = collect_runs(root, run_prefix=command_run_prefix, ablation=ablation)
    records_by_name = {str(record["run"]): record for record in source_records}
    records_by_name.update({str(record["run"]): record for record in budget_records})
    records = list(records_by_name.values())
    component_rows = [component_summary_row(record) for record in records]
    budget_rows = budget_replay_rows(budget_records)
    coverage_rows = action_rows(records)
    missing_commands = missing_budget_command_rows(
        source_records,
        budget_records,
        root,
        command_run_prefix=command_run_prefix,
    )
    categories = {(row["dataset"], row["category"]) for row in component_rows}
    observed_actions = sorted(
        row["action"] for row in coverage_rows if row["observed"] is True
    )
    max_budget_violation = max(
        [float(row["budget_violation_rate"]) for row in budget_rows] or [1.0]
    )
    component_ready = (
        len(categories) >= min_categories
        and all(int(row.get("images", 0)) > 0 for row in component_rows)
        and bool(component_rows)
    )
    multi_action_ready = (
        not missing_commands
        and len(categories) >= min_categories
        and all(action in observed_actions for action in REQUIRED_ACTIONS)
        and len(budget_rows) == len(BUDGETS_MS)
        and max_budget_violation <= 0.01
    )
    hardware_profiles = collect_hardware_profiles(root)
    hardware_profile_keys = {
        hardware_profile_key(profile)
        for profile in hardware_profiles
        if any(hardware_profile_key(profile))
    }
    energy_measurements = collect_energy_measurements(root)
    valid_energy_measurements = [
        record for record in energy_measurements if energy_joules(record) > 0.0
    ]
    summary = {
        "schema": "lite-seer-ad-production-deployment-latency-v1",
        "evidence_level": "production_inference_component_latency_v1",
        "run_prefix": run_prefix,
        "budget_run_prefix": command_run_prefix,
        "ablation": ablation,
        "runs": len(records),
        "source_runs": len(source_records),
        "budget_runs": len(budget_records),
        "categories": len(categories),
        "min_categories": min_categories,
        "images": sum(int(row.get("images", 0)) for row in component_rows),
        "component_latency_protocol": "per_image_synchronized_component_breakdown_v1",
        "production_component_latency_ready": component_ready,
        "budgets_ms": BUDGETS_MS,
        "budget_rows": len(budget_rows),
        "expected_budget_runs": len(categories) * len(BUDGETS_MS),
        "missing_budget_runs": len(missing_commands),
        "budget_sweep_coverage_ready": not missing_commands and bool(categories),
        "observed_actions": observed_actions,
        "required_actions": REQUIRED_ACTIONS,
        "multi_action_budget_sweep_ready": multi_action_ready,
        "max_budget_violation_rate": max_budget_violation,
        "hardware_profiles": len(hardware_profile_keys),
        "hardware_profile_sources": sorted(
            {
                str(profile.get("source"))
                for profile in hardware_profiles
                if profile.get("source")
            }
        ),
        "cross_hardware_ready": len(hardware_profile_keys) >= 2,
        "energy_measurements": len(valid_energy_measurements),
        "energy_measurement_sources": sorted(
            {
                str(record.get("source"))
                for record in valid_energy_measurements
                if record.get("source")
            }
        ),
        "energy_measurement_ready": bool(valid_energy_measurements),
        "release_gate_passed": component_ready and multi_action_ready,
        "release_gate_reason": (
            "Production inference component latency is covered, but historical logs do not cover the full multi-action six-budget LC-RDS protocol."
            if component_ready and not multi_action_ready
            else "Production inference latency and budget sweep are ready."
            if component_ready and multi_action_ready
            else "Production inference component latency coverage is incomplete."
        ),
        "required_for_release": [
            "collect six-budget frozen pipeline inference logs for 10/25/50/75/100/150 ms",
            "observe skip/repair-5/repair-10/repair-25/native-refine in production ROI logs",
            "record hardware profile, GPU memory, and energy for production runs",
            "add a second hardware profile before cross-hardware deployment claims",
        ],
    }
    return component_rows, budget_rows, coverage_rows, missing_commands, summary


def write_outputs(
    root: Path,
    out_dir: Path,
    *,
    run_prefix: str = "fulltest",
    ablation: str = "utility_lc_rds",
    min_categories: int = 33,
    command_run_prefix: str = "production_lc_rds_budget",
) -> dict[str, Any]:
    component_rows, budget_rows, coverage_rows, missing_commands, summary = build_outputs(
        root,
        run_prefix=run_prefix,
        ablation=ablation,
        min_categories=min_categories,
        command_run_prefix=command_run_prefix,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_production_component_latency.csv", component_rows)
    write_csv(out_dir / "table_production_budget_replay.csv", budget_rows)
    write_csv(out_dir / "table_production_action_coverage.csv", coverage_rows)
    write_csv(out_dir / "table_missing_production_budget_commands.csv", missing_commands)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("tables/deployment_production_latency"))
    parser.add_argument("--run-prefix", default="fulltest")
    parser.add_argument("--ablation", default="utility_lc_rds")
    parser.add_argument("--min-categories", type=int, default=33)
    parser.add_argument("--command-run-prefix", default="production_lc_rds_budget")
    args = parser.parse_args()
    summary = write_outputs(
        args.root,
        args.out_dir,
        run_prefix=args.run_prefix,
        ablation=args.ablation,
        min_categories=args.min_categories,
        command_run_prefix=args.command_run_prefix,
    )
    print(
        f"Wrote production deployment latency to {args.out_dir} "
        f"(production_component_latency_ready={summary['production_component_latency_ready']})"
    )


if __name__ == "__main__":
    main()
