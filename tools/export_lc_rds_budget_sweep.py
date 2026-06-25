"""Export a measured LC-RDS synthetic action budget sweep.

This audit measures every LC-RDS action with a deterministic local workload,
then replays the six paper budgets with those measured wall-clock latencies.
It proves action executability and budget accounting, but deliberately does
not claim production deployment readiness because it does not run the frozen
detector/verifier/repair stack on real images.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seer_ad_v2.models.scheduler.lc_rds import ACTION_NAMES, action_from_name
from tools.export_lc_rds_budget_audit import (
    as_float,
    collect_run_dirs,
    group_by_image,
    normalized_action,
    parse_run_dir,
    row_gain,
    row_nfe,
)


BUDGETS_MS = [10, 25, 50, 75, 100, 150]
REQUIRED_ACTIONS = ["skip", "repair-5", "repair-10", "repair-25", "native-refine"]
ACTION_GAIN = {
    "skip": 0.0,
    "repair-5": 0.24,
    "repair-10": 0.38,
    "repair-25": 0.58,
    "native-refine": 0.66,
}


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


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


def action_steps(action: str) -> int:
    return action_from_name(action).steps


def action_workload(action: str, seed: int) -> float:
    if action == "skip":
        return 0.0
    steps = action_steps(action)
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 1.0, size=(64, 64)).astype(np.float32)
    mask = rng.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)
    for idx in range(max(1, steps)):
        rolled = (
            np.roll(state, 1, axis=0)
            + np.roll(state, -1, axis=0)
            + np.roll(state, 1, axis=1)
            + np.roll(state, -1, axis=1)
        ) * 0.25
        state = 0.72 * state + 0.28 * rolled + 0.01 * mask
        if action == "native-refine":
            state = np.tanh(state + 0.001 * (idx + 1)).astype(np.float32)
    return float(np.mean(state))


def measure_action(action: str, warmups: int, repeats: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for idx in range(warmups):
        action_workload(action, idx)
    samples = []
    for idx in range(repeats):
        cuda_sync()
        start = time.perf_counter()
        checksum = action_workload(action, idx + 1000)
        cuda_sync()
        latency_ms = (time.perf_counter() - start) * 1000.0
        samples.append(
            {
                "action": action,
                "sample_index": idx,
                "latency_ms": latency_ms,
                "nfe": action_steps(action),
                "checksum": checksum,
            }
        )
    latencies = [float(row["latency_ms"]) for row in samples]
    summary = {
        "action": action,
        "samples": len(samples),
        "latency_mean_ms": mean(latencies) if latencies else 0.0,
        "latency_median_ms": median(latencies) if latencies else 0.0,
        "latency_p95_ms": quantile(latencies, 0.95),
        "latency_p99_ms": quantile(latencies, 0.99),
        "nfe": action_steps(action),
        "gain_proxy": ACTION_GAIN[action],
    }
    return samples, summary


def replay_budget(action_summaries: list[dict[str, Any]], budget_ms: int) -> dict[str, Any]:
    candidates = []
    for row in action_summaries:
        action = str(row["action"])
        if action == "skip":
            continue
        latency = float(row["latency_p95_ms"])
        gain = float(row["gain_proxy"])
        candidates.append((gain / max(latency, 1e-6), gain, latency, row))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    spent = 0.0
    gain_sum = 0.0
    nfe_sum = 0
    actions: list[str] = []
    for _, gain, latency, row in candidates:
        if spent + latency > float(budget_ms):
            continue
        spent += latency
        gain_sum += gain
        nfe_sum += int(row["nfe"])
        actions.append(str(row["action"]))
    if not actions:
        actions.append("skip")
    return {
        "budget_ms": budget_ms,
        "selected_actions": " ".join(actions),
        "selected_action_count": len([action for action in actions if action != "skip"]),
        "latency_ms": spent,
        "latency_p95_accounting_ms": spent,
        "nfe": nfe_sum,
        "repair_gain_proxy": gain_sum,
        "repaired_area_ratio_proxy": 0.015 * nfe_sum,
        "budget_violation": 1.0 if spent > float(budget_ms) + 1e-9 else 0.0,
        "gain_per_ms": gain_sum / max(spent, 1.0),
    }


def replay_roi_budget(
    image_rows: list[dict[str, Any]],
    budget_ms: int,
    action_latency_ms: dict[str, float],
) -> dict[str, Any]:
    candidates = []
    for row in image_rows:
        action = normalized_action(row)
        latency = action_latency_ms.get(action, 0.0)
        gain = row_gain(row)
        candidates.append((gain / max(latency, 1e-6), gain, latency, row))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    spent = 0.0
    gain_sum = 0.0
    nfe_sum = 0
    area_sum = 0.0
    repaired = 0
    selected_actions: list[str] = []
    for _, gain, latency, row in candidates:
        action = normalized_action(row)
        if latency <= 0.0 or action == "skip":
            continue
        if spent + latency > float(budget_ms):
            continue
        spent += latency
        gain_sum += gain
        nfe_sum += row_nfe(row)
        area_sum += as_float(row.get("area_ratio"), 0.0)
        repaired += 1
        selected_actions.append(action)
    return {
        "budget_ms": budget_ms,
        "latency_ms": spent,
        "repair_gain": gain_sum,
        "nfe": nfe_sum,
        "repaired_area_ratio": area_sum,
        "repaired_rois": repaired,
        "selected_actions": " ".join(sorted(set(selected_actions))) if selected_actions else "skip",
        "budget_violation": 1.0 if spent > float(budget_ms) + 1e-9 else 0.0,
        "gain_per_ms": gain_sum / max(spent, 1.0),
    }


def summarize_roi_budget(observations: list[dict[str, Any]], budget_ms: int) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in observations]
    gains = [float(row["repair_gain"]) for row in observations]
    nfes = [float(row["nfe"]) for row in observations]
    areas = [float(row["repaired_area_ratio"]) for row in observations]
    violations = [float(row["budget_violation"]) for row in observations]
    latency_mean = mean(latencies) if latencies else 0.0
    gain_mean = mean(gains) if gains else 0.0
    return {
        "budget_ms": budget_ms,
        "images": len(observations),
        "latency_mean_ms": latency_mean,
        "latency_median_ms": median(latencies) if latencies else 0.0,
        "latency_p95_ms": quantile(latencies, 0.95),
        "latency_p99_ms": quantile(latencies, 0.99),
        "repair_gain_mean": gain_mean,
        "nfe_mean": mean(nfes) if nfes else 0.0,
        "repaired_area_ratio_mean": mean(areas) if areas else 0.0,
        "budget_violation_rate": mean(violations) if violations else 0.0,
        "gain_per_ms": gain_mean / max(latency_mean, 1.0),
    }


def pareto_area(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    points = sorted(
        (float(row["latency_mean_ms"]), float(row["repair_gain_mean"]))
        for row in rows
    )
    max_latency = max(point[0] for point in points) or 1.0
    max_gain = max(point[1] for point in points) or 1.0
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += ((y0 / max_gain) + (y1 / max_gain)) * 0.5 * (
            (x1 - x0) / max_latency
        )
    return float(area)


def measured_roi_budget_sweep(
    root: Path,
    action_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    action_latency_ms = {
        str(row["action"]): float(row["latency_p95_ms"]) for row in action_rows
    }
    category_rows: list[dict[str, Any]] = []
    observations_by_budget: dict[int, list[dict[str, Any]]] = {
        budget: [] for budget in BUDGETS_MS
    }
    categories: set[tuple[str, str]] = set()
    image_count = 0
    roi_count = 0
    observed_actions: set[str] = set()
    for run_dir in collect_run_dirs(root):
        parsed = parse_run_dir(run_dir)
        if parsed is None:
            continue
        dataset, category = parsed
        roi_path = run_dir / "roi_budget.json"
        if not roi_path.exists():
            continue
        rows = json.loads(roi_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            continue
        categories.add((dataset, category))
        roi_count += len(rows)
        observed_actions.update(normalized_action(row) for row in rows)
        image_rows = group_by_image(rows)
        image_count += len(image_rows)
        for budget in BUDGETS_MS:
            observations = [
                replay_roi_budget(items, budget, action_latency_ms)
                for items in image_rows.values()
            ]
            observations_by_budget[budget].extend(observations)
            category_rows.append(
                {
                    "dataset": dataset,
                    "category": category,
                    **summarize_roi_budget(observations, budget),
                }
            )
    budget_rows = [
        summarize_roi_budget(observations_by_budget[budget], budget)
        for budget in BUDGETS_MS
    ]
    area = pareto_area(budget_rows)
    for row in budget_rows:
        row["pareto_area"] = area
    max_violation = max(
        [float(row["budget_violation_rate"]) for row in budget_rows] or [0.0]
    )
    summary = {
        "roi_measured_budget_replay_ready": bool(categories) and max_violation <= 0.01,
        "roi_measured_categories": len(categories),
        "roi_measured_images": image_count,
        "roi_measured_rows": roi_count,
        "roi_measured_observed_actions": sorted(observed_actions),
        "roi_measured_latency_source": "table_action_latency_summary.csv latency_p95_ms",
        "roi_measured_budget_rows": len(budget_rows),
        "roi_measured_pareto_area": area,
        "max_roi_measured_budget_violation_rate": max_violation,
    }
    return category_rows, budget_rows, summary


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


def build_sweep(
    root: Path | None = None,
    warmups: int = 3,
    repeats: int = 20,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    action_samples: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for action in REQUIRED_ACTIONS:
        samples, summary = measure_action(action, warmups=warmups, repeats=repeats)
        action_samples.extend(samples)
        action_rows.append(summary)
    budget_rows = [replay_budget(action_rows, budget) for budget in BUDGETS_MS]
    observed_actions = {row["action"] for row in action_rows if int(row["samples"]) > 0}
    max_violation = max(float(row["budget_violation"]) for row in budget_rows)
    roi_category_rows: list[dict[str, Any]] = []
    roi_budget_rows: list[dict[str, Any]] = []
    roi_summary: dict[str, Any] = {
        "roi_measured_budget_replay_ready": False,
        "roi_measured_categories": 0,
        "roi_measured_images": 0,
        "roi_measured_rows": 0,
        "roi_measured_observed_actions": [],
        "roi_measured_latency_source": "table_action_latency_summary.csv latency_p95_ms",
        "roi_measured_budget_rows": 0,
        "roi_measured_pareto_area": 0.0,
        "max_roi_measured_budget_violation_rate": 0.0,
    }
    if root is not None:
        roi_category_rows, roi_budget_rows, roi_summary = measured_roi_budget_sweep(
            root,
            action_rows,
        )
    summary = {
        "schema": "lite-seer-ad-lc-rds-budget-sweep-v1",
        "evidence_level": "measured_synthetic_action_budget_sweep_v1",
        "release_gate_passed": False,
        "release_gate_reason": (
            "All scheduler actions are measured with deterministic synthetic workloads, "
            "but this is not the final frozen detector/verifier/repair production sweep."
        ),
        "budgets_ms": BUDGETS_MS,
        "required_actions": REQUIRED_ACTIONS,
        "scheduler_code_actions": ACTION_NAMES,
        "observed_actions": sorted(observed_actions),
        "action_coverage_ready": all(action in observed_actions for action in REQUIRED_ACTIONS),
        "warmups": warmups,
        "repeats": repeats,
        "max_budget_violation_rate": max_violation,
        "latency_accounting": "measured_action_p95_wall_clock_ms",
        "budget_rows": len(budget_rows),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "required_for_release": [
            "run the frozen detector, HN-SEV verifier, LC-RDS scheduler, selected repair executor, and IO stack end-to-end",
            "measure per-budget wall-clock latency on real prediction samples",
            "record true production budget violation, memory, energy, and cross-hardware evidence",
        ],
        **roi_summary,
    }
    return action_samples, action_rows, budget_rows, roi_category_rows, roi_budget_rows, summary


def write_outputs(root: Path, out_dir: Path, warmups: int = 3, repeats: int = 20) -> dict[str, Any]:
    (
        action_samples,
        action_rows,
        budget_rows,
        roi_category_rows,
        roi_budget_rows,
        summary,
    ) = build_sweep(root=root, warmups=warmups, repeats=repeats)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_action_latency_samples.csv", action_samples)
    write_csv(out_dir / "table_action_latency_summary.csv", action_rows)
    write_csv(out_dir / "table_budget_sweep.csv", budget_rows)
    write_csv(out_dir / "table_category_roi_measured_budget_sweep.csv", roi_category_rows)
    write_csv(out_dir / "table_roi_measured_budget_sweep.csv", roi_budget_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("tables/lc_rds_budget_sweep"))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    summary = write_outputs(args.root, args.out_dir, warmups=args.warmups, repeats=args.repeats)
    print(
        f"Wrote LC-RDS synthetic budget sweep to {args.out_dir} "
        f"(action_coverage_ready={summary['action_coverage_ready']})"
    )


if __name__ == "__main__":
    main()
