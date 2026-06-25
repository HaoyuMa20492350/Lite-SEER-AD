"""Export an LC-RDS budget audit from existing ROI logs.

This exporter uses historical ``roi_budget.json`` files to replay LC-RDS
choices under the paper budgets. It is useful for traceability and for finding
missing deployment evidence, but it deliberately marks the output as an
offline replay rather than a measured synchronized budget sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seer_ad_v2.models.scheduler.lc_rds import ACTION_NAMES as SCHEDULER_ACTION_NAMES


BUDGETS_MS = [10, 25, 50, 75, 100, 150]
REQUIRED_ACTIONS = ["skip", "repair-5", "repair-10", "repair-25", "native-refine"]
ACTION_LATENCY_MS = {
    "skip": 0.0,
    "repair-5": 20.0,
    "repair5": 20.0,
    "repair-10": 40.0,
    "repair10": 40.0,
    "repair-25": 75.0,
    "repair25": 75.0,
    "native-refine": 80.0,
    "native_refine": 80.0,
}
ACTION_NFE = {
    "skip": 0,
    "repair-5": 5,
    "repair5": 5,
    "repair-10": 10,
    "repair10": 10,
    "repair-25": 25,
    "repair25": 25,
    "native-refine": 40,
    "native_refine": 40,
}
RUN_PATTERN = re.compile(
    r"^feature_fixedpixel_(?P<dataset>mvtec15|visa|mpdd)_(?P<category>.+)_feature_pixel_policy$"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_efficiency(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                rows[row["metric"]] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
    return rows


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


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalized_action(row: dict[str, Any]) -> str:
    action = str(row.get("scheduler_action") or "skip").replace("_", "-")
    if action.startswith("repair") and "-" not in action:
        action = action.replace("repair", "repair-")
    return action


def canonical_action(action: str) -> str:
    normalized = str(action or "skip").strip().replace("_", "-")
    if normalized.startswith("repair") and "-" not in normalized:
        normalized = normalized.replace("repair", "repair-")
    return normalized


def parse_budget_v2_config(root: Path) -> tuple[list[str], list[int]]:
    path = root / "configs/scheduler/lc_rds_budget_v2.yaml"
    if not path.exists():
        return [], []
    actions: list[str] = []
    budgets: list[int] = []
    in_actions = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "actions:":
            in_actions = True
            continue
        if in_actions and line.startswith("- "):
            actions.append(canonical_action(line[2:].strip()))
            continue
        if in_actions and line and not line.startswith("- "):
            in_actions = False
        if line.startswith("budgets_ms:"):
            numbers = re.findall(r"\d+", line)
            budgets = [int(value) for value in numbers]
    return actions, budgets


def parse_run_dir(path: Path) -> tuple[str, str] | None:
    match = RUN_PATTERN.match(path.name)
    if not match:
        return None
    return match.group("dataset"), match.group("category")


def collect_run_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in (root / "runs").glob("feature_fixedpixel_*_feature_pixel_policy")
        if path.is_dir() and parse_run_dir(path) is not None
    )


def group_by_image(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            image_index = int(row.get("image_index", 0))
        except (TypeError, ValueError):
            image_index = 0
        grouped[image_index].append(row)
    return dict(grouped)


def row_latency(row: dict[str, Any]) -> float:
    return ACTION_LATENCY_MS.get(normalized_action(row), 0.0)


def row_nfe(row: dict[str, Any]) -> int:
    action = normalized_action(row)
    if action in ACTION_NFE:
        return ACTION_NFE[action]
    return int(as_float(row.get("nfe"), 0.0))


def row_gain(row: dict[str, Any]) -> float:
    return max(0.0, as_float(row.get("repair_gain"), 0.0))


def replay_budget(image_rows: list[dict[str, Any]], budget_ms: float) -> dict[str, float]:
    candidates = []
    for row in image_rows:
        latency = row_latency(row)
        gain = row_gain(row)
        density = gain / max(latency, 1.0)
        candidates.append((density, gain, latency, row))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    spent = 0.0
    gain_sum = 0.0
    nfe_sum = 0
    area_sum = 0.0
    repaired = 0
    for _, gain, latency, row in candidates:
        if latency <= 0.0:
            continue
        if spent + latency > budget_ms:
            continue
        spent += latency
        gain_sum += gain
        nfe_sum += row_nfe(row)
        area_sum += as_float(row.get("area_ratio"), 0.0)
        repaired += 1
    return {
        "latency_ms": spent,
        "repair_gain": gain_sum,
        "nfe": float(nfe_sum),
        "repaired_area_ratio": area_sum,
        "repaired_rois": float(repaired),
        "budget_violation": 1.0 if spent > budget_ms + 1e-9 else 0.0,
    }


def summarize_budget_observations(
    observations: list[dict[str, float]], budget_ms: int
) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in observations]
    gains = [row["repair_gain"] for row in observations]
    nfes = [row["nfe"] for row in observations]
    areas = [row["repaired_area_ratio"] for row in observations]
    violations = [row["budget_violation"] for row in observations]
    latency_mean = mean(latencies) if latencies else 0.0
    gain_mean = mean(gains) if gains else 0.0
    return {
        "budget_ms": int(budget_ms),
        "images": len(observations),
        "latency_mean_ms": latency_mean,
        "latency_median_ms": median(latencies) if latencies else 0.0,
        "latency_p95_ms": quantile(latencies, 0.95),
        "latency_p99_ms": quantile(latencies, 0.99),
        "nfe_mean": mean(nfes) if nfes else 0.0,
        "repaired_area_ratio_mean": mean(areas) if areas else 0.0,
        "repair_gain_mean": gain_mean,
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_action_space_rows(
    root: Path, observed_counts: Counter[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_actions, config_budgets = parse_budget_v2_config(root)
    config_action_set = set(config_actions)
    code_action_set = {canonical_action(action) for action in SCHEDULER_ACTION_NAMES}
    observed_action_set = set(observed_counts)
    rows = []
    for action in REQUIRED_ACTIONS:
        rows.append(
            {
                "action": action,
                "required": True,
                "in_budget_v2_config": action in config_action_set,
                "in_scheduler_code": action in code_action_set,
                "observed_in_roi_logs": action in observed_action_set,
                "observed_count": int(observed_counts.get(action, 0)),
                "latency_estimate_ms": ACTION_LATENCY_MS.get(action, 0.0),
                "nfe": ACTION_NFE.get(action, 0),
            }
        )
    summary = {
        "required_actions": REQUIRED_ACTIONS,
        "budget_v2_config_actions": config_actions,
        "budget_v2_config_budgets_ms": config_budgets,
        "scheduler_code_actions": sorted(code_action_set),
        "observed_actions": sorted(observed_action_set),
        "action_space_ready": all(
            action in config_action_set and action in code_action_set
            for action in REQUIRED_ACTIONS
        ),
        "observed_required_actions_complete": all(
            action in observed_action_set for action in REQUIRED_ACTIONS
        ),
        "missing_config_actions": [
            action for action in REQUIRED_ACTIONS if action not in config_action_set
        ],
        "missing_code_actions": [
            action for action in REQUIRED_ACTIONS if action not in code_action_set
        ],
        "missing_observed_actions": [
            action for action in REQUIRED_ACTIONS if action not in observed_action_set
        ],
    }
    return rows, summary


def build_audit(
    root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    run_rows: list[dict[str, Any]] = []
    category_budget_rows: list[dict[str, Any]] = []
    all_budget_observations: dict[int, list[dict[str, float]]] = {
        budget: [] for budget in BUDGETS_MS
    }
    observed_action_counts: Counter[str] = Counter()

    for run_dir in collect_run_dirs(root):
        parsed = parse_run_dir(run_dir)
        if parsed is None:
            continue
        dataset, category = parsed
        roi_path = run_dir / "roi_budget.json"
        if not roi_path.exists():
            continue
        roi_rows = read_json(roi_path)
        if not isinstance(roi_rows, list):
            continue
        efficiency = read_efficiency(run_dir / "efficiency.csv")
        action_counts = Counter(normalized_action(row) for row in roi_rows)
        observed_action_counts.update(action_counts)
        image_rows = group_by_image(roi_rows)
        run_rows.append(
            {
                "dataset": dataset,
                "category": category,
                "run_dir": run_dir.as_posix(),
                "images": len(image_rows),
                "roi_rows": len(roi_rows),
                "observed_actions": " ".join(
                    f"{name}:{count}" for name, count in sorted(action_counts.items())
                ),
                "observed_latency_mean_ms": efficiency.get("latency_ms_mean", 0.0),
                "observed_nfe_mean": efficiency.get("nfe_mean", 0.0),
                "observed_repaired_area_ratio_mean": efficiency.get(
                    "repaired_area_ratio_mean", 0.0
                ),
                "observed_gain_mean": mean([row_gain(row) for row in roi_rows])
                if roi_rows
                else 0.0,
            }
        )

        for budget in BUDGETS_MS:
            observations = [
                replay_budget(rows, float(budget)) for rows in image_rows.values()
            ]
            all_budget_observations[budget].extend(observations)
            summary = summarize_budget_observations(observations, budget)
            category_budget_rows.append(
                {
                    "dataset": dataset,
                    "category": category,
                    **summary,
                }
            )

    budget_rows = [
        summarize_budget_observations(all_budget_observations[budget], budget)
        for budget in BUDGETS_MS
    ]
    area = pareto_area(budget_rows)
    for row in budget_rows:
        row["pareto_area"] = area

    action_space_rows, action_space_summary = build_action_space_rows(
        root, observed_action_counts
    )
    max_violation = max(
        [float(row["budget_violation_rate"]) for row in budget_rows] or [0.0]
    )
    summary = {
        "schema": "lite-seer-ad-lc-rds-budget-audit-v1",
        "evidence_level": "offline_replay_from_roi_logs_v1",
        "release_gate_passed": False,
        "release_gate_reason": (
            "This audit replays historical ROI logs with action latency estimates; "
            "it is not a synchronized measured budget sweep."
        ),
        "budgets_ms": BUDGETS_MS,
        "action_latency_ms": ACTION_LATENCY_MS,
        "action_space": action_space_summary,
        "runs": len(run_rows),
        "categories": len({(row["dataset"], row["category"]) for row in run_rows}),
        "images": sum(int(row["images"]) for row in run_rows),
        "roi_rows": sum(int(row["roi_rows"]) for row in run_rows),
        "max_budget_violation_rate": max_violation,
        "pareto_area": area,
        "required_for_release": [
            "measured synchronized action/runtime latency at batch=1",
            "true per-budget scheduler runs for 10/25/50/75/100/150 ms",
            "p95/p99 and budget violation measured from wall-clock samples",
            "detector/verifier/scheduler/repair/IO timing breakdown",
            "observed execution evidence for every required action, including repair-5 and native-refine",
        ],
    }
    return run_rows, category_budget_rows, budget_rows, action_space_rows, summary


def write_outputs(root: Path, out_dir: Path) -> dict[str, Any]:
    run_rows, category_budget_rows, budget_rows, action_space_rows, summary = build_audit(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_observed_runs.csv", run_rows)
    write_csv(out_dir / "table_category_budget_replay.csv", category_budget_rows)
    write_csv(out_dir / "table_budget_summary.csv", budget_rows)
    write_csv(out_dir / "table_action_space.csv", action_space_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tables/lc_rds_budget_audit"),
    )
    args = parser.parse_args()

    summary = write_outputs(args.root, args.out_dir)
    print(
        f"Wrote LC-RDS budget audit for {summary['categories']} categories "
        f"to {args.out_dir} (release_gate_passed={summary['release_gate_passed']})"
    )


if __name__ == "__main__":
    main()
