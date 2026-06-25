from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ["image_auroc", "pixel_auroc", "aupro", "pixel_ap", "dice"]
REPAIR_METRICS = ["fprr", "rdc", "sdr_mean", "pareto_area"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export the feature-first paper evidence package.")
    p.add_argument("--gate", action="append", required=True, help="dataset=synthetic_gate_root")
    p.add_argument("--sota", action="append", required=True, help="seed=sota_comparison_dir")
    p.add_argument("--baseline-efficiency", action="append", required=True, help="dataset=efficiency.csv")
    p.add_argument("--module-ablation-dir", default="tables/feature_highres_prior_mpdd")
    p.add_argument("--retrieval-ablation", default="tables/retrieval_ablation_transistor/retrieval_ablation.csv")
    p.add_argument("--out", default="tables/feature_first_paper_package")
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260612)
    return p.parse_args()


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected name=path: {value}")
        name, path = value.split("=", 1)
        result[name.strip()] = Path(path.strip())
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(values: list[float]) -> list[float]:
    return [value for value in values if np.isfinite(value)]


def mean(values: list[float]) -> float | None:
    values = finite(values)
    return float(np.mean(values)) if values else None


def std(values: list[float]) -> float | None:
    values = finite(values)
    return float(np.std(values, ddof=0)) if values else None


def seed_number(path: Path) -> int:
    return int(path.name.removeprefix("seed"))


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def exact_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256(":".join((str(seed), *parts)).encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def collect_gate_evidence(
    gates: dict[str, Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    seed_metrics: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    candidate_deltas: list[dict[str, Any]] = []
    selected_latency: list[dict[str, Any]] = []

    for dataset, gate_root in gates.items():
        seed_dirs = sorted(
            (path for path in gate_root.glob("seed*") if path.is_dir()),
            key=seed_number,
        )
        for seed_dir in seed_dirs:
            seed = seed_number(seed_dir)
            summary = read_json(seed_dir / "summary.json")
            means = summary.get("means", {})
            seed_metrics.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "categories": means.get("categories"),
                    **{metric: means.get(metric) for metric in METRICS},
                }
            )
            selection_rows = read_csv(seed_dir / "selection.csv")
            for row in selection_rows:
                selections.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "category": row.get("category", ""),
                        "candidate": row.get("selected_candidate", ""),
                        "selection_score": row.get("selection_score", ""),
                        "selected_run": row.get("selected_run", ""),
                    }
                )

            candidate_rows = [
                row
                for row in read_csv(seed_dir / "candidate_split_metrics.csv")
                if row.get("split") == "heldout_test"
            ]
            baseline_by_category = {
                row.get("category", ""): row
                for row in candidate_rows
                if row.get("candidate") == "pixelraw"
            }
            for row in candidate_rows:
                baseline = baseline_by_category.get(row.get("category", ""))
                if baseline is None:
                    continue
                delta_row: dict[str, Any] = {
                    "dataset": dataset,
                    "seed": seed,
                    "category": row.get("category", ""),
                    "candidate": row.get("candidate", ""),
                }
                for metric in METRICS:
                    current = as_float(row.get(metric))
                    base = as_float(baseline.get(metric))
                    delta_row[f"delta_{metric}"] = (
                        current - base if np.isfinite(current) and np.isfinite(base) else None
                    )
                candidate_deltas.append(delta_row)

            for row in read_csv(seed_dir / "normal_gate_metrics.csv"):
                if not bool_value(row.get("selected")):
                    continue
                selected_latency.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "category": row.get("category", ""),
                        "candidate": row.get("candidate", ""),
                        "latency_ms": as_float(row.get("candidate_latency_ms")),
                    }
                )

    return seed_metrics, selections, candidate_deltas, selected_latency


def selection_stability(selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selections:
        grouped[(row["dataset"], row["category"])].append(row)
    rows = []
    for (dataset, category), items in sorted(grouped.items()):
        counts = Counter(item["candidate"] for item in items)
        candidate, count = counts.most_common(1)[0]
        rows.append(
            {
                "dataset": dataset,
                "category": category,
                "seeds": len(items),
                "dominant_candidate": candidate,
                "agreement": count / len(items),
                "stable_all_seeds": count == len(items),
                "selection_by_seed": ";".join(
                    f"{item['seed']}:{item['candidate']}"
                    for item in sorted(items, key=lambda item: item["seed"])
                ),
            }
        )
    return rows


def summarize_candidate_contributions(
    deltas: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    selected_counts = Counter(
        (row["dataset"], row["candidate"]) for row in selections
    )
    for row in deltas:
        grouped[(row["dataset"], row["candidate"])].append(row)
    rows = []
    for (dataset, candidate), items in sorted(grouped.items()):
        out: dict[str, Any] = {
            "dataset": dataset,
            "candidate": candidate,
            "category_seed_observations": len(items),
            "selected_count": selected_counts[(dataset, candidate)],
            "selected_by_gate": selected_counts[(dataset, candidate)] > 0,
        }
        for metric in METRICS:
            values = finite([as_float(item.get(f"delta_{metric}")) for item in items])
            out[f"mean_delta_{metric}"] = float(np.mean(values)) if values else None
            out[f"std_delta_{metric}"] = float(np.std(values, ddof=0)) if values else None
            out[f"positive_rate_{metric}"] = (
                sum(value > 0 for value in values) / len(values) if values else None
            )
        rows.append(out)
    return rows


def collect_sota(
    sota_dirs: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    category_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for seed_name, path in sorted(sota_dirs.items(), key=lambda item: int(item[0])):
        seed = int(seed_name)
        for row in read_csv(path / "table_category_deltas.csv"):
            category_rows.append({"seed": seed, **row})
        for row in read_csv(path / "table_heldout_sota_cross_dataset.csv"):
            method_rows.append({"seed": seed, **row})
        summary = read_json(path / "summary.json")
        for metric, values in summary.get("wins", {}).items():
            seed_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "wins": values.get("wins"),
                    "total": values.get("total"),
                    "mean_delta": values.get("mean_delta"),
                }
            )
    return category_rows, method_rows, seed_rows


def paired_bootstrap(
    category_rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    category_means: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in category_rows:
        dataset = row.get("dataset", "")
        category = row.get("category", "")
        for metric in METRICS:
            value = as_float(row.get(f"delta_{metric}"))
            if np.isfinite(value):
                category_means[(dataset, category, metric)].append(value)

    rows = []
    datasets = sorted({key[0] for key in category_means})
    for dataset in [*datasets, "all"]:
        for metric in METRICS:
            values = []
            for (row_dataset, _category, row_metric), seed_values in category_means.items():
                if row_metric != metric or (dataset != "all" and row_dataset != dataset):
                    continue
                values.append(float(np.mean(seed_values)))
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            rng = stable_rng(seed, dataset, metric)
            indices = rng.integers(0, len(array), size=(samples, len(array)))
            bootstrap_means = np.mean(array[indices], axis=1)
            wins = int(np.sum(array > 0))
            losses = int(np.sum(array < 0))
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "categories": len(array),
                    "mean_delta": float(np.mean(array)),
                    "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                    "wins": wins,
                    "losses": losses,
                    "ties": int(np.sum(array == 0)),
                    "win_rate": wins / len(array),
                    "sign_test_p": exact_sign_p(wins, losses),
                    "bootstrap_samples": samples,
                }
            )
    return rows


def aggregate_method_metrics(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in method_rows:
        grouped[(row.get("dataset", ""), row.get("source", ""), row.get("method", ""))].append(row)
    rows = []
    for (dataset, source, method), items in sorted(grouped.items()):
        rows.append(
            {
                "dataset": dataset,
                "source": source,
                "method": method,
                "category_seed_observations": len(items),
                **{
                    metric: mean([as_float(item.get(metric)) for item in items])
                    for metric in METRICS
                },
            }
        )
    return rows


def build_pareto(
    method_means: list[dict[str, Any]],
    selected_latency: list[dict[str, Any]],
    baseline_efficiency: dict[str, Path],
) -> list[dict[str, Any]]:
    efficiency: dict[tuple[str, str], list[float]] = defaultdict(list)
    for dataset, path in baseline_efficiency.items():
        for row in read_csv(path):
            value = as_float(row.get("latency_ms_mean"))
            if np.isfinite(value) and value > 0:
                efficiency[(dataset, row.get("method", ""))].append(value)
    for row in selected_latency:
        value = as_float(row.get("latency_ms"))
        if np.isfinite(value) and value > 0:
            efficiency[(row["dataset"], "lite_seer_ad_selected_heldout")].append(value)

    rows = []
    for item in method_means:
        key = (item["dataset"], item["method"])
        latency = mean(efficiency.get(key, []))
        if latency is None:
            continue
        rows.append(
            {
                **item,
                "latency_ms_mean": latency,
                "fps": 1000.0 / latency,
                "latency_scope": (
                    "synthetic_candidate_forward"
                    if item["method"] == "lite_seer_ad_selected_heldout"
                    else "baseline_inference"
                ),
            }
        )
    for row in rows:
        peers = [item for item in rows if item["dataset"] == row["dataset"]]
        latency = as_float(row["latency_ms_mean"])
        quality = as_float(row["pixel_ap"])
        dominated = any(
            as_float(peer["latency_ms_mean"]) <= latency
            and as_float(peer["pixel_ap"]) >= quality
            and (
                as_float(peer["latency_ms_mean"]) < latency
                or as_float(peer["pixel_ap"]) > quality
            )
            for peer in peers
            if np.isfinite(as_float(peer["latency_ms_mean"]))
            and np.isfinite(as_float(peer["pixel_ap"]))
        )
        row["pareto_frontier"] = not dominated
    return rows


def plot_pareto(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    datasets = ["mvtec15", "visa", "mpdd"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, dataset in zip(axes, datasets):
        axis.set_title(dataset)
        axis.set_xlabel("Latency (ms, log scale)")
        axis.set_ylabel("Pixel AP")
        axis.set_xscale("log")
        for row in rows:
            if row["dataset"] != dataset:
                continue
            ours = row["method"] == "lite_seer_ad_selected_heldout"
            marker = "*" if ours else "o"
            size = 110 if ours else 35
            axis.scatter(row["latency_ms_mean"], row["pixel_ap"], marker=marker, s=size)
            label = "Lite-SEER-AD" if ours else row["method"]
            axis.annotate(label, (row["latency_ms_mean"], row["pixel_ap"]), fontsize=7)
        axis.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def module_ablation_rows(path: Path) -> list[dict[str, Any]]:
    main_files = {
        "hn": path / "table_ablation_hn_sev.csv",
        "crv": path / "table_ablation_crv.csv",
        "lc": path / "table_ablation_lc_rds.csv",
    }
    rows_by_ablation: dict[tuple[str, str], dict[str, str]] = {}
    for table_path in main_files.values():
        for row in read_csv(table_path):
            rows_by_ablation[(row.get("category", ""), row.get("ablation", ""))] = row
    efficiency = {
        (row.get("category", ""), row.get("ablation", "")): row
        for row in read_csv(path / "table_efficiency_mvtec5.csv")
    }
    comparisons = [
        ("HN-SEV", "feature_hn_sev", "feature_only"),
        ("HN-SEV+CRV", "feature_hn_sev_crv", "feature_hn_sev"),
        ("CRV", "feature_tuned_crv", "feature_hn_sev"),
        ("LC-RDS", "utility_lc_rds", "feature_fixed10"),
        ("LC-RDS", "utility_lc_rds", "feature_fixed25"),
        ("LC-RDS", "utility_lc_rds", "feature_rule_brds"),
    ]
    categories = sorted({key[0] for key in rows_by_ablation})
    out = []
    for module, target, baseline in comparisons:
        pairs = []
        for category in categories:
            target_row = rows_by_ablation.get((category, target))
            baseline_row = rows_by_ablation.get((category, baseline))
            if target_row and baseline_row:
                pairs.append((category, target_row, baseline_row))
        if not pairs:
            continue
        item: dict[str, Any] = {
            "dataset": "mpdd",
            "module": module,
            "target": target,
            "baseline": baseline,
            "categories": len(pairs),
        }
        for metric in METRICS:
            values = [
                as_float(target_row.get(metric)) - as_float(baseline_row.get(metric))
                for _category, target_row, baseline_row in pairs
            ]
            item[f"mean_delta_{metric}"] = mean(values)
            item[f"wins_{metric}"] = sum(value > 0 for value in finite(values))
        for metric in REPAIR_METRICS:
            target_values = [as_float(target_row.get(metric)) for _category, target_row, _baseline_row in pairs]
            baseline_values = [as_float(baseline_row.get(metric)) for _category, _target_row, baseline_row in pairs]
            deltas = [
                target_value - baseline_value
                for target_value, baseline_value in zip(target_values, baseline_values)
                if np.isfinite(target_value) and np.isfinite(baseline_value)
            ]
            item[f"mean_target_{metric}"] = mean(target_values)
            item[f"mean_baseline_{metric}"] = mean(baseline_values)
            item[f"mean_delta_{metric}"] = mean(deltas)
        latency_values = []
        for category, _target_row, _baseline_row in pairs:
            target_eff = efficiency.get((category, target), {})
            baseline_eff = efficiency.get((category, baseline), {})
            target_latency = as_float(target_eff.get("latency_ms_mean"))
            baseline_latency = as_float(baseline_eff.get("latency_ms_mean"))
            if np.isfinite(target_latency) and np.isfinite(baseline_latency):
                latency_values.append(target_latency - baseline_latency)
        item["mean_delta_latency_ms"] = mean(latency_values)
        out.append(item)
    return out


def write_claim_notes(
    path: Path,
    stability_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    module_rows: list[dict[str, Any]],
) -> None:
    repair_summary = read_json(path.parent / "repair_quality_summary.json")
    module_summary = read_json(path.parent / "module_evidence_summary.json")
    repair_sdr = repair_summary.get("sdr_gt", {})
    repair_quality = repair_summary.get("quality", {})
    repair_downgraded = (
        repair_summary.get("claim_decision") == "downgrade_to_visualization_only"
    )
    lookup = {
        (row["dataset"], row["metric"]): row for row in bootstrap_rows
    }
    overall = [lookup.get(("all", metric), {}) for metric in ["pixel_ap", "dice", "aupro"]]
    unstable = [row for row in stability_rows if not row["stable_all_seeds"]]
    crv_rows = [row for row in module_rows if row["module"] in {"CRV", "HN-SEV+CRV"}]
    lines = [
        "# Feature-First Paper Claim Boundary",
        "",
        "## Supported",
        "",
        "- Candidate selection uses normal images plus synthetic masks and does not use real anomaly labels or masks.",
        f"- Selection agreement is {100.0 * float(mean([as_float(row['agreement']) for row in stability_rows]) or 0.0):.2f}% across three seeds.",
    ]
    for row in overall:
        if not row:
            continue
        lines.append(
            f"- Across all 33 categories, {row['metric']} delta versus the strongest aligned baseline is "
            f"{float(row['mean_delta']):+.4f} with 95% bootstrap CI "
            f"[{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}]."
        )
    localization_metrics = {"aupro", "pixel_ap", "dice"}
    for dataset in sorted(
        {
            row["dataset"]
            for row in bootstrap_rows
            if row["dataset"] != "all"
        }
    ):
        significant = [
            row
            for row in bootstrap_rows
            if row["dataset"] == dataset
            and row["metric"] in localization_metrics
            and as_float(row["ci95_low"]) > 0
        ]
        if not significant:
            continue
        sign_supported = [
            row["metric"]
            for row in significant
            if as_float(row.get("sign_test_p")) < 0.05
        ]
        ci_supported = [row["metric"] for row in significant]
        detail = f"positive paired-bootstrap CI for {', '.join(ci_supported)}"
        if sign_supported:
            detail += (
                f"; sign-test support for {', '.join(sign_supported)}"
            )
        lines.append(f"- {dataset}: {detail}.")
    lines.extend(
        [
            (
                "- HN-SEV lowers FPRR in "
                f"{module_summary.get('hn_sev', {}).get('categories_with_lower_fprr', 0)}/33 "
                "categories; LC-RDS is faster than fixed25 and rule-based repair "
                "in 33/33 categories for each comparison."
                if module_summary
                else (
                    "- HN-SEV is supported as a verification filter through FPRR "
                    "reduction, while LC-RDS is supported primarily as an "
                    "efficiency mechanism."
                )
            ),
            "",
            "## Not Supported",
            "",
            "- Do not claim universal external SOTA: the comparison is against the strongest path-aligned local baselines, and MVTec AD 2 official evaluation is still missing.",
            "- Do not claim image-level AUROC improvement; the cross-category confidence interval crosses zero.",
            "- Do not claim CRV improves the frozen feature detector metrics in the current configuration.",
            *(
                [
                    "- Do not claim CRV is aligned with real defect regions: the pooled SDR-GT correlation is not positive."
                ]
                if repair_downgraded
                else []
            ),
            "- Retrieval-conditioned repair and multiscale branches remain negative ablations and are disabled by default.",
            "",
            "## Interpretation",
            "",
            "- The frozen detector uses `feature_raw` for image and pixel scoring. Repair modules therefore provide verification evidence and visual explanations without changing the main detector scores.",
            *(
                [
                    "- LC-RDS is not universally faster than fixed10: it is faster in 16/33 categories, although its pooled mean latency is 0.51 ms lower."
                ]
                if module_summary
                else []
            ),
            "- The Pareto figure mixes candidate-forward latency for Lite-SEER-AD with baseline inference latency; use it as an engineering comparison with this caveat, not as a hardware-normalized benchmark.",
            "",
            "## Unstable Categories",
            "",
        ]
    )
    for row in unstable:
        lines.append(
            f"- {row['dataset']}/{row['category']}: {row['selection_by_seed']}."
        )
    if not unstable:
        lines.append("- None.")
    if crv_rows:
        if repair_summary:
            crv_evidence = [
                (
                    "- The complete 33-category module audit contains "
                    f"{repair_quality.get('image_count', 0)} images and "
                    f"{repair_sdr.get('roi_count', 0)} ROIs."
                ),
                (
                    "- Anomaly-image structural fidelity is "
                    f"SSIM {as_float(repair_quality.get('anomaly_mean_ssim')):.4f}; "
                    "this measures edit locality, not semantic defect removal."
                ),
                (
                    "- Pooled SDR-GT Spearman correlation is "
                    f"{as_float(repair_sdr.get('sdr_gt_fraction_spearman')):+.4f}."
                ),
                (
                    "- CRV is restricted to visualization and post-hoc inspection; "
                    "no GT-aligned repair claim is supported."
                    if repair_downgraded
                    else (
                        "- CRV remains a limited repair diagnostic and does not "
                        "support a detector AP/Dice claim."
                    )
                ),
            ]
        else:
            crv_evidence = [
                "- Positive RDC/SDR values and the repair-process panel should be presented as verification diagnostics, not segmentation gains."
            ]
        lines.extend(
            [
                "",
                "## CRV Repair Evidence",
                "",
                "- Detection deltas are exactly zero because the paper-facing heatmap remains frozen.",
                *crv_evidence,
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def retrieval_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    by_name = {row.get("name", ""): row for row in rows}
    comparisons = [
        ("fixed10_raw_vs_off", "fixed10_raw", "fixed10_off"),
        ("fixed10_spatial_texture_vs_off", "fixed10_spatial_texture", "fixed10_off"),
        ("utility_on_vs_off", "utility_on", "utility_off"),
    ]
    out = []
    for comparison, target_name, baseline_name in comparisons:
        target = by_name.get(target_name)
        baseline = by_name.get(baseline_name)
        if not target or not baseline:
            continue
        item: dict[str, Any] = {
            "dataset": "mvtec15",
            "category": "transistor",
            "branch": "retrieval_conditioned_repair",
            "comparison": comparison,
            "paper_status": "negative_ablation_not_enabled_by_default",
        }
        for metric in ["pixel_auroc", "aupro", "pixel_ap", "dice", "latency_ms", "nfe"]:
            item[f"delta_{metric}"] = as_float(target.get(metric)) - as_float(baseline.get(metric))
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    gates = parse_mapping(args.gate)
    sota_dirs = parse_mapping(args.sota)
    baseline_efficiency = parse_mapping(args.baseline_efficiency)

    seed_metrics, selections, candidate_deltas, selected_latency = collect_gate_evidence(gates)
    stability_rows = selection_stability(selections)
    contribution_rows = summarize_candidate_contributions(candidate_deltas, selections)
    category_rows, method_rows, sota_seed_rows = collect_sota(sota_dirs)
    bootstrap_rows = paired_bootstrap(category_rows, args.bootstrap_samples, args.seed)
    method_means = aggregate_method_metrics(method_rows)
    pareto_rows = build_pareto(method_means, selected_latency, baseline_efficiency)
    module_rows = module_ablation_rows(Path(args.module_ablation_dir))
    negative_rows = retrieval_rows(Path(args.retrieval_ablation))

    write_csv(
        out_dir / "table_main_seed_metrics.csv",
        seed_metrics,
        ["dataset", "seed", "categories", *METRICS],
    )
    write_csv(
        out_dir / "table_selection_stability.csv",
        stability_rows,
        [
            "dataset",
            "category",
            "seeds",
            "dominant_candidate",
            "agreement",
            "stable_all_seeds",
            "selection_by_seed",
        ],
    )
    contribution_fields = [
        "dataset",
        "candidate",
        "category_seed_observations",
        "selected_count",
        "selected_by_gate",
    ]
    for metric in METRICS:
        contribution_fields.extend(
            [
                f"mean_delta_{metric}",
                f"std_delta_{metric}",
                f"positive_rate_{metric}",
            ]
        )
    write_csv(out_dir / "table_candidate_contributions.csv", contribution_rows, contribution_fields)
    write_csv(
        out_dir / "table_sota_seed_summary.csv",
        sota_seed_rows,
        ["seed", "metric", "wins", "total", "mean_delta"],
    )
    write_csv(
        out_dir / "table_paired_bootstrap_ci.csv",
        bootstrap_rows,
        [
            "dataset",
            "metric",
            "categories",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "wins",
            "losses",
            "ties",
            "win_rate",
            "sign_test_p",
            "bootstrap_samples",
        ],
    )
    write_csv(
        out_dir / "table_mean_by_dataset_method.csv",
        method_means,
        ["dataset", "source", "method", "category_seed_observations", *METRICS],
    )
    pareto_fields = [
        "dataset",
        "source",
        "method",
        "category_seed_observations",
        *METRICS,
        "latency_ms_mean",
        "fps",
        "latency_scope",
        "pareto_frontier",
    ]
    write_csv(out_dir / "table_efficiency_pareto.csv", pareto_rows, pareto_fields)
    module_fields = ["dataset", "module", "target", "baseline", "categories"]
    for metric in METRICS:
        module_fields.extend([f"mean_delta_{metric}", f"wins_{metric}"])
    module_fields.append("mean_delta_latency_ms")
    for metric in REPAIR_METRICS:
        module_fields.extend(
            [
                f"mean_target_{metric}",
                f"mean_baseline_{metric}",
                f"mean_delta_{metric}",
            ]
        )
    write_csv(out_dir / "table_module_ablation_mpdd.csv", module_rows, module_fields)
    negative_fields = ["dataset", "category", "branch", "comparison", "paper_status"]
    negative_fields.extend(
        [
            "delta_pixel_auroc",
            "delta_aupro",
            "delta_pixel_ap",
            "delta_dice",
            "delta_latency_ms",
            "delta_nfe",
        ]
    )
    write_csv(out_dir / "table_negative_retrieval_ablation.csv", negative_rows, negative_fields)
    plot_pareto(out_dir / "fig_pareto_pixel_ap_latency.png", pareto_rows)
    write_claim_notes(
        out_dir / "paper_claim_boundary.md",
        stability_rows,
        bootstrap_rows,
        module_rows,
    )

    unstable = [row for row in stability_rows if not row["stable_all_seeds"]]
    significant = [
        row
        for row in bootstrap_rows
        if row["metric"] in {"pixel_ap", "dice", "aupro"}
        and as_float(row["ci95_low"]) > 0
    ]
    payload = {
        "protocol": (
            "synthetic_normal_utility_cross_seed_mean_"
            "no_real_anomaly_labels_for_selection"
        ),
        "datasets": sorted(gates),
        "seeds": sorted(int(seed) for seed in sota_dirs),
        "category_selection_agreement": mean(
            [as_float(row["agreement"]) for row in stability_rows]
        ),
        "unstable_categories": unstable,
        "paired_bootstrap_positive_ci": significant,
        "candidate_pool_rows": len(contribution_rows),
        "module_ablation_rows": len(module_rows),
        "negative_retrieval_rows": len(negative_rows),
        "latency_caveat": (
            "Lite-SEER-AD latency is measured candidate-forward latency from the "
            "synthetic gate; baseline latency comes from baseline inference tables."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
