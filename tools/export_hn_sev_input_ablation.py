"""Export HN-SEV input-ablation coverage.

This audit scans existing HN-SEV ablation tables and separates exact requested
training-input ablations from weaker proxy evidence. It should not be used to
claim HN-SEV is recall-safe unless the exact required variants are complete.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


EXACT_REQUIREMENTS = {
    "synthetic_only": {"synthetic_only_sev"},
    "clean_normal_added": {"clean_normal_sev", "with_clean_normal"},
    "hard_negative_added": {"hard_negative_sev", "with_hard_negative"},
    "feature_prototype": {"no_prototype", "with_prototype"},
}
PROXY_VARIANTS = {
    "feature_full_proxy": {"feature_hn_sev", "feature_hn_sev_crv", "full"},
    "feature_only_proxy": {"feature_only"},
    "residual_only_proxy": {"residual_only"},
    "no_sev_proxy": {"no_sev"},
}
METRICS = ["fprr", "pixel_ap", "aupro", "dice"]
EXACT_ABLATIONS = set().union(*EXACT_REQUIREMENTS.values())
CONFIG_FOR_DATASET = {
    "mvtec15": "configs/mvtec.yaml",
    "visa": "configs/visa.yaml",
    "mpdd": "configs/mpdd.yaml",
}
PRIMARY_ABLATION_FOR_REQUIREMENT = {
    "synthetic_only": "synthetic_only_sev",
    "clean_normal_added": "clean_normal_sev",
    "hard_negative_added": "hard_negative_sev",
    "feature_prototype": "with_prototype",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def has_metric_values(row: dict[str, Any]) -> bool:
    return any(np.isfinite(as_float(row.get(metric))) for metric in METRICS)


def dataset_from_config(config_value: Any, run_dir: Path) -> str:
    text = str(config_value or "").replace("\\", "/").lower()
    if "mpdd" in text:
        return "mpdd"
    if "visa" in text:
        return "visa"
    if "mvtec" in text:
        return "mvtec15"
    cfg = read_yaml(run_dir / "config.yaml")
    name = str((cfg.get("dataset", {}) or {}).get("name", "")).lower()
    if name == "mvtec":
        return "mvtec15"
    return name


def normalize_dataset(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "mvtec":
        return "mvtec15"
    return text


def category_from_metadata(run_dir: Path, run_args: dict[str, Any], metrics: dict[str, Any]) -> str:
    args = run_args.get("args", {}) or {}
    if metrics.get("category"):
        return str(metrics["category"])
    if args.get("category"):
        return str(args["category"])
    cfg = read_yaml(run_dir / "config.yaml")
    category = (cfg.get("dataset", {}) or {}).get("category")
    return str(category or "")


def infer_exact_ablation(run_dir: Path, run_args: dict[str, Any], metrics: dict[str, Any]) -> str:
    args = run_args.get("args", {}) or {}
    explicit = (
        metrics.get("hn_sev_input_ablation")
        or metrics.get("input_ablation_label")
        or args.get("input_ablation_label")
        or args.get("ablation")
    )
    if explicit in EXACT_ABLATIONS:
        return str(explicit)
    name = run_dir.name
    for ablation in sorted(EXACT_ABLATIONS, key=len, reverse=True):
        if ablation in name:
            return ablation
    if args.get("synthetic_only") is True:
        return "synthetic_only_sev"
    if args.get("disable_prototype") is True:
        return "no_prototype"
    return ""


def row_from_run_metrics(root: Path, metrics_path: Path) -> dict[str, Any] | None:
    run_dir = metrics_path.parent
    run_args = read_json(run_dir / "run_args.json")
    metrics = read_json(metrics_path)
    ablation = infer_exact_ablation(run_dir, run_args, metrics)
    if not ablation:
        return None
    args = run_args.get("args", {}) or {}
    return {
        "dataset": dataset_from_config(args.get("config"), run_dir),
        "category": category_from_metadata(run_dir, run_args, metrics),
        "ablation": ablation,
        "fprr": metrics.get("fprr", ""),
        "pixel_ap": metrics.get("pixel_ap", ""),
        "aupro": metrics.get("aupro", metrics.get("aupro_proxy", "")),
        "dice": metrics.get("dice", ""),
        "source_table": metrics_path.relative_to(root).as_posix(),
        "source_kind": "run_metrics",
    }


def row_from_hn_sev_metadata(root: Path, metrics_path: Path) -> dict[str, Any] | None:
    run_dir = metrics_path.parent
    run_args = read_json(run_dir / "run_args.json")
    metrics = read_json(metrics_path)
    ablation = infer_exact_ablation(run_dir, run_args, metrics)
    if not ablation:
        return None
    args = run_args.get("args", {}) or {}
    return {
        "dataset": dataset_from_config(args.get("config"), run_dir),
        "category": category_from_metadata(run_dir, run_args, metrics),
        "ablation": ablation,
        "fprr": "",
        "pixel_ap": "",
        "aupro": "",
        "dice": "",
        "source_table": metrics_path.relative_to(root).as_posix(),
        "source_kind": "training_metadata",
    }


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "tables").rglob("table_ablation_hn_sev.csv")):
        for row in read_csv(path):
            ablation = str(row.get("ablation", "")).strip()
            if not ablation:
                continue
            rows.append(
                {
                    **row,
                    "ablation": ablation,
                    "source_table": path.relative_to(root).as_posix(),
                    "source_kind": "table_ablation_hn_sev",
                }
            )
    for path in sorted((root / "runs").glob("*/metrics.json")):
        row = row_from_run_metrics(root, path)
        if row is not None:
            rows.append(row)
    for path in sorted((root / "runs").glob("*/hn_sev_metrics.json")):
        row = row_from_hn_sev_metadata(root, path)
        if row is not None:
            rows.append(row)
    return rows


def category_key(row: dict[str, Any]) -> tuple[str, str]:
    return normalize_dataset(row.get("dataset", "")), str(row.get("category", ""))


def coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_category[category_key(row)].add(str(row["ablation"]))

    out = []
    for (dataset, category), variants in sorted(by_category.items()):
        record: dict[str, Any] = {
            "dataset": dataset,
            "category": category,
            "variant_count": len(variants),
            "variants": " ".join(sorted(variants)),
        }
        for name, accepted in EXACT_REQUIREMENTS.items():
            record[f"exact_{name}"] = bool(variants & accepted)
            record[f"metric_{name}"] = any(
                row.get("ablation") in accepted and has_metric_values(row)
                for row in rows
                if category_key(row) == (dataset, category)
            )
        for name, accepted in PROXY_VARIANTS.items():
            record[name] = bool(variants & accepted)
        record["exact_all_required"] = all(
            bool(variants & accepted) for accepted in EXACT_REQUIREMENTS.values()
        )
        record["metric_all_required"] = all(
            bool(record[f"metric_{name}"]) for name in EXACT_REQUIREMENTS
        )
        out.append(record)
    return out


def metric_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["ablation"])].append(row)

    summaries = []
    for ablation, items in sorted(grouped.items()):
        record: dict[str, Any] = {
            "ablation": ablation,
            "rows": len(items),
            "categories": len({category_key(row) for row in items}),
        }
        for metric in METRICS:
            values = np.asarray([as_float(row.get(metric)) for row in items], dtype=np.float64)
            values = values[np.isfinite(values)]
            record[f"{metric}_mean"] = float(values.mean()) if values.size else None
        summaries.append(record)
    return summaries


def build_summary(rows: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> dict[str, Any]:
    categories = len({(row["dataset"], row["category"]) for row in coverage})
    exact_counts = {
        name: sum(1 for row in coverage if row[f"exact_{name}"])
        for name in EXACT_REQUIREMENTS
    }
    proxy_counts = {
        name: sum(1 for row in coverage if row[name])
        for name in PROXY_VARIANTS
    }
    complete_exact_categories = sum(1 for row in coverage if row["exact_all_required"])
    complete_metric_categories = sum(1 for row in coverage if row["metric_all_required"])
    release_gate_passed = (
        categories >= 33
        and complete_exact_categories >= 33
        and complete_metric_categories >= 33
    )
    return {
        "schema": "lite-seer-ad-hn-sev-input-ablation-v1",
        "evidence_level": "mixed_table_run_metadata_coverage_v2",
        "release_gate_passed": release_gate_passed,
        "release_gate_reason": (
            "All 33 categories contain exact requested HN-SEV input ablations with metric rows."
            if release_gate_passed
            else "Existing artifacts provide module/proxy ablations or training metadata, but exact synthetic-only, +clean-normal, +hard-negative, and +feature/prototype input ablations with evaluated metrics are incomplete."
        ),
        "source_rows": len(rows),
        "categories": categories,
        "complete_exact_categories": complete_exact_categories,
        "complete_metric_categories": complete_metric_categories,
        "exact_requirement_counts": exact_counts,
        "proxy_variant_counts": proxy_counts,
        "required_for_release": [
            "synthetic_only_sev exact training-input ablation for all 33 categories",
            "clean_normal_added exact training-input ablation for all 33 categories",
            "hard_negative_added exact training-input ablation for all 33 categories",
            "feature/prototype exact ablation, including no_prototype or with_prototype, for all 33 categories",
            "link exact input ablations to TP retention/ROI recall, not FPRR alone",
        ],
    }


def candidate_model_dir(root: Path, dataset: str, category: str) -> Path | None:
    candidates = [
        root / "runs" / f"feature_{dataset}_{category}_models",
        root / "runs" / f"feature_highres_prior_{dataset}_{category}_models",
        root / "runs" / f"{dataset}_ours_{category}_models",
        root / "runs" / f"{dataset}_gate_{category}_models",
    ]
    for candidate in candidates:
        if (candidate / "diffusion.pt").is_file():
            return candidate
    return candidates[0]


def command_for_missing_variant(root: Path, dataset: str, category: str, requirement: str) -> str:
    dataset = normalize_dataset(dataset)
    config = CONFIG_FOR_DATASET.get(dataset, f"configs/{dataset}.yaml")
    label = PRIMARY_ABLATION_FOR_REQUIREMENT[requirement]
    model_dir = candidate_model_dir(root, dataset, category)
    checkpoint = model_dir / "diffusion.pt" if model_dir is not None else Path("<diffusion.pt>")
    hard_dir = model_dir / "hard_negatives" if model_dir is not None else Path("<hard_negatives>")
    feature_prior = model_dir / "feature_prior.pt" if model_dir is not None else Path("<feature_prior.pt>")
    run_name = f"hn_sev_exact_{dataset}_{category}_{label}"
    parts = [
        "python",
        "train_hn_sev.py",
        "--config",
        config,
        "--category",
        category,
        "--checkpoint",
        str(checkpoint),
        "--input-ablation-label",
        label,
        "--run-name",
        run_name,
    ]
    if requirement == "synthetic_only":
        parts.extend(["--synthetic-only", "--disable-clean-normal", "--disable-prototype"])
    elif requirement == "clean_normal_added":
        parts.extend(["--synthetic-only", "--disable-prototype"])
    elif requirement == "hard_negative_added":
        parts.extend(["--hard-negative-dir", str(hard_dir), "--disable-prototype"])
    elif requirement == "feature_prototype":
        parts.extend(["--hard-negative-dir", str(hard_dir)])
        if feature_prior.is_file():
            parts.extend(["--feature-prior-checkpoint", str(feature_prior)])
    source_args = read_json(model_dir / "run_args.json").get("args", {}) if model_dir is not None else {}
    for arg_name, cli_name in [
        ("epochs", "--epochs"),
        ("batch_size", "--batch-size"),
        ("image_size", "--image-size"),
        ("max_samples", "--max-samples"),
        ("seed", "--seed"),
    ]:
        value = source_args.get(arg_name)
        if value is not None:
            parts.extend([cli_name, str(value)])
    device = source_args.get("device")
    if device:
        parts.extend(["--device", str(device)])
    return " ".join(parts)


def inference_command_for_missing_variant(root: Path, dataset: str, category: str, requirement: str) -> str:
    dataset = normalize_dataset(dataset)
    config = CONFIG_FOR_DATASET.get(dataset, f"configs/{dataset}.yaml")
    label = PRIMARY_ABLATION_FOR_REQUIREMENT[requirement]
    model_dir = candidate_model_dir(root, dataset, category)
    checkpoint = model_dir / "diffusion.pt" if model_dir is not None else Path("<diffusion.pt>")
    feature_prior = model_dir / "feature_prior.pt" if model_dir is not None else Path("<feature_prior.pt>")
    train_run_name = f"hn_sev_exact_{dataset}_{category}_{label}"
    eval_run_name = f"hn_sev_exact_eval_{dataset}_{category}_{label}"
    source_args = read_json(model_dir / "run_args.json").get("args", {}) if model_dir is not None else {}
    parts = [
        "python",
        "infer.py",
        "--config",
        config,
        "--category",
        category,
        "--checkpoint",
        str(checkpoint),
        "--run-name",
        eval_run_name,
        "--ablation",
        "feature_hn_sev",
        "--sev-checkpoint",
        str(Path("runs") / train_run_name / "hn_sev.pt"),
        "--image-score-mode",
        "top5",
        "--image-score-source",
        "feature_raw_cosine",
        "--pixel-heatmap-source",
        "feature_raw",
    ]
    if feature_prior.is_file():
        parts.extend(["--feature-prior-checkpoint", str(feature_prior)])
    for arg_name, cli_name in [
        ("image_size", "--image-size"),
        ("seed", "--seed"),
    ]:
        value = source_args.get(arg_name)
        if value is not None:
            parts.extend([cli_name, str(value)])
    device = source_args.get("device")
    if device:
        parts.extend(["--device", str(device)])
    reconstruction_steps = source_args.get("reconstruction_steps")
    if reconstruction_steps is not None:
        parts.extend(["--reconstruction-steps", str(reconstruction_steps)])
    return " ".join(parts)


def evaluate_command_for_missing_variant(dataset: str, category: str, requirement: str) -> str:
    dataset = normalize_dataset(dataset)
    label = PRIMARY_ABLATION_FOR_REQUIREMENT[requirement]
    eval_run_name = f"hn_sev_exact_eval_{dataset}_{category}_{label}"
    run_dir = Path("runs") / eval_run_name
    return " ".join(
        [
            "python",
            "evaluate.py",
            "--pred_dir",
            str(run_dir),
            "--out",
            str(run_dir / "eval_metrics.json"),
        ]
    )


def missing_command_rows(root: Path, coverage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in coverage:
        dataset = str(row["dataset"])
        category = str(row["category"])
        for requirement in EXACT_REQUIREMENTS:
            if row.get(f"metric_{requirement}") is True:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "requirement": requirement,
                    "target_ablation": PRIMARY_ABLATION_FOR_REQUIREMENT[requirement],
                    "train_run_name": f"hn_sev_exact_{dataset}_{category}_{PRIMARY_ABLATION_FOR_REQUIREMENT[requirement]}",
                    "eval_run_name": f"hn_sev_exact_eval_{dataset}_{category}_{PRIMARY_ABLATION_FOR_REQUIREMENT[requirement]}",
                    "has_training_or_table_row": bool(row.get(f"exact_{requirement}")),
                    "has_metric_row": bool(row.get(f"metric_{requirement}")),
                    "train_command": command_for_missing_variant(
                        root, dataset, category, requirement
                    ),
                    "infer_command": inference_command_for_missing_variant(
                        root, dataset, category, requirement
                    ),
                    "evaluate_command": evaluate_command_for_missing_variant(
                        dataset, category, requirement
                    ),
                    "next_step": "train variant, run infer/evaluate with the resulting hn_sev.pt, then rerun this audit",
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


def write_outputs(root: Path, out_dir: Path) -> dict[str, Any]:
    rows = collect_rows(root)
    coverage = coverage_rows(rows)
    metrics = metric_summary(rows)
    summary = build_summary(rows, coverage)
    missing = missing_command_rows(root, coverage)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_input_ablation_coverage.csv", coverage)
    write_csv(out_dir / "table_input_ablation_metric_summary.csv", metrics)
    write_csv(out_dir / "table_missing_exact_ablation_commands.csv", missing)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("tables/hn_sev_input_ablation"))
    args = parser.parse_args()
    summary = write_outputs(args.root, args.out_dir)
    print(
        f"Wrote HN-SEV input-ablation audit to {args.out_dir} "
        f"(release_gate_passed={summary['release_gate_passed']})"
    )


if __name__ == "__main__":
    main()
