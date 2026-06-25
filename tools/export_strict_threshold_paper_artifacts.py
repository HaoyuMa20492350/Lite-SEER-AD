from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METRICS = (
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "pixel_ap",
    "f1",
    "iou",
    "dice",
    "oracle_f1",
    "oracle_iou",
    "oracle_dice",
    "normal_pixel_fpr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export paper tables and figures from strict fixed-threshold "
            "held-out metrics."
        )
    )
    parser.add_argument(
        "--metrics",
        default="tables/strict_fixed_threshold/strict_selected_metrics.csv",
    )
    parser.add_argument(
        "--out",
        default="tables/strict_fixed_threshold_paper",
    )
    parser.add_argument(
        "--paper-figures",
        default="paper/figures",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_metrics(
    rows: Iterable[dict[str, str]],
    *,
    prefix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = list(rows)
    out: dict[str, Any] = dict(prefix or {})
    out["runs"] = len(selected)
    for metric in METRICS:
        values = [
            float(row[metric])
            for row in selected
            if row.get(metric) not in {"", None}
            and np.isfinite(float(row[metric]))
        ]
        out[metric] = float(np.mean(values)) if values else None
    fixed_dice = out.get("dice")
    oracle_dice = out.get("oracle_dice")
    out["oracle_dice_gap"] = (
        float(oracle_dice - fixed_dice)
        if fixed_dice is not None and oracle_dice is not None
        else None
    )
    return out


def grouped_means(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [
        mean_metrics(
            grouped,
            prefix={key: value for key, value in zip(keys, group_key)},
        )
        for group_key, grouped in sorted(groups.items())
    ]


def validate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Strict fixed-threshold metrics are empty")
    required = {
        "dataset",
        "split_seed",
        "category",
        "threshold_protocol",
        "normal_pixel_fpr",
        "dice",
        "oracle_dice",
    }
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"Strict metrics are missing columns: {sorted(missing)}")
    duplicate_count = len(rows) - len(
        {
            (row["dataset"], row["split_seed"], row["category"])
            for row in rows
        }
    )
    protocols = sorted({row["threshold_protocol"] for row in rows})
    max_fpr = max(float(row["normal_pixel_fpr"]) for row in rows)
    uses_labels = any(
        row.get("uses_real_anomaly_labels_for_threshold", "").lower()
        == "true"
        for row in rows
    )
    uses_masks = any(
        row.get("uses_real_anomaly_masks_for_threshold", "").lower() == "true"
        for row in rows
    )
    if duplicate_count:
        raise ValueError(f"Strict metrics contain {duplicate_count} duplicate rows")
    if protocols != ["synthetic_normal_fixed_threshold_v1"]:
        raise ValueError(f"Unexpected threshold protocols: {protocols}")
    if max_fpr > 0.005:
        raise ValueError(f"Normal-pixel FPR cap exceeded: {max_fpr}")
    if uses_labels or uses_masks:
        raise ValueError("Strict threshold evidence uses real anomaly supervision")
    return {
        "rows": len(rows),
        "unique_category_seed_records": len(rows),
        "datasets": sorted({row["dataset"] for row in rows}),
        "categories": len({(row["dataset"], row["category"]) for row in rows}),
        "split_seeds": sorted({row["split_seed"] for row in rows}),
        "threshold_protocols": protocols,
        "max_normal_pixel_fpr": max_fpr,
        "uses_real_anomaly_labels_for_threshold": uses_labels,
        "uses_real_anomaly_masks_for_threshold": uses_masks,
    }


def _dataset_label(dataset: str) -> str:
    return {
        "mvtec15": "MVTec AD",
        "visa": "VisA",
        "mpdd": "MPDD",
        "overall": "Overall",
    }.get(dataset, dataset)


def dataset_order(row: dict[str, Any]) -> int:
    return {"mvtec15": 0, "visa": 1, "mpdd": 2, "overall": 3}.get(
        str(row.get("dataset", "")),
        99,
    )


def export_figures(
    dataset_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for paper figures") from exc

    ordered = sorted(
        dataset_rows,
        key=dataset_order,
    )
    labels = [_dataset_label(str(row["dataset"])) for row in ordered]
    fixed = np.asarray([float(row["dice"]) for row in ordered])
    oracle = np.asarray([float(row["oracle_dice"]) for row in ordered])
    x = np.arange(len(labels))
    width = 0.36

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(x - width / 2, fixed, width, label="Fixed threshold")
    axis.bar(x + width / 2, oracle, width, label="Test-GT oracle")
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, max(0.6, float(oracle.max()) * 1.15))
    axis.set_ylabel("Dice")
    axis.set_title("Deployable fixed-threshold Dice vs. oracle Dice")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    fixed_path = out_dir / "fig_fixed_vs_oracle_dice.png"
    figure.savefig(fixed_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    fprs = np.asarray(
        [float(row["normal_pixel_fpr"]) * 100.0 for row in category_rows]
    )
    gaps = np.asarray(
        [float(row["oracle_dice_gap"]) for row in category_rows]
    )
    colors = [
        {"mvtec15": "#4472C4", "visa": "#ED7D31", "mpdd": "#70AD47"}.get(
            str(row["dataset"]),
            "#808080",
        )
        for row in category_rows
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    axes[0].hist(fprs, bins=12, color="#4472C4", alpha=0.85)
    axes[0].axvline(0.5, color="#C00000", linestyle="--", label="0.5% cap")
    axes[0].set_xlabel("Normal-pixel FPR (%)")
    axes[0].set_ylabel("Categories")
    axes[0].set_title("Frozen-threshold normal FPR")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].scatter(fprs, gaps, c=colors, alpha=0.85, edgecolors="none")
    for dataset, color in (
        ("mvtec15", "#4472C4"),
        ("visa", "#ED7D31"),
        ("mpdd", "#70AD47"),
    ):
        axes[1].scatter(
            [],
            [],
            color=color,
            label=_dataset_label(dataset),
        )
    axes[1].axvline(0.5, color="#C00000", linestyle="--")
    axes[1].axhline(0.0, color="#707070", linewidth=0.8)
    axes[1].set_xlabel("Normal-pixel FPR (%)")
    axes[1].set_ylabel("Oracle Dice - fixed Dice")
    axes[1].set_title("Optimism from test-GT thresholding")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")
    figure.tight_layout()
    audit_path = out_dir / "fig_threshold_fpr_and_oracle_gap.png"
    figure.savefig(audit_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return [str(fixed_path), str(audit_path)]


def markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Dataset | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | "
        "Fixed Dice | Oracle Dice | Gap |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        values = dict(row)
        values["dataset"] = _dataset_label(str(row["dataset"]))
        lines.append(
            "| {dataset} | {image_auroc:.4f} | {pixel_auroc:.4f} | "
            "{aupro:.4f} | {pixel_ap:.4f} | {dice:.4f} | "
            "{oracle_dice:.4f} | {oracle_dice_gap:.4f} |".format(
                **values,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(metrics_path)
    audit = validate_rows(rows)
    dataset_rows = sorted(grouped_means(rows, ("dataset",)), key=dataset_order)
    seed_rows = grouped_means(rows, ("dataset", "split_seed"))
    category_rows = grouped_means(rows, ("dataset", "category"))
    overall = mean_metrics(rows, prefix={"dataset": "overall"})

    write_csv(out_dir / "table_strict_mean_by_dataset.csv", dataset_rows)
    write_csv(out_dir / "table_strict_mean_by_seed.csv", seed_rows)
    write_csv(out_dir / "table_strict_mean_by_category.csv", category_rows)
    write_csv(out_dir / "table_strict_overall.csv", [overall])
    figures = export_figures(dataset_rows, category_rows, out_dir)
    paper_figure_dir = Path(args.paper_figures)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    paper_figures = []
    for figure_value in figures:
        source = Path(figure_value)
        target = paper_figure_dir / source.name
        shutil.copy2(source, target)
        paper_figures.append(str(target))
    (out_dir / "table_strict_mean_by_dataset.md").write_text(
        markdown_table(dataset_rows + [overall]),
        encoding="utf-8",
    )
    summary = {
        **audit,
        "source_metrics": str(metrics_path),
        "overall": overall,
        "figures": figures,
        "paper_figures": paper_figures,
        "tables": [
            str(out_dir / "table_strict_mean_by_dataset.csv"),
            str(out_dir / "table_strict_mean_by_seed.csv"),
            str(out_dir / "table_strict_mean_by_category.csv"),
            str(out_dir / "table_strict_overall.csv"),
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
