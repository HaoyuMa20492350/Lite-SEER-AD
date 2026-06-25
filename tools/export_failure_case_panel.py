from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CASES = {
    "grid": "lowest_anomaly_image_score",
    "pill": "lowest_anomaly_pixel_ap",
    "hazelnut": "lowest_anomaly_fixed_dice",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a post-hoc GT-aware failure panel from frozen held-out "
            "Lite-SEER-AD predictions. These cases are diagnostic only."
        )
    )
    parser.add_argument(
        "--selection-root",
        default="tables/synthetic_gate_fusion_aggregate_mvtec15",
    )
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument(
        "--out",
        default="tables/failure_case_panel_mvtec15",
    )
    parser.add_argument("--paper-figures", default="paper/figures")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fixed_dice(mask: np.ndarray, heatmap: np.ndarray, threshold: float) -> float:
    truth = np.asarray(mask) > 0
    prediction = np.asarray(heatmap) >= float(threshold)
    intersection = float(np.count_nonzero(truth & prediction))
    denominator = float(np.count_nonzero(truth) + np.count_nonzero(prediction))
    return 2.0 * intersection / max(denominator, 1e-8)


def pixel_ap(mask: np.ndarray, heatmap: np.ndarray) -> float:
    truth = (np.asarray(mask).reshape(-1) > 0).astype(np.uint8)
    if truth.sum() == 0:
        return float("nan")
    return float(
        average_precision_score(
            truth,
            np.asarray(heatmap, dtype=np.float64).reshape(-1),
        )
    )


def case_metrics(
    labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    heatmaps: np.ndarray,
    threshold: float,
) -> list[dict[str, float | int]]:
    rows = []
    for index in range(len(labels)):
        rows.append(
            {
                "index": index,
                "label": int(labels[index]),
                "image_score": float(image_scores[index]),
                "pixel_ap": pixel_ap(masks[index], heatmaps[index]),
                "fixed_dice": fixed_dice(
                    masks[index],
                    heatmaps[index],
                    threshold,
                ),
            }
        )
    return rows


def select_failure_index(
    metrics: list[dict[str, float | int]],
    criterion: str,
) -> dict[str, float | int]:
    anomalies = [row for row in metrics if int(row["label"]) == 1]
    if not anomalies:
        raise ValueError("Failure-case selection requires anomalous images")
    key = {
        "lowest_anomaly_image_score": "image_score",
        "lowest_anomaly_pixel_ap": "pixel_ap",
        "lowest_anomaly_fixed_dice": "fixed_dice",
    }.get(criterion)
    if key is None:
        raise ValueError(f"Unknown failure criterion: {criterion}")
    finite = [
        row
        for row in anomalies
        if np.isfinite(float(row[key]))
    ]
    if not finite:
        raise ValueError(f"No finite anomalous values for {criterion}")
    return min(finite, key=lambda row: float(row[key]))


def resolve_image_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def load_rgb(path: Path, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_panel(cases: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(
        len(cases),
        4,
        figsize=(11.6, 3.1 * len(cases)),
        squeeze=False,
    )
    column_titles = ("Input", "Ground truth", "Anomaly heatmap", "Fixed mask")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11)
    for row_index, case in enumerate(cases):
        image = case["image"]
        mask = case["mask"] > 0
        heatmap = case["heatmap"]
        prediction = heatmap >= case["threshold"]
        finite = heatmap[np.isfinite(heatmap)]
        vmin = float(np.percentile(finite, 1.0))
        vmax = float(np.percentile(finite, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1e-8

        axes[row_index, 0].imshow(image)
        axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].imshow(image)
        axes[row_index, 2].imshow(
            heatmap,
            cmap="turbo",
            alpha=0.62,
            vmin=vmin,
            vmax=vmax,
        )
        axes[row_index, 3].imshow(image)
        axes[row_index, 3].imshow(
            np.ma.masked_where(~prediction, prediction),
            cmap="Reds",
            alpha=0.55,
            vmin=0,
            vmax=1,
        )
        if mask.any():
            axes[row_index, 3].contour(
                mask.astype(np.uint8),
                levels=[0.5],
                colors=["#00FF66"],
                linewidths=1.2,
            )
        axes[row_index, 0].set_ylabel(
            f"{case['category']}\n"
            f"score={case['image_score']:.3f}\n"
            f"AP={case['pixel_ap']:.3f}, Dice={case['fixed_dice']:.3f}",
            fontsize=9,
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Post-hoc failure cases from frozen seed-7 held-out predictions",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    split_root = Path(args.selection_root) / f"seed{args.split_seed}"
    selections = {
        row["category"]: row
        for row in read_csv(split_root / "selection.csv")
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    index_rows = []
    for category, criterion in DEFAULT_CASES.items():
        selection = selections[category]
        run_dir = (
            split_root
            / "selected_runs"
            / f"{category}_{selection['selected_candidate']}_heldout"
        )
        policy = json.loads(
            (run_dir / "pixel_threshold_policy.json").read_text(
                encoding="utf-8"
            )
        )
        threshold = float(policy["threshold"])
        with np.load(run_dir / "predictions.npz") as payload:
            labels = payload["labels"]
            image_scores = payload["image_scores"]
            masks = payload["masks"]
            heatmap_key = (
                "detection_heatmaps"
                if "detection_heatmaps" in payload.files
                else "heatmaps"
            )
            heatmaps = payload[heatmap_key]
            paths = payload["paths"].astype(str)
            selected = select_failure_index(
                case_metrics(
                    labels,
                    image_scores,
                    masks,
                    heatmaps,
                    threshold,
                ),
                criterion,
            )
            index = int(selected["index"])
            image_path = resolve_image_path(paths[index])
            image = load_rgb(image_path, tuple(heatmaps[index].shape))
            case = {
                "category": category,
                "criterion": criterion,
                "index": index,
                "path": str(image_path),
                "image": image,
                "mask": masks[index],
                "heatmap": heatmaps[index],
                "threshold": threshold,
                "image_score": float(selected["image_score"]),
                "pixel_ap": float(selected["pixel_ap"]),
                "fixed_dice": float(selected["fixed_dice"]),
            }
            cases.append(case)
            index_rows.append(
                {
                    "category": category,
                    "criterion": criterion,
                    "split_seed": args.split_seed,
                    "selected_candidate": selection["selected_candidate"],
                    "run_dir": str(run_dir),
                    "prediction_index": index,
                    "image_path": str(image_path),
                    "image_score": case["image_score"],
                    "pixel_ap": case["pixel_ap"],
                    "fixed_dice": case["fixed_dice"],
                    "fixed_threshold": threshold,
                    "heatmap_key": heatmap_key,
                    "case_selection_role": "post_hoc_failure_analysis",
                    "uses_real_anomaly_labels_for_method_selection": False,
                    "uses_real_anomaly_masks_for_method_selection": False,
                    "uses_real_anomaly_masks_for_case_selection": True,
                }
            )

    figure_path = out_dir / "fig_frozen_failure_cases.png"
    render_panel(cases, figure_path)
    write_csv(out_dir / "table_failure_case_index.csv", index_rows)
    summary = {
        "protocol": "post_hoc_gt_aware_failure_case_audit_v1",
        "split_seed": args.split_seed,
        "categories": list(DEFAULT_CASES),
        "figure": str(figure_path),
        "uses_real_anomaly_labels_for_method_selection": False,
        "uses_real_anomaly_masks_for_method_selection": False,
        "uses_real_anomaly_masks_for_case_selection": True,
        "case_selection_role": "post_hoc_failure_analysis_only",
        "cases": index_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    paper_dir = Path(args.paper_figures)
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper_path = paper_dir / figure_path.name
    shutil.copy2(figure_path, paper_path)
    print(
        json.dumps(
            {
                **summary,
                "paper_figure": str(paper_path),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
