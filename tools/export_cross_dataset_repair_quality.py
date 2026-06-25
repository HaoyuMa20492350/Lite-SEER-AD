from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer_ad_v2.evaluation.repair_quality import (
    SDR_KEYS,
    image_repair_quality,
    roi_ground_truth_records,
    sdr_gt_summary,
)


DATASETS = {
    "mvtec15": {
        "prefix": "feature_mvtec15",
        "categories": [
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
        ],
    },
    "visa": {
        "prefix": "feature_visa",
        "categories": [
            "candle",
            "capsules",
            "cashew",
            "chewinggum",
            "fryum",
            "macaroni1",
            "macaroni2",
            "pcb1",
            "pcb2",
            "pcb3",
            "pcb4",
            "pipe_fryum",
        ],
    },
    "mpdd": {
        "prefix": "feature_mpdd",
        "categories": [
            "bracket_black",
            "bracket_brown",
            "bracket_white",
            "connector",
            "metal_plate",
            "tubes",
        ],
    },
}

DETECTOR_METRICS = [
    "image_auroc",
    "pixel_auroc",
    "aupro",
    "pixel_ap",
    "dice",
    "fprr",
    "rdc",
    "sdr_mean",
    "pareto_area",
]
QUALITY_METRICS = [
    "psnr",
    "ssim",
    "background_psnr",
    "background_mae",
    "foreground_mae",
    "boundary_consistency",
    "identity",
]
EVIDENCE_SCOPE = "independent_module_run_128_not_main_detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export cross-dataset repair quality and SDR-GT evidence from the "
            "frozen HN-SEV/CRV module runs."
        )
    )
    parser.add_argument("--runs", default="runs")
    parser.add_argument(
        "--out",
        default="tables/feature_first_fusion_aggregate_paper_package",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_efficiency(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["metric"]: float(row["value"])
            for row in csv.DictReader(handle)
        }


def finite_mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = np.asarray(
        [float(row[key]) for row in rows if key in row],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def summarize_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    anomaly = [row for row in rows if int(row["label"]) == 1]
    summary: dict[str, Any] = {
        "image_count": len(rows),
        "anomaly_image_count": len(anomaly),
    }
    for key in QUALITY_METRICS:
        summary[f"mean_{key}"] = finite_mean(rows, key)
        summary[f"anomaly_mean_{key}"] = finite_mean(anomaly, key)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_dir(runs_root: Path, dataset: str, category: str) -> Path:
    prefix = str(DATASETS[dataset]["prefix"])
    return runs_root / f"{prefix}_{category}_feature_hn_sev_crv"


def collect_category(
    runs_root: Path,
    dataset: str,
    category: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = run_dir(runs_root, dataset, category)
    required = [
        root / "predictions.npz",
        root / "roi_budget.json",
        root / "metrics.json",
        root / "efficiency.csv",
        root / "images",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{dataset}/{category} is missing required evidence: {missing}"
        )

    with np.load(root / "predictions.npz", allow_pickle=False) as predictions:
        labels = np.asarray(predictions["labels"]).reshape(-1)
        masks = np.asarray(predictions["masks"])
    roi_rows = read_json(root / "roi_budget.json")
    roi_records = roi_ground_truth_records(masks, labels, roi_rows)

    image_dirs = sorted(
        path for path in (root / "images").iterdir() if path.is_dir()
    )
    if len(image_dirs) != len(labels):
        raise ValueError(
            f"{dataset}/{category}: {len(image_dirs)} image directories for "
            f"{len(labels)} prediction labels"
        )
    quality_records = []
    for image_index, image_dir in enumerate(image_dirs):
        input_path = image_dir / "input.png"
        repair_path = image_dir / "repair.png"
        ground_truth_path = image_dir / "ground_truth.png"
        missing_images = [
            str(path)
            for path in (input_path, repair_path, ground_truth_path)
            if not path.exists()
        ]
        if missing_images:
            raise FileNotFoundError(
                f"{dataset}/{category}/{image_dir.name}: {missing_images}"
            )
        original = np.asarray(Image.open(input_path).convert("RGB"))
        repaired = np.asarray(Image.open(repair_path).convert("RGB"))
        saved_mask = np.asarray(Image.open(ground_truth_path).convert("L")) > 0
        prediction_mask = np.asarray(masks[image_index]) > 0
        if saved_mask.shape != prediction_mask.shape:
            raise ValueError(
                f"{dataset}/{category}/{image_dir.name}: saved and prediction "
                f"mask shapes differ ({saved_mask.shape} != {prediction_mask.shape})"
            )
        if not np.array_equal(saved_mask, prediction_mask):
            raise ValueError(
                f"{dataset}/{category}/{image_dir.name}: saved ground truth "
                "does not match predictions.npz"
            )
        quality_records.append(
            {
                "dataset": dataset,
                "category": category,
                "image_index": image_index,
                "label": int(labels[image_index]),
                **image_repair_quality(original, repaired, saved_mask),
            }
        )

    metrics = read_json(root / "metrics.json")
    efficiency = read_efficiency(root / "efficiency.csv")
    category_summary = {
        "dataset": dataset,
        "category": category,
        "run": root.as_posix(),
        "evidence_scope": EVIDENCE_SCOPE,
        **{key: metrics.get(key) for key in DETECTOR_METRICS},
        "latency_ms_mean": efficiency.get("latency_ms_mean"),
        "nfe_mean": efficiency.get("nfe_mean"),
        **summarize_quality(quality_records),
        **sdr_gt_summary(roi_records),
    }
    return category_summary, quality_records, roi_records


def plot_sdr_gt(path: Path, roi_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    anomaly = [row for row in roi_rows if int(row["label"]) == 1]
    datasets = list(DATASETS)
    colors = {"mvtec15": "#2b66b1", "visa": "#287a4b", "mpdd": "#a96800"}
    figure, axes = plt.subplots(1, len(SDR_KEYS), figsize=(15, 3.8), sharey=True)
    for axis, key in zip(axes, SDR_KEYS):
        for dataset in datasets:
            rows = [row for row in anomaly if row["dataset"] == dataset]
            axis.scatter(
                [row[key] for row in rows],
                [row["gt_fraction"] for row in rows],
                s=9,
                alpha=0.28,
                color=colors[dataset],
                label=dataset if key == SDR_KEYS[0] else None,
            )
        axis.set_title(key.replace("_", " ").upper())
        axis.set_xlabel("repair diagnostic")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("ground-truth fraction in ROI")
    figure.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "SDR-GT diagnostic correlation across frozen 128px module runs",
        y=0.98,
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.92))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    category_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    for dataset, specification in DATASETS.items():
        for category in specification["categories"]:
            category_summary, category_quality, category_roi = collect_category(
                runs_root,
                dataset,
                str(category),
            )
            category_rows.append(category_summary)
            quality_rows.extend(category_quality)
            roi_rows.extend(
                {
                    "dataset": dataset,
                    "category": category,
                    **row,
                }
                for row in category_roi
            )

    quality_summary_rows = []
    sdr_summary_rows = []
    grouped_quality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_roi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in quality_rows:
        grouped_quality[str(row["dataset"])].append(row)
        grouped_quality["all"].append(row)
    for row in roi_rows:
        grouped_roi[str(row["dataset"])].append(row)
        grouped_roi["all"].append(row)
    for dataset in [*DATASETS, "all"]:
        quality_summary_rows.append(
            {
                "dataset": dataset,
                "categories": (
                    len(DATASETS[dataset]["categories"])
                    if dataset in DATASETS
                    else sum(
                        len(specification["categories"])
                        for specification in DATASETS.values()
                    )
                ),
                "evidence_scope": EVIDENCE_SCOPE,
                **summarize_quality(grouped_quality[dataset]),
            }
        )
        sdr_summary_rows.append(
            {
                "dataset": dataset,
                "categories": (
                    len(DATASETS[dataset]["categories"])
                    if dataset in DATASETS
                    else sum(
                        len(specification["categories"])
                        for specification in DATASETS.values()
                    )
                ),
                "evidence_scope": EVIDENCE_SCOPE,
                **sdr_gt_summary(grouped_roi[dataset]),
            }
        )

    correlation_fields = [
        key
        for key in category_rows[0]
        if "gt_fraction_" in key or "gt_hit_" in key
    ]
    correlation_category_rows = [
        {
            "dataset": row["dataset"],
            "category": row["category"],
            "evidence_scope": row["evidence_scope"],
            "roi_count": row["roi_count"],
            "anomaly_roi_count": row["anomaly_roi_count"],
            "gt_hit_rate": row["gt_hit_rate"],
            "positive_sdr_gt_hit_rate": row["positive_sdr_gt_hit_rate"],
            **{key: row[key] for key in correlation_fields},
        }
        for row in category_rows
    ]

    write_csv(out_dir / "table_repair_quality_by_category.csv", category_rows)
    write_csv(out_dir / "table_repair_quality_summary.csv", quality_summary_rows)
    write_csv(
        out_dir / "table_sdr_gt_correlation_by_category.csv",
        correlation_category_rows,
    )
    write_csv(
        out_dir / "table_sdr_gt_correlation_summary.csv",
        sdr_summary_rows,
    )
    plot_sdr_gt(out_dir / "fig_sdr_gt_correlation.png", roi_rows)

    overall_quality = next(
        row for row in quality_summary_rows if row["dataset"] == "all"
    )
    overall_sdr = next(row for row in sdr_summary_rows if row["dataset"] == "all")
    sdr_spearman = float(overall_sdr["sdr_gt_fraction_spearman"])
    supports_gt_alignment = bool(np.isfinite(sdr_spearman) and sdr_spearman > 0.1)
    claim_decision = (
        "retain_gt_aligned_repair_diagnostic"
        if supports_gt_alignment
        else "downgrade_to_visualization_only"
    )
    summary = {
        "datasets": list(DATASETS),
        "categories": len(category_rows),
        "expected_categories": 33,
        "missing_categories": [],
        "evidence_scope": EVIDENCE_SCOPE,
        "detector_claim": (
            "The frozen detector metrics are context only. These independent "
            "module runs do not establish AP/Dice gains."
        ),
        "quality": overall_quality,
        "sdr_gt": overall_sdr,
        "correlation_supports_gt_alignment": supports_gt_alignment,
        "claim_decision": claim_decision,
        "claim_boundary": (
            (
                "SDR-GT supports a limited repair-diagnostic association, but "
                "is not used to claim a main-detector improvement."
            )
            if supports_gt_alignment
            else (
                "Pooled SDR-GT alignment is not positive. CRV is restricted "
                "to visualization and post-hoc inspection, with no GT-aligned "
                "repair or main-detector improvement claim."
            )
        ),
        "repair_quality_interpretation": (
            "PSNR, SSIM, background error, foreground change, and boundary "
            "consistency measure structural fidelity and edit locality. No "
            "paired anomaly-free target exists, so they do not prove semantic "
            "defect removal."
        ),
    }
    (out_dir / "repair_quality_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    print(
        f"Wrote repair evidence for {len(category_rows)} categories, "
        f"{len(quality_rows)} images, and {len(roi_rows)} ROIs to {out_dir}"
    )


if __name__ == "__main__":
    main()
