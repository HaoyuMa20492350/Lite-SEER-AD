"""Export an HN-SEV retention, calibration, and taxonomy audit.

The audit uses existing ROI logs plus dataset masks. It estimates whether
HN-SEV suppresses false-positive regions while retaining GT-overlapping ROIs.
Input-source ablations are reported as missing unless their dedicated tables
exist, so this artifact cannot silently overstate HN-SEV completion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


RUN_PATTERN = re.compile(
    r"^feature_fixedpixel_(?P<dataset>mvtec15|visa|mpdd)_(?P<category>.+)_feature_pixel_policy$"
)
ANALYSIS_SIZE = (128, 128)
PERIODIC_TEXTURE_CATEGORIES = {
    "grid",
    "carpet",
    "tile",
    "wood",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "macaroni1",
    "macaroni2",
}
REFLECTIVE_CATEGORIES = {
    "bottle",
    "hazelnut",
    "metal_nut",
    "capsule",
    "capsules",
    "candle",
    "cashew",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def normalize_rel_path(path: str) -> Path:
    return Path(*Path(path.replace("\\", "/")).parts)


def collect_run_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in (root / "runs").glob("feature_fixedpixel_*_feature_pixel_policy")
        if path.is_dir() and RUN_PATTERN.match(path.name)
    )


def parse_run_dir(path: Path) -> tuple[str, str] | None:
    match = RUN_PATTERN.match(path.name)
    if not match:
        return None
    return match.group("dataset"), match.group("category")


def resolve_source(root: Path, source_path: str) -> Path:
    rel = normalize_rel_path(source_path)
    return (root / rel).resolve()


def resolve_mask(root: Path, source_path: str) -> tuple[str, Path | None]:
    rel = normalize_rel_path(source_path)
    parts = list(rel.parts)
    lower_parts = [part.lower() for part in parts]

    if "mvtec-ad" in lower_parts and "test" in lower_parts:
        test_idx = lower_parts.index("test")
        defect = parts[test_idx + 1]
        if defect.lower() == "good":
            return "normal", None
        category_idx = test_idx - 1
        category = parts[category_idx]
        stem = Path(parts[-1]).stem
        mask = (
            root
            / Path(*parts[:category_idx])
            / category
            / "ground_truth"
            / defect
            / f"{stem}_mask.png"
        )
        return "anomaly", mask

    if "mpdd" in lower_parts and "test" in lower_parts:
        test_idx = lower_parts.index("test")
        defect = parts[test_idx + 1]
        if defect.lower() == "good":
            return "normal", None
        category_idx = test_idx - 1
        category = parts[category_idx]
        stem = Path(parts[-1]).stem
        mask = (
            root
            / Path(*parts[:category_idx])
            / category
            / "ground_truth"
            / defect
            / f"{stem}_mask.png"
        )
        return "anomaly", mask

    if "visa" in lower_parts and "images" in lower_parts:
        image_idx = lower_parts.index("images")
        split = parts[image_idx + 1]
        if split.lower() == "normal":
            return "normal", None
        stem = Path(parts[-1]).stem
        mask = root / Path(*parts[:image_idx]) / "Masks" / "Anomaly" / f"{stem}.png"
        return "anomaly", mask

    return "unresolved", None


def load_mask(mask_path: Path | None, size: tuple[int, int]) -> np.ndarray | None:
    if mask_path is None or not mask_path.exists():
        return None
    mask = Image.open(mask_path).convert("L").resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def bbox_intersects_mask(bbox: list[Any], mask: np.ndarray | None) -> bool:
    if mask is None:
        return False
    x1, y1, x2, y2 = [int(round(as_float(value))) for value in bbox]
    h, w = mask.shape
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return False
    return bool(mask[y1:y2, x1:x2].any())


def is_kept(row: dict[str, Any]) -> bool:
    return int(as_float(row.get("hn_sev_positive"), 0.0)) == 1


def confidence(row: dict[str, Any]) -> float:
    return max(0.0, min(1.0, as_float(row.get("hn_sev_confidence"), 0.0)))


def taxonomy(category: str, bbox: list[Any], is_normal: bool) -> str:
    x1, y1, x2, y2 = [as_float(value) for value in bbox]
    edge_margin = 6.0
    if (
        x1 <= edge_margin
        or y1 <= edge_margin
        or x2 >= ANALYSIS_SIZE[0] - edge_margin
        or y2 >= ANALYSIS_SIZE[1] - edge_margin
    ):
        return "edge_drift"
    if category in PERIODIC_TEXTURE_CATEGORIES:
        return "periodic_texture"
    if category in REFLECTIVE_CATEGORIES:
        return "reflection"
    if is_normal:
        return "background_shift"
    return "background_shift"


def calibration_metrics(labels: list[int], probs: list[float], bins: int = 10) -> dict[str, float]:
    if not labels:
        return {"ece": 0.0, "brier": 0.0}
    labels_arr = np.asarray(labels, dtype=np.float64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    brier = float(np.mean((probs_arr - labels_arr) ** 2))
    ece = 0.0
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        if idx == bins - 1:
            mask = (probs_arr >= low) & (probs_arr <= high)
        else:
            mask = (probs_arr >= low) & (probs_arr < high)
        if not mask.any():
            continue
        acc = float(labels_arr[mask].mean())
        conf = float(probs_arr[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return {"ece": float(ece), "brier": brier}


def image_key(row: dict[str, Any]) -> str:
    return str(row.get("source_path", "")) or f"image_{row.get('image_index', '')}"


def classify_rows(
    root: Path, dataset: str, category: str, roi_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mask_cache: dict[str, tuple[str, np.ndarray | None, bool]] = {}
    unresolved = Counter()
    classified: list[dict[str, Any]] = []
    for row in roi_rows:
        source = str(row.get("source_path", ""))
        if source not in mask_cache:
            label, mask_path = resolve_mask(root, source)
            mask = load_mask(mask_path, ANALYSIS_SIZE) if label == "anomaly" else None
            resolved = label == "normal" or (label == "anomaly" and mask is not None)
            if not resolved:
                unresolved[label or "missing_mask"] += 1
            mask_cache[source] = (label, mask, resolved)
        label, mask, resolved = mask_cache[source]
        gt_roi = label == "anomaly" and bbox_intersects_mask(row.get("bbox", []), mask)
        normal_or_background = label == "normal" or (label == "anomaly" and not gt_roi)
        classified.append(
            {
                **row,
                "dataset": dataset,
                "category": category,
                "image_key": image_key(row),
                "resolved_gt": resolved,
                "image_label": label,
                "gt_roi": bool(gt_roi),
                "hn_sev_kept": is_kept(row),
                "confidence": confidence(row),
                "taxonomy": taxonomy(category, row.get("bbox", []), normal_or_background),
            }
        )
    return classified, dict(unresolved)


def summarize_classified(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = [row for row in rows if row["resolved_gt"]]
    gt_rois = [row for row in resolved if row["gt_roi"]]
    normal_bg_rois = [row for row in resolved if not row["gt_roi"]]
    kept_gt = [row for row in gt_rois if row["hn_sev_kept"]]
    suppressed_bg = [row for row in normal_bg_rois if not row["hn_sev_kept"]]

    anomaly_images = {
        row["image_key"]
        for row in resolved
        if row["image_label"] == "anomaly"
    }
    before_recalled = {
        row["image_key"]
        for row in gt_rois
    }
    after_recalled = {
        row["image_key"]
        for row in kept_gt
    }
    labels = [1 if row["gt_roi"] else 0 for row in resolved]
    probs = [float(row["confidence"]) for row in resolved]
    calib = calibration_metrics(labels, probs)
    taxonomy_counts = Counter(row["taxonomy"] for row in normal_bg_rois)
    return {
        "roi_rows_resolved": len(resolved),
        "gt_roi_candidates": len(gt_rois),
        "gt_roi_kept": len(kept_gt),
        "background_or_normal_roi_candidates": len(normal_bg_rois),
        "background_or_normal_roi_suppressed": len(suppressed_bg),
        "tp_retention": safe_div(len(kept_gt), len(gt_rois)),
        "background_suppression_rate": safe_div(len(suppressed_bg), len(normal_bg_rois)),
        "roi_recall_before_hn_sev": safe_div(len(before_recalled), len(anomaly_images)),
        "roi_recall_after_hn_sev": safe_div(len(after_recalled), len(anomaly_images)),
        "roi_recall_delta": safe_div(len(after_recalled), len(anomaly_images))
        - safe_div(len(before_recalled), len(anomaly_images)),
        "ece": calib["ece"],
        "brier": calib["brier"],
        "taxonomy_reflection": taxonomy_counts.get("reflection", 0),
        "taxonomy_edge_drift": taxonomy_counts.get("edge_drift", 0),
        "taxonomy_periodic_texture": taxonomy_counts.get("periodic_texture", 0),
        "taxonomy_background_shift": taxonomy_counts.get("background_shift", 0),
    }


def select_case_rows(rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    buckets = [
        ("tp_kept", lambda row: row["gt_roi"] and row["hn_sev_kept"]),
        ("tp_suppressed", lambda row: row["gt_roi"] and not row["hn_sev_kept"]),
        ("background_suppressed", lambda row: (not row["gt_roi"]) and not row["hn_sev_kept"]),
        ("background_kept", lambda row: (not row["gt_roi"]) and row["hn_sev_kept"]),
    ]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, Any]] = set()
    for case_type, predicate in buckets:
        candidates = sorted(
            [row for row in rows if row["resolved_gt"] and predicate(row)],
            key=lambda row: float(row["confidence"]),
            reverse=True,
        )
        for row in candidates[: max(1, limit // len(buckets))]:
            key = (row["image_key"], row.get("roi_id"))
            if key in seen:
                continue
            seen.add(key)
            selected.append({**row, "case_type": case_type})
            if len(selected) >= limit:
                return selected
    return selected


def render_case_panel(root: Path, row: dict[str, Any], out_path: Path) -> bool:
    source_path = resolve_source(root, str(row.get("source_path", "")))
    if not source_path.exists():
        return False
    image = Image.open(source_path).convert("RGB").resize(ANALYSIS_SIZE)
    label, mask_path = resolve_mask(root, str(row.get("source_path", "")))
    mask = load_mask(mask_path, ANALYSIS_SIZE) if label == "anomaly" else None
    if mask is not None:
        overlay = Image.new("RGBA", ANALYSIS_SIZE, (0, 0, 0, 0))
        overlay_arr = np.asarray(overlay).copy()
        overlay_arr[mask] = np.asarray([0, 96, 255, 80], dtype=np.uint8)
        image = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay_arr)).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [int(round(as_float(value))) for value in row.get("bbox", [])]
    color = (0, 180, 0) if row["hn_sev_kept"] else (220, 20, 20)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    draw.text(
        (2, 2),
        f"{row['case_type']} c={row['confidence']:.2f}",
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return True


def build_audit(root: Path, out_dir: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    unresolved_total = Counter()
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
        classified, unresolved = classify_rows(root, dataset, category, roi_rows)
        all_rows.extend(classified)
        unresolved_total.update(unresolved)
        category_rows.append(
            {
                "dataset": dataset,
                "category": category,
                "run_dir": run_dir.as_posix(),
                **summarize_classified(classified),
                "unresolved_rows": sum(unresolved.values()),
            }
        )

    overall = summarize_classified(all_rows)
    cases = select_case_rows(all_rows)
    case_rows: list[dict[str, Any]] = []
    if out_dir is not None:
        for idx, row in enumerate(cases):
            rel_panel = Path("case_panels") / f"hn_sev_case_{idx:02d}.png"
            panel_path = out_dir / rel_panel
            rendered = render_case_panel(root, row, panel_path)
            case_rows.append(
                {
                    "case_id": idx,
                    "case_type": row["case_type"],
                    "dataset": row["dataset"],
                    "category": row["category"],
                    "source_path": row.get("source_path", ""),
                    "roi_id": row.get("roi_id", ""),
                    "gt_roi": row["gt_roi"],
                    "hn_sev_kept": row["hn_sev_kept"],
                    "confidence": row["confidence"],
                    "taxonomy": row["taxonomy"],
                    "panel_path": rel_panel.as_posix() if rendered else "",
                }
            )

    ablation_paths = [
        root / "tables/hn_sev_input_ablation/summary.json",
        root / "tables/hn_sev_full_audit/input_ablation_summary.json",
    ]
    input_ablation_summaries = [
        read_json(path) for path in ablation_paths if path.exists()
    ]
    input_ablation_ready = any(
        summary.get("release_gate_passed") is True
        for summary in input_ablation_summaries
        if isinstance(summary, dict)
    )
    release_gate_passed = (
        overall["tp_retention"] >= 0.95
        and overall["roi_recall_after_hn_sev"] >= overall["roi_recall_before_hn_sev"] - 0.02
        and overall["background_suppression_rate"] > 0.0
        and input_ablation_ready
    )
    if not input_ablation_ready:
        release_gate_reason = (
            "Retention/calibration/taxonomy are computed from ROI logs and masks; "
            "complete exact input-source ablations are still required."
        )
    elif release_gate_passed:
        release_gate_reason = (
            "Retention, calibration, taxonomy, and input ablations are present; "
            "TP retention and ROI recall satisfy the recall-safe gate."
        )
    else:
        release_gate_reason = (
            "Retention, calibration, taxonomy, and input ablations are present, "
            "but TP retention or ROI recall fails the recall-safe gate; limit "
            "HN-SEV to false-positive suppression."
        )
    summary = {
        "schema": "lite-seer-ad-hn-sev-audit-v1",
        "evidence_level": "roi_mask_retention_calibration_v1",
        "release_gate_passed": release_gate_passed,
        "release_gate_reason": release_gate_reason,
        "datasets": sorted({row["dataset"] for row in category_rows}),
        "categories": len(category_rows),
        "roi_rows": len(all_rows),
        "unresolved_rows": int(sum(unresolved_total.values())),
        "unresolved_breakdown": dict(unresolved_total),
        "input_ablation_ready": input_ablation_ready,
        "input_ablation_summary_paths": [
            path.relative_to(root).as_posix() for path in ablation_paths if path.exists()
        ],
        "before_after_visualizations": len(case_rows),
        "overall": {
            key: (None if isinstance(value, float) and math.isnan(value) else value)
            for key, value in overall.items()
        },
        "required_for_release": [
            "synthetic-only HN-SEV input ablation",
            "+clean normal HN-SEV input ablation",
            "+hard negative HN-SEV input ablation",
            "+feature/prototype HN-SEV input ablation",
            "paper limitation if TP retention or ROI recall drops materially",
        ],
    }
    return all_rows, category_rows, case_rows, summary


def write_outputs(root: Path, out_dir: Path) -> dict[str, Any]:
    _, category_rows, case_rows, summary = build_audit(root, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_category_hn_sev_audit.csv", category_rows)
    write_csv(out_dir / "table_before_after_cases.csv", case_rows)
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
        default=Path("tables/hn_sev_retention_calibration"),
    )
    args = parser.parse_args()
    summary = write_outputs(args.root, args.out_dir)
    print(
        f"Wrote HN-SEV audit for {summary['categories']} categories to "
        f"{args.out_dir} (release_gate_passed={summary['release_gate_passed']})"
    )


if __name__ == "__main__":
    main()
