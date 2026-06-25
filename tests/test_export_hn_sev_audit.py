from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.export_hn_sev_audit import (
    build_audit,
    calibration_metrics,
    resolve_mask,
    write_outputs,
)


def write_image(path: Path, color: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), (color, color, color)).save(path)


def write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (128, 128), 0)
    for x in range(40, 80):
        for y in range(40, 80):
            mask.putpixel((x, y), 255)
    mask.save(path)


def write_fake_mvtec_run(root: Path) -> None:
    image = root / "SEER-AD-dataset/MVTec-AD/bottle/test/broken_large/000.png"
    normal = root / "SEER-AD-dataset/MVTec-AD/bottle/test/good/000.png"
    mask = root / "SEER-AD-dataset/MVTec-AD/bottle/ground_truth/broken_large/000_mask.png"
    write_image(image)
    write_image(normal)
    write_mask(mask)

    run_dir = root / "runs/feature_fixedpixel_mvtec15_bottle_feature_pixel_policy"
    run_dir.mkdir(parents=True)
    rows = [
        {
            "image_index": 0,
            "source_path": "SEER-AD-dataset\\MVTec-AD\\bottle\\test\\broken_large\\000.png",
            "roi_id": 0,
            "bbox": [45, 45, 70, 70],
            "hn_sev_confidence": 0.9,
            "hn_sev_positive": 1,
        },
        {
            "image_index": 0,
            "source_path": "SEER-AD-dataset\\MVTec-AD\\bottle\\test\\broken_large\\000.png",
            "roi_id": 1,
            "bbox": [0, 0, 20, 20],
            "hn_sev_confidence": 0.1,
            "hn_sev_positive": 0,
        },
        {
            "image_index": 1,
            "source_path": "SEER-AD-dataset\\MVTec-AD\\bottle\\test\\good\\000.png",
            "roi_id": 0,
            "bbox": [0, 0, 20, 20],
            "hn_sev_confidence": 0.2,
            "hn_sev_positive": 0,
        },
    ]
    (run_dir / "roi_budget.json").write_text(json.dumps(rows), encoding="utf-8")


def test_resolve_mask_for_mvtec_anomaly_and_normal(tmp_path: Path) -> None:
    label, mask = resolve_mask(
        tmp_path,
        "SEER-AD-dataset\\MVTec-AD\\bottle\\test\\broken_large\\000.png",
    )
    assert label == "anomaly"
    assert mask == tmp_path / "SEER-AD-dataset/MVTec-AD/bottle/ground_truth/broken_large/000_mask.png"

    label, mask = resolve_mask(
        tmp_path,
        "SEER-AD-dataset\\MVTec-AD\\bottle\\test\\good\\000.png",
    )
    assert label == "normal"
    assert mask is None


def test_calibration_metrics_are_reasonable() -> None:
    metrics = calibration_metrics([1, 0], [0.9, 0.1], bins=2)
    assert metrics["brier"] < 0.02
    assert metrics["ece"] < 0.11


def test_build_audit_computes_retention_and_keeps_gate_false_without_ablation(
    tmp_path: Path,
) -> None:
    write_fake_mvtec_run(tmp_path)

    _, category_rows, case_rows, summary = build_audit(tmp_path, tmp_path / "out")

    assert len(category_rows) == 1
    row = category_rows[0]
    assert row["gt_roi_candidates"] == 1
    assert row["gt_roi_kept"] == 1
    assert row["tp_retention"] == 1.0
    assert row["background_or_normal_roi_suppressed"] == 2
    assert summary["overall"]["roi_recall_before_hn_sev"] == 1.0
    assert summary["overall"]["roi_recall_after_hn_sev"] == 1.0
    assert summary["release_gate_passed"] is False
    assert summary["input_ablation_ready"] is False


def test_write_outputs_creates_summary_tables_and_panels(tmp_path: Path) -> None:
    write_fake_mvtec_run(tmp_path)
    out_dir = tmp_path / "tables/hn_sev_retention_calibration"

    summary = write_outputs(tmp_path, out_dir)

    assert summary["categories"] == 1
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "table_category_hn_sev_audit.csv").is_file()
    assert (out_dir / "table_before_after_cases.csv").is_file()
    assert summary["before_after_visualizations"] >= 1
