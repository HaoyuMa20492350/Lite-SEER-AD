from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.export_hn_sev_input_ablation import (
    build_summary,
    collect_rows,
    coverage_rows,
    metric_summary,
    write_outputs,
)


def write_ablation_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "category", "ablation", "fprr", "pixel_ap", "aupro", "dice"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_run_args(path: Path, *, config: str, category: str, ablation: str | None = None) -> None:
    args = {"config": config, "category": category}
    if ablation is not None:
        args["ablation"] = ablation
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"command": "infer", "args": args}), encoding="utf-8")


def test_coverage_separates_exact_requirements_from_proxy_variants(tmp_path: Path) -> None:
    write_ablation_table(
        tmp_path / "tables/example/table_ablation_hn_sev.csv",
        [
            {
                "dataset": "mvtec15",
                "category": "bottle",
                "ablation": "synthetic_only_sev",
                "fprr": 0.8,
                "pixel_ap": 0.5,
                "aupro": 0.6,
                "dice": 0.4,
            },
            {
                "dataset": "mvtec15",
                "category": "bottle",
                "ablation": "clean_normal_sev",
                "fprr": 0.7,
                "pixel_ap": 0.55,
                "aupro": 0.65,
                "dice": 0.45,
            },
            {
                "dataset": "mvtec15",
                "category": "bottle",
                "ablation": "hard_negative_sev",
                "fprr": 0.9,
                "pixel_ap": 0.6,
                "aupro": 0.7,
                "dice": 0.5,
            },
            {
                "dataset": "mvtec15",
                "category": "bottle",
                "ablation": "no_prototype",
                "fprr": 0.4,
                "pixel_ap": 0.45,
                "aupro": 0.5,
                "dice": 0.35,
            },
            {
                "dataset": "visa",
                "category": "pcb1",
                "ablation": "feature_hn_sev",
                "fprr": 0.6,
                "pixel_ap": 0.4,
                "aupro": 0.5,
                "dice": 0.3,
            },
        ],
    )

    rows = collect_rows(tmp_path)
    coverage = coverage_rows(rows)
    bottle = next(row for row in coverage if row["category"] == "bottle")
    pcb1 = next(row for row in coverage if row["category"] == "pcb1")

    assert bottle["exact_all_required"] is True
    assert pcb1["exact_all_required"] is False
    assert pcb1["feature_full_proxy"] is True
    assert pcb1["exact_synthetic_only"] is False
    assert bottle["metric_all_required"] is True


def test_metric_summary_and_release_gate_require_all_33_exact_categories(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(33):
        category = f"cat{index:02d}"
        for ablation in [
            "synthetic_only_sev",
            "clean_normal_sev",
            "hard_negative_sev",
            "no_prototype",
        ]:
            rows.append(
                {
                    "dataset": "mvtec15",
                    "category": category,
                    "ablation": ablation,
                    "fprr": 0.5,
                    "pixel_ap": 0.6,
                    "aupro": 0.7,
                    "dice": 0.8,
                }
            )
    coverage = coverage_rows(rows)
    summary = build_summary(rows, coverage)
    metrics = metric_summary(rows)

    assert summary["release_gate_passed"] is True
    assert summary["complete_exact_categories"] == 33
    assert summary["complete_metric_categories"] == 33
    assert next(row for row in metrics if row["ablation"] == "synthetic_only_sev")[
        "categories"
    ] == 33

    coverage[0]["exact_all_required"] = False
    incomplete = build_summary(rows, coverage)
    assert incomplete["release_gate_passed"] is False


def test_collect_rows_reads_run_metrics_and_training_metadata_without_overclaiming(
    tmp_path: Path,
) -> None:
    metric_run = tmp_path / "runs/mvtec_bottle_synthetic_only_sev"
    write_run_args(
        metric_run / "run_args.json",
        config="configs/mvtec.yaml",
        category="bottle",
        ablation="synthetic_only_sev",
    )
    (metric_run / "metrics.json").write_text(
        json.dumps(
            {
                "fprr": 0.8,
                "pixel_ap": 0.5,
                "aupro": 0.6,
                "dice": 0.4,
            }
        ),
        encoding="utf-8",
    )

    training_run = tmp_path / "runs/mvtec_bottle_no_prototype_train"
    write_run_args(
        training_run / "run_args.json",
        config="configs/mvtec.yaml",
        category="bottle",
    )
    (training_run / "hn_sev_metrics.json").write_text(
        json.dumps({"hn_sev_input_ablation": "no_prototype"}),
        encoding="utf-8",
    )

    rows = collect_rows(tmp_path)
    coverage = coverage_rows(rows)
    bottle = next(row for row in coverage if row["category"] == "bottle")
    summary = build_summary(rows, coverage)

    assert {row["source_kind"] for row in rows} == {"run_metrics", "training_metadata"}
    assert bottle["exact_synthetic_only"] is True
    assert bottle["metric_synthetic_only"] is True
    assert bottle["exact_feature_prototype"] is True
    assert bottle["metric_feature_prototype"] is False
    assert summary["complete_exact_categories"] == 0
    assert summary["complete_metric_categories"] == 0
    assert summary["release_gate_passed"] is False


def test_write_outputs_creates_tables_and_keeps_realistic_partial_gate_false(
    tmp_path: Path,
) -> None:
    write_ablation_table(
        tmp_path / "tables/run/table_ablation_hn_sev.csv",
        [
            {
                "dataset": "mvtec15",
                "category": "bottle",
                "ablation": "feature_hn_sev",
                "fprr": 0.8,
                "pixel_ap": 0.5,
                "aupro": 0.6,
                "dice": 0.4,
            }
        ],
    )
    out_dir = tmp_path / "tables/hn_sev_input_ablation"

    summary = write_outputs(tmp_path, out_dir)

    assert summary["release_gate_passed"] is False
    assert summary["proxy_variant_counts"]["feature_full_proxy"] == 1
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "table_input_ablation_coverage.csv").is_file()
    assert (out_dir / "table_input_ablation_metric_summary.csv").is_file()
    persisted = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert persisted["schema"] == "lite-seer-ad-hn-sev-input-ablation-v1"
    assert persisted["evidence_level"] == "mixed_table_run_metadata_coverage_v2"
