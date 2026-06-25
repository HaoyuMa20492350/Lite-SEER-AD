from __future__ import annotations

import math

from seer_ad_v2.evaluation.module_evidence import (
    comparison_rows,
    summarize_comparisons,
)


def test_comparison_rows_and_summary() -> None:
    metrics = [
        {
            "category": "a",
            "ablation": "target",
            "pixel_ap": "0.6",
            "fprr": "0.2",
        },
        {
            "category": "a",
            "ablation": "baseline",
            "pixel_ap": "0.5",
            "fprr": "0.8",
        },
        {
            "category": "b",
            "ablation": "target",
            "pixel_ap": "0.4",
            "fprr": "0.3",
        },
        {
            "category": "b",
            "ablation": "baseline",
            "pixel_ap": "0.5",
            "fprr": "0.9",
        },
    ]
    efficiency = [
        {"category": "a", "ablation": "target", "latency_ms_mean": "10"},
        {"category": "a", "ablation": "baseline", "latency_ms_mean": "20"},
        {"category": "b", "ablation": "target", "latency_ms_mean": "15"},
        {"category": "b", "ablation": "baseline", "latency_ms_mean": "30"},
    ]
    rows = comparison_rows(
        "demo",
        metrics,
        efficiency,
        module="M",
        comparison="target_vs_baseline",
        target="target",
        baseline="baseline",
    )
    assert len(rows) == 2
    assert math.isclose(rows[0]["delta_fprr"], -0.6, abs_tol=1e-12)

    summary = summarize_comparisons(rows)
    demo = next(row for row in summary if row["dataset"] == "demo")
    overall = next(row for row in summary if row["dataset"] == "all")
    assert demo["categories"] == 2
    assert demo["negative_delta_fprr"] == 2
    assert demo["negative_delta_latency_ms_mean"] == 2
    assert math.isclose(demo["mean_delta_pixel_ap"], 0.0, abs_tol=1e-12)
    assert overall["categories"] == 2
