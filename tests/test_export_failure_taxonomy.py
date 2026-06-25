from __future__ import annotations

from tools.export_failure_taxonomy import build_summary, build_taxonomy_rows


def test_taxonomy_computes_best_baseline_delta() -> None:
    rows = [
        {
            "dataset": "mvtec15",
            "source": "ours_selected",
            "method": "lite",
            "category": "grid",
            "selected_candidate": "fixed",
            "image_auroc": "0.62",
            "pixel_auroc": "0.96",
            "aupro": "0.91",
            "pixel_ap": "0.26",
            "dice": "0.35",
        },
        {
            "dataset": "mvtec15",
            "source": "baseline",
            "method": "patchcore",
            "category": "grid",
            "image_auroc": "0.71",
            "pixel_auroc": "0.90",
            "aupro": "0.88",
            "pixel_ap": "0.13",
            "dice": "0.21",
        },
        {
            "dataset": "mvtec15",
            "source": "baseline",
            "method": "padim",
            "category": "grid",
            "image_auroc": "0.73",
            "pixel_auroc": "0.85",
            "aupro": "0.57",
            "pixel_ap": "0.10",
            "dice": "0.23",
        },
    ]

    taxonomy = build_taxonomy_rows(
        rows,
        specs=[
            {
                "dataset": "mvtec15",
                "category": "grid",
                "family": "periodic texture",
                "failure_mode": "image-score miss",
                "recommended_action": "frequency candidate",
            }
        ],
    )

    assert taxonomy[0]["taxonomy_status"] == "covered"
    assert taxonomy[0]["best_baseline_dice_method"] == "padim"
    assert round(taxonomy[0]["dice_delta_vs_best"], 4) == 0.12
    assert taxonomy[0]["severity"] == "critical"


def test_summary_fails_when_required_category_is_missing() -> None:
    taxonomy = build_taxonomy_rows(
        [],
        specs=[
            {
                "dataset": "mpdd",
                "category": "bracket_white",
                "family": "low contrast",
                "failure_mode": "missing",
                "recommended_action": "local contrast",
            }
        ],
    )
    summary = build_summary(taxonomy)

    assert summary["release_gate_passed"] is False
    assert summary["covered_weak_categories"] == 0
