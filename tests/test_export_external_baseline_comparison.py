from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.export_external_baseline_comparison import (
    build_comparison,
    exact_sign_p,
    failure_rows,
    holm_adjust,
    load_external_method,
    paired_inference,
)


class ExportExternalBaselineComparisonTests(unittest.TestCase):
    def test_build_comparison_supports_multiple_methods(self) -> None:
        lite = {
            "bottle": {
                metric: "0.8"
                for metric in (
                    "image_auroc",
                    "pixel_auroc",
                    "aupro",
                    "pixel_ap",
                    "dice",
                    "oracle_dice",
                )
            }
        }
        external = {
            "patchcore": {
                "bottle": {
                    "display_method": "PatchCore-Official",
                    "source_kind": "author_official",
                    "official_implementation": True,
                    "source_commit": "a" * 40,
                    **{metric: 0.7 for metric in lite["bottle"]},
                }
            },
            "padim": {
                "bottle": {
                    "display_method": "PaDiM-Anomalib",
                    "source_kind": "maintained_reference",
                    "official_implementation": False,
                    "source_commit": "b" * 40,
                    **{metric: 0.9 for metric in lite["bottle"]},
                }
            },
        }
        rows, summary = build_comparison(lite, external)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            summary["comparisons"]["patchcore"]["metrics"]["aupro"][
                "lite_category_wins"
            ],
            1,
        )
        self.assertEqual(
            summary["comparisons"]["padim"]["metrics"]["aupro"][
                "external_category_wins"
            ],
            1,
        )
        self.assertFalse(
            summary["comparisons"]["padim"]["official_implementation"]
        )

    def test_load_external_method_enriches_metrics_from_provenance(self) -> None:
        with TemporaryDirectory() as temp_dir:
            category_dir = Path(temp_dir) / "patchcore" / "bottle"
            category_dir.mkdir(parents=True)
            (category_dir / "metrics.json").write_text(
                json.dumps({"category": "bottle", "image_auroc": 0.9}),
                encoding="utf-8",
            )
            (category_dir / "provenance.json").write_text(
                json.dumps(
                    {
                        "source_kind": "author_official",
                        "official_implementation": True,
                    }
                ),
                encoding="utf-8",
            )

            rows = load_external_method(Path(temp_dir), "patchcore")

        self.assertEqual(rows["bottle"]["source_kind"], "author_official")
        self.assertTrue(rows["bottle"]["official_implementation"])

    def test_paired_inference_is_deterministic(self) -> None:
        rows = [
            {
                "method": "example",
                "display_method": "Example",
                "category": category,
                **{
                    f"delta_{metric}": delta
                    for metric in (
                        "image_auroc",
                        "pixel_auroc",
                        "aupro",
                        "pixel_ap",
                        "dice",
                        "oracle_dice",
                    )
                },
            }
            for category, delta in (
                ("a", 0.1),
                ("b", 0.2),
                ("c", -0.1),
            )
        ]
        first = paired_inference(rows, samples=1000, seed=7)
        second = paired_inference(rows, samples=1000, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["wins"], 2)
        self.assertEqual(first[0]["losses"], 1)

    def test_exact_sign_and_holm_adjustment(self) -> None:
        self.assertEqual(exact_sign_p(0, 0), None)
        self.assertAlmostEqual(exact_sign_p(3, 0), 0.25)
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_failure_rows_rank_most_negative_delta_first(self) -> None:
        rows = []
        for category, delta in (("a", -0.1), ("b", -0.4), ("c", 0.2)):
            row = {
                "method": "example",
                "display_method": "Example",
                "category": category,
            }
            for metric in (
                "image_auroc",
                "pixel_auroc",
                "aupro",
                "pixel_ap",
                "dice",
                "oracle_dice",
            ):
                row[f"delta_{metric}"] = delta
                row[f"lite_{metric}"] = 0.5 + delta
                row[f"external_{metric}"] = 0.5
            rows.append(row)
        failures = failure_rows(rows, worst_categories=1)
        self.assertEqual(failures[0]["category"], "b")
        self.assertTrue(failures[0]["lite_loses"])


if __name__ == "__main__":
    unittest.main()
