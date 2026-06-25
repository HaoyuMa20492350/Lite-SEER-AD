from __future__ import annotations

import unittest

from tools.export_strict_threshold_paper_artifacts import (
    grouped_means,
    mean_metrics,
    validate_rows,
)


def row(dataset: str, seed: str, category: str, dice: float) -> dict[str, str]:
    values = {
        "dataset": dataset,
        "split_seed": seed,
        "category": category,
        "threshold_protocol": "synthetic_normal_fixed_threshold_v1",
        "normal_pixel_fpr": "0.004",
        "uses_real_anomaly_labels_for_threshold": "False",
        "uses_real_anomaly_masks_for_threshold": "False",
    }
    for metric in (
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
    ):
        values[metric] = str(dice + 0.1 if metric.startswith("oracle_") else dice)
    return values


class StrictThresholdPaperArtifactTests(unittest.TestCase):
    def test_validation_and_grouped_means(self) -> None:
        rows = [
            row("mvtec15", "seed7", "bottle", 0.4),
            row("mvtec15", "seed13", "bottle", 0.6),
        ]
        audit = validate_rows(rows)
        self.assertEqual(audit["rows"], 2)
        means = grouped_means(rows, ("dataset",))
        self.assertAlmostEqual(means[0]["dice"], 0.5)
        self.assertAlmostEqual(means[0]["oracle_dice_gap"], 0.1)

    def test_real_anomaly_supervision_is_rejected(self) -> None:
        bad = row("mvtec15", "seed7", "bottle", 0.4)
        bad["uses_real_anomaly_masks_for_threshold"] = "True"
        with self.assertRaises(ValueError):
            validate_rows([bad])

    def test_mean_metrics_counts_runs(self) -> None:
        result = mean_metrics(
            [
                row("visa", "seed7", "pcb1", 0.2),
                row("visa", "seed13", "pcb1", 0.4),
            ]
        )
        self.assertEqual(result["runs"], 2)
        self.assertAlmostEqual(result["dice"], 0.3)


if __name__ == "__main__":
    unittest.main()
