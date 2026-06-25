from __future__ import annotations

import unittest

import numpy as np

from seer_ad_v2.evaluation.repair_quality import (
    image_repair_quality,
    roi_ground_truth_records,
    safe_pearson,
    safe_spearman,
    sdr_gt_summary,
)


class RepairQualityTests(unittest.TestCase):
    def test_correlations_handle_ties(self) -> None:
        self.assertAlmostEqual(safe_pearson([0, 1, 2], [0, 2, 4]), 1.0)
        self.assertAlmostEqual(safe_spearman([0, 1, 1, 2], [0, 1, 1, 2]), 1.0)
        self.assertTrue(np.isnan(safe_pearson([1, 1], [0, 1])))

    def test_roi_ground_truth_summary(self) -> None:
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        masks[1, 2:6, 2:6] = 1
        rows = [
            {"image_index": 0, "bbox": [0, 0, 4, 4], "sdr": 0.0},
            {"image_index": 1, "bbox": [2, 2, 6, 6], "sdr": 1.0},
            {"image_index": 1, "bbox": [0, 0, 2, 2], "sdr": 0.0},
        ]
        records = roi_ground_truth_records(masks, np.asarray([0, 1]), rows)
        summary = sdr_gt_summary(records)
        self.assertEqual(summary["roi_count"], 3)
        self.assertEqual(summary["anomaly_roi_count"], 2)
        self.assertEqual(summary["positive_sdr_gt_hit_rate"], 1.0)
        self.assertAlmostEqual(summary["sdr_gt_fraction_spearman"], 1.0)

    def test_image_quality_separates_background_and_foreground(self) -> None:
        original = np.zeros((8, 8, 3), dtype=np.float32)
        repaired = original.copy()
        repaired[2:6, 2:6] = 0.5
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        quality = image_repair_quality(original, repaired, mask)
        self.assertEqual(quality["background_mae"], 0.0)
        self.assertGreater(quality["foreground_mae"], 0.0)
        self.assertTrue(np.isinf(quality["background_psnr"]))


if __name__ == "__main__":
    unittest.main()
