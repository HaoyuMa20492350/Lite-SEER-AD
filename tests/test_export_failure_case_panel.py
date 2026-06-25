from __future__ import annotations

import unittest

import numpy as np

from tools.export_failure_case_panel import (
    fixed_dice,
    select_failure_index,
)


class ExportFailureCasePanelTests(unittest.TestCase):
    def test_fixed_dice_uses_frozen_threshold(self) -> None:
        mask = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
        heatmap = np.asarray([[0.8, 0.2], [0.7, 0.1]], dtype=np.float32)
        self.assertAlmostEqual(fixed_dice(mask, heatmap, 0.5), 0.5)

    def test_select_failure_index_ignores_normal_rows(self) -> None:
        rows = [
            {
                "index": 0,
                "label": 0,
                "image_score": 0.01,
                "pixel_ap": float("nan"),
                "fixed_dice": 0.0,
            },
            {
                "index": 1,
                "label": 1,
                "image_score": 0.2,
                "pixel_ap": 0.6,
                "fixed_dice": 0.5,
            },
            {
                "index": 2,
                "label": 1,
                "image_score": 0.3,
                "pixel_ap": 0.2,
                "fixed_dice": 0.1,
            },
        ]
        selected = select_failure_index(
            rows,
            "lowest_anomaly_pixel_ap",
        )
        self.assertEqual(selected["index"], 2)


if __name__ == "__main__":
    unittest.main()
