from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from seer_ad_v2.evaluation.metrics_detection import (
    aupro_score,
    detection_metrics,
)
from seer_ad_v2.evaluation.pixel_threshold_policy import (
    load_pixel_threshold_policy,
    save_pixel_threshold_policy,
    select_synthetic_normal_threshold,
)


class PixelThresholdPolicyTests(unittest.TestCase):
    def test_selects_high_dice_threshold_under_normal_fpr_cap(self) -> None:
        clean = np.zeros((2, 8, 8), dtype=np.float32)
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        masks[:, 2:6, 2:6] = 1
        synthetic = masks.astype(np.float32)
        policy = select_synthetic_normal_threshold(
            clean,
            synthetic,
            masks,
            max_normal_fpr=0.005,
        )
        self.assertLessEqual(policy["observed_normal_pixel_fpr"], 0.005)
        self.assertGreater(policy["synthetic_dice"], 0.99)
        self.assertFalse(policy["uses_real_anomaly_labels"])
        self.assertFalse(policy["uses_real_anomaly_masks"])

    def test_uses_adjusted_normal_quantile_fallback_for_tied_scores(self) -> None:
        clean = np.ones((1, 4, 4), dtype=np.float32)
        synthetic = np.ones((1, 4, 4), dtype=np.float32)
        masks = np.ones((1, 4, 4), dtype=np.uint8)
        policy = select_synthetic_normal_threshold(
            clean,
            synthetic,
            masks,
            max_normal_fpr=0.005,
        )
        self.assertTrue(policy["fallback_used"])
        self.assertGreater(policy["threshold"], 1.0)
        self.assertEqual(policy["observed_normal_pixel_fpr"], 0.0)

    def test_fixed_metrics_preserve_oracle_diagnostics(self) -> None:
        labels = np.asarray([0, 1], dtype=np.uint8)
        masks = np.asarray([[[0, 0]], [[1, 0]]], dtype=np.uint8)
        heatmaps = np.asarray([[[0.1, 0.2]], [[0.9, 0.8]]], dtype=np.float32)
        metrics = detection_metrics(
            labels,
            np.asarray([0.1, 0.9], dtype=np.float32),
            masks,
            heatmaps,
            pixel_threshold=0.95,
            threshold_protocol="unit_test_fixed",
        )
        self.assertEqual(metrics["threshold_protocol"], "unit_test_fixed")
        self.assertEqual(metrics["dice"], 0.0)
        self.assertGreater(metrics["oracle_dice"], 0.99)

    def test_policy_round_trip(self) -> None:
        policy = {
            "protocol": "unit_test",
            "threshold": 0.25,
            "uses_real_anomaly_labels": False,
            "uses_real_anomaly_masks": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pixel_threshold_policy.json"
            save_pixel_threshold_policy(policy, path)
            self.assertEqual(load_pixel_threshold_policy(path), policy)

    def test_component_aupro_is_high_for_perfect_localization(self) -> None:
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        masks[:, 2:6, 2:6] = 1
        perfect = masks.astype(np.float32)
        inverted = 1.0 - perfect
        self.assertGreater(aupro_score(masks, perfect), 0.99)
        self.assertLess(aupro_score(masks, inverted), 0.2)


if __name__ == "__main__":
    unittest.main()
