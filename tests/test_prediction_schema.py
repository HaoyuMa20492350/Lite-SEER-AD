from __future__ import annotations

import unittest

import numpy as np

from seer_ad_v2.evaluation.prediction_schema import (
    CANONICAL_HEATMAP_KEYS,
    prediction_heatmap_payload,
    resolve_prediction_heatmaps,
)


class PredictionSchemaTests(unittest.TestCase):
    def test_payload_writes_canonical_and_legacy_keys(self) -> None:
        detection = np.zeros((2, 4, 4), dtype=np.float32)
        verification = np.ones_like(detection)
        score = np.full_like(detection, 2.0)
        payload = prediction_heatmap_payload(detection, verification, score)
        self.assertTrue(set(CANONICAL_HEATMAP_KEYS).issubset(payload))
        self.assertTrue(np.array_equal(payload["heatmaps"], detection))
        self.assertTrue(
            np.array_equal(payload["verification_heatmaps"], verification)
        )
        self.assertTrue(np.array_equal(payload["image_score_heatmaps"], score))

    def test_resolver_upgrades_legacy_schema(self) -> None:
        detection = np.zeros((1, 2, 2), dtype=np.float32)
        verification = np.ones_like(detection)
        score = np.full_like(detection, 3.0)
        resolved = resolve_prediction_heatmaps(
            {
                "heatmaps": detection,
                "final_heatmaps": verification,
                "score_heatmaps": score,
            }
        )
        self.assertTrue(np.array_equal(resolved[0], detection))
        self.assertTrue(np.array_equal(resolved[1], verification))
        self.assertTrue(np.array_equal(resolved[2], score))

    def test_payload_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            prediction_heatmap_payload(
                np.zeros((1, 2, 2), dtype=np.float32),
                np.zeros((1, 3, 3), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
