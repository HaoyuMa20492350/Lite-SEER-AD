from __future__ import annotations

import unittest

import numpy as np

from tools.select_image_score_aggregation import (
    choose_mode,
    fuse_evidence_payloads,
    synthetic_mode_metrics,
)


class SelectImageScoreAggregationTests(unittest.TestCase):
    def test_synthetic_mode_metrics_prefers_separated_maps(self) -> None:
        clean = np.zeros((4, 8, 8), dtype=np.float32)
        synthetic = np.zeros((4, 8, 8), dtype=np.float32)
        synthetic[:, :2, :2] = 1.0
        metrics = synthetic_mode_metrics(clean, synthetic, "top5")
        self.assertEqual(metrics["synthetic_image_auroc"], 1.0)
        self.assertGreater(metrics["robust_margin"], 0)

    def test_choose_mode_uses_cross_seed_mean_then_minimum(self) -> None:
        rows = [
            {
                "mode": "max",
                "synthetic_image_auroc": value,
                "robust_margin": 1.0,
            }
            for value in (0.9, 0.7)
        ]
        rows.extend(
            {
                "mode": "top5",
                "synthetic_image_auroc": value,
                "robust_margin": 1.0,
            }
            for value in (0.81, 0.81)
        )
        selected = choose_mode(rows, ["max", "top5"])
        self.assertEqual(selected["mode"], "top5")

    def test_fuse_evidence_payloads_aligns_metadata_and_sizes(self) -> None:
        metadata = {
            "paths": np.asarray(["a", "b"]),
            "variant_ids": np.asarray([0, 1], dtype=np.int32),
            "mask_modes": np.asarray(["blob", "scratch"]),
        }
        source_a = {
            **metadata,
            "clean_heatmaps": np.zeros((2, 4, 4), dtype=np.float32),
            "synthetic_heatmaps": np.ones((2, 4, 4), dtype=np.float32),
            "flipped_synthetic_heatmaps": np.ones(
                (2, 4, 4), dtype=np.float32
            ),
            "photometric_synthetic_heatmaps": np.ones(
                (2, 4, 4), dtype=np.float32
            ),
            "synthetic_masks": np.ones((2, 4, 4), dtype=np.uint8),
        }
        source_b = {
            **metadata,
            "clean_heatmaps": np.zeros((2, 8, 8), dtype=np.float32),
            "synthetic_heatmaps": np.full(
                (2, 8, 8), 2.0, dtype=np.float32
            ),
            "flipped_synthetic_heatmaps": np.full(
                (2, 8, 8), 2.0, dtype=np.float32
            ),
            "photometric_synthetic_heatmaps": np.full(
                (2, 8, 8), 2.0, dtype=np.float32
            ),
            "synthetic_masks": np.ones((2, 8, 8), dtype=np.uint8),
        }
        fused = fuse_evidence_payloads(
            source_a,
            source_b,
            weight_a=0.5,
            scale_a=(0.0, 1.0),
            scale_b=(0.0, 1.0),
            image_score_mode="top5",
            seed=7,
        )
        self.assertEqual(fused["clean_heatmaps"].shape, (2, 8, 8))
        self.assertTrue(
            np.array_equal(
                fused["clean_score_heatmaps"],
                fused["clean_heatmaps"],
            )
        )
        self.assertTrue(
            np.allclose(fused["synthetic_heatmaps"], 1.5)
        )


if __name__ == "__main__":
    unittest.main()
