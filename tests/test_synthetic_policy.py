from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from seer_ad_v2.data.defect_synthesis import synthesize_anomaly
from seer_ad_v2.data.datasets import DTDTextureDataset
from seer_ad_v2.evaluation.pixel_policy import candidate_pixel_maps
from seer_ad_v2.evaluation.synthetic_validation import (
    evaluate_synthetic_validation,
    synthetic_normal_utility,
)
from tools.select_pixel_policy_with_normal_gate import gate_scores


class SyntheticPolicyTests(unittest.TestCase):
    def test_synthesis_is_reproducible_with_explicit_rng(self) -> None:
        image = torch.zeros(3, 32, 32)
        for mode in ("blob", "scratch", "spot", "patch"):
            first_image, first_mask = synthesize_anomaly(
                image,
                rng=np.random.RandomState(17),
                mask_mode=mode,
            )
            second_image, second_mask = synthesize_anomaly(
                image,
                rng=np.random.RandomState(17),
                mask_mode=mode,
            )
            self.assertTrue(torch.equal(first_mask, second_mask))
            self.assertTrue(torch.allclose(first_image, second_image))
            self.assertGreater(float(first_mask.sum()), 0.0)

    def test_dtd_sampling_is_deterministic_with_explicit_rng(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(
                np.full((8, 8, 3), 127, dtype=np.uint8)
            ).save(root / "texture.png")
            bank = DTDTextureDataset(root)
            first = bank.sample(
                (8, 8),
                rng=np.random.RandomState(11),
            )
            second = bank.sample(
                (8, 8),
                rng=np.random.RandomState(11),
            )
            self.assertTrue(np.array_equal(first, second))

    def test_candidate_policy_replays_postprocess_and_fixed_calibration(self) -> None:
        maps = np.zeros((1, 8, 8), dtype=np.float32)
        maps[:, 3:5, 3:5] = 2.0
        output = SimpleNamespace(
            heatmaps=maps,
            raw_heatmaps=maps,
            raw_distance_heatmaps=maps,
            raw_cosine_heatmaps=maps,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            post = candidate_pixel_maps(
                output,
                {"pixel_heatmap_source": "postprocess_gaussian:1"},
                run_dir,
            )
            self.assertEqual(post.shape, maps.shape)
            self.assertLess(float(post.max()), float(maps.max()))

            np.savez_compressed(
                run_dir / "normal_pixel_stats.npz",
                mean=np.zeros((8, 8), dtype=np.float32),
                std=np.ones((8, 8), dtype=np.float32),
                median=np.zeros((8, 8), dtype=np.float32),
                iqr=np.ones((8, 8), dtype=np.float32),
            )
            fixed = candidate_pixel_maps(
                output,
                {
                    "pixel_heatmap_source": "fixed_train_normal_feature_raw",
                    "pixel_policy_calibration_mode": "relu_robust_zscore",
                    "pixel_policy_postprocess_mode": "raw",
                    "pixel_policy_normal_source_map": "feature_raw",
                },
                run_dir,
            )
            self.assertTrue(np.allclose(fixed, maps))

    def test_synthetic_metrics_and_utility_penalize_false_positives(self) -> None:
        clean = np.zeros((2, 8, 8), dtype=np.float32)
        masks = np.zeros((2, 8, 8), dtype=np.uint8)
        masks[:, 2:6, 2:6] = 1
        synthetic = masks.astype(np.float32)
        metrics = evaluate_synthetic_validation(
            clean,
            synthetic,
            masks,
            np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            [synthetic.copy()],
        )
        self.assertGreater(metrics["pixel_ap"], 0.99)
        self.assertGreater(metrics["augmentation_stability"], 0.99)
        self.assertLess(metrics["normal_pixel_fpr"], 0.01)

        degraded = dict(metrics)
        degraded["normal_pixel_fpr"] = 1.0
        degraded["latency_ms"] = 100.0
        self.assertGreater(synthetic_normal_utility(metrics), synthetic_normal_utility(degraded))

    def test_gate_uses_synthetic_utility_not_candidate_name(self) -> None:
        stats = {
            "highres256": {"normal_score_p95": 1.0},
            "retrieval": {"normal_score_p95": 1.0},
        }
        synthetic = {
            "highres256": {
                "pixel_ap": 0.4,
                "aupro": 0.5,
                "dice": 0.4,
                "pixel_auroc": 0.8,
                "image_auroc": 0.8,
                "augmentation_stability": 0.7,
                "normal_pixel_fpr": 0.1,
                "latency_ms": 10.0,
            },
            "retrieval": {
                "pixel_ap": 0.8,
                "aupro": 0.8,
                "dice": 0.7,
                "pixel_auroc": 0.9,
                "image_auroc": 0.9,
                "augmentation_stability": 0.9,
                "normal_pixel_fpr": 0.02,
                "latency_ms": 20.0,
            },
        }
        scores = gate_scores(
            stats,
            "synthetic_normal_utility",
            1.5,
            1.4,
            synthetic_by_name=synthetic,
            utility_kwargs={},
        )
        self.assertGreater(scores["retrieval"], scores["highres256"])


if __name__ == "__main__":
    unittest.main()
