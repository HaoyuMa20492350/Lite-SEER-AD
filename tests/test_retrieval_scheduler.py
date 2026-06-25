from __future__ import annotations

import unittest

import numpy as np
import torch

from seer_ad_v2.data.hard_negative_mining import ROI
from seer_ad_v2.models.diffusion.local_refiner import local_repair
from seer_ad_v2.models.feature_prior import retrieve_normal_reference
from seer_ad_v2.models.scheduler.lc_rds import (
    ExpectedUtilityScheduler,
    choose_action_with_model,
    production_budget_guard_latency_estimates,
    roi_features,
)


class _FixedExtractor(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = (len(x), 1, 4, 4)
        one = torch.ones(shape, device=x.device)
        zero = torch.zeros(shape, device=x.device)
        return {"layer1": torch.cat([one, zero, zero], dim=1)}


class _IdentityDiffusion:
    def reconstruct(self, model: torch.nn.Module, x0: torch.Tensor, steps: int = 5) -> torch.Tensor:
        return x0


class _LegacyFourActionScheduler(torch.nn.Module):
    action_names = ["skip", "repair-10", "repair-25", "native-refine"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=x.device)


class RetrievalSchedulerTests(unittest.TestCase):
    def test_retrieval_returns_matching_normal_patch(self) -> None:
        state = {
            "retrieval_features_norm": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "retrieval_patches": torch.stack(
                [
                    torch.full((3, 8, 8), 255, dtype=torch.uint8),
                    torch.zeros((3, 8, 8), dtype=torch.uint8),
                ]
            ),
        }
        image = torch.zeros(1, 3, 16, 16)
        roi = ROI(0, 0, 16, 16, 256, 1.0)
        patch, similarity, index = retrieve_normal_reference(
            {"prior_state": state},
            _FixedExtractor(),
            ["layer1"],
            image,
            roi,
            "cpu",
            output_size=(16, 16),
        )
        self.assertEqual(index, 0)
        self.assertAlmostEqual(similarity, 1.0, places=6)
        self.assertIsNotNone(patch)
        self.assertEqual(tuple(patch.shape), (1, 3, 16, 16))
        self.assertTrue(torch.allclose(patch, torch.ones_like(patch)))

    def test_local_repair_accepts_retrieved_initialization(self) -> None:
        image = torch.zeros(1, 3, 16, 16)
        reference = torch.ones(1, 3, 16, 16)
        roi = ROI(0, 0, 16, 16, 256, 1.0)
        repaired = local_repair(
            image,
            torch.nn.Identity(),
            _IdentityDiffusion(),
            roi,
            steps=1,
            reference_patch=reference,
            reference_weight=1.0,
        )
        self.assertTrue(torch.allclose(repaired, reference))

    def test_expected_utility_scheduler_enforces_accumulated_budget(self) -> None:
        scheduler = ExpectedUtilityScheduler(latency_budget_ms=60.0)
        roi = ROI(0, 0, 16, 16, 256, 0.9)
        features = roi_features(roi, 0.8, 0.7, (64, 64), 1)
        first = scheduler.choose(features, spent_ms=0.0, expected_gain=0.8)
        self.assertNotEqual(first.name, "skip")
        scheduler.observe(first, latency_ms=55.0, realized_gain=0.4, predicted_gain=0.8)
        second = scheduler.choose(features, spent_ms=55.0, expected_gain=0.8)
        self.assertEqual(second.name, "skip")
        self.assertGreaterEqual(scheduler.expected_latency[first.name], 55.0)

    def test_expected_utility_scheduler_can_use_repair5_under_tight_budget(self) -> None:
        scheduler = ExpectedUtilityScheduler(latency_budget_ms=25.0)
        roi = ROI(0, 0, 12, 12, 144, 0.9)
        features = roi_features(roi, 0.9, 0.8, (64, 64), 1)
        action = scheduler.choose(features, spent_ms=0.0, expected_gain=0.9)

        self.assertEqual(action.name, "repair-5")

    def test_production_budget_guard_uses_conservative_latency_estimates(self) -> None:
        roi = ROI(0, 0, 12, 12, 144, 0.9)
        features = roi_features(roi, 0.9, 0.8, (64, 64), 1)

        tight = ExpectedUtilityScheduler(
            latency_budget_ms=75.0,
            latency_estimates=production_budget_guard_latency_estimates(),
        )
        moderate = ExpectedUtilityScheduler(
            latency_budget_ms=100.0,
            latency_estimates=production_budget_guard_latency_estimates(),
        )
        high = ExpectedUtilityScheduler(
            latency_budget_ms=150.0,
            latency_estimates=production_budget_guard_latency_estimates(),
        )

        self.assertEqual(tight.choose(features, spent_ms=0.0, expected_gain=0.9).name, "skip")
        self.assertEqual(moderate.choose(features, spent_ms=0.0, expected_gain=0.9).name, "repair-5")
        self.assertIn(high.choose(features, spent_ms=0.0, expected_gain=0.9).name, {"repair-25", "native-refine"})

    def test_legacy_four_action_model_keeps_saved_action_mapping(self) -> None:
        features = np.zeros(8, dtype=np.float32)
        action = choose_action_with_model(_LegacyFourActionScheduler(), features)

        self.assertEqual(action.name, "repair-10")


if __name__ == "__main__":
    unittest.main()
