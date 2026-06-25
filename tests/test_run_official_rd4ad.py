from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from tools.run_official_rd4ad import (
    backfill_cached_provenance,
    official_image_scores,
    raw_anomaly_maps,
    rd4ad_loss,
)


class RunOfficialRD4ADTests(unittest.TestCase):
    def test_loss_is_zero_for_identical_features(self) -> None:
        features = [
            torch.randn(2, 4, 3, 3),
            torch.randn(2, 8, 2, 2),
        ]
        loss = rd4ad_loss(features, [item.clone() for item in features])
        torch.testing.assert_close(loss, torch.zeros(()), atol=1e-6, rtol=0)

    def test_raw_maps_and_scores_have_expected_shape(self) -> None:
        teacher = [torch.ones(2, 3, 4, 4)]
        student = [torch.ones(2, 3, 4, 4)]
        student[0][1, 0] = -1

        maps = raw_anomaly_maps(
            teacher,
            student,
            out_size=8,
            sigma=0,
        )
        scores = official_image_scores(maps)

        self.assertEqual(maps.shape, (2, 8, 8))
        self.assertEqual(scores.shape, (2,))
        self.assertAlmostEqual(float(scores[0]), 0.0, places=6)
        self.assertGreater(float(scores[1]), 0.0)

    def test_image_scores_validate_shape(self) -> None:
        with self.assertRaises(ValueError):
            official_image_scores(np.zeros((8, 8), dtype=np.float32))

    def test_cached_provenance_is_backfilled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.json"
            path.write_text(
                json.dumps({"training_configuration": {"epochs": 200}}),
                encoding="utf-8",
            )

            self.assertTrue(backfill_cached_provenance(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(payload["checkpoint_source"])
            self.assertTrue(payload["paper_eligible_full_training"])
            self.assertFalse(backfill_cached_provenance(path))


if __name__ == "__main__":
    unittest.main()
