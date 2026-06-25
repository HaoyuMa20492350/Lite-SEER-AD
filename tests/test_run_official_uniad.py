from __future__ import annotations

import unittest

import torch

from tools.run_official_uniad import official_image_scores


class RunOfficialUniADTests(unittest.TestCase):
    def test_official_image_scores_use_max_of_avg_pool(self) -> None:
        heatmaps = torch.zeros(2, 1, 20, 20)
        heatmaps[0, 0, :16, :16] = 2.0
        heatmaps[1, 0, 4:20, 4:20] = 3.0

        scores = official_image_scores(heatmaps)

        torch.testing.assert_close(scores, torch.tensor([2.0, 3.0]))

    def test_official_image_scores_validate_shape(self) -> None:
        with self.assertRaises(ValueError):
            official_image_scores(torch.zeros(2, 20, 20))


if __name__ == "__main__":
    unittest.main()
