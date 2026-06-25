from __future__ import annotations

import unittest

import torch

from tools.run_official_draem import official_image_scores


class RunOfficialDRAEMTests(unittest.TestCase):
    def test_official_image_scores_use_padded_average_pool_max(self) -> None:
        heatmaps = torch.zeros(2, 1, 32, 32)
        heatmaps[0, 0, 8:29, 8:29] = 2.0
        heatmaps[1, 0, 2:23, 2:23] = 3.0

        scores = official_image_scores(heatmaps)

        torch.testing.assert_close(scores, torch.tensor([2.0, 3.0]))

    def test_official_image_scores_validate_shape(self) -> None:
        with self.assertRaises(ValueError):
            official_image_scores(torch.zeros(2, 32, 32))


if __name__ == "__main__":
    unittest.main()
