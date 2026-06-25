from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from tools.run_official_ddad import (
    center_crop_224,
    neighborhood_mean,
    official_image_scores,
    selected_categories,
)


class RunOfficialDDADTests(unittest.TestCase):
    def test_neighborhood_mean_matches_author_unfold_operator(self) -> None:
        features = torch.arange(
            2 * 3 * 5 * 7,
            dtype=torch.float32,
        ).reshape(2, 3, 5, 7)
        unfolded = F.unfold(features, kernel_size=3, padding=1)
        expected = unfolded.reshape(2, 3, 3, 3, 35).mean((2, 3))
        expected = expected.reshape(2, 3, 5, 7)

        torch.testing.assert_close(neighborhood_mean(features), expected)

    def test_center_crop_and_image_score(self) -> None:
        heatmaps = torch.zeros(2, 1, 256, 256)
        heatmaps[0, 0, 16, 16] = 2.0
        heatmaps[1, 0, 239, 239] = 3.0
        cropped = center_crop_224(heatmaps)

        self.assertEqual(tuple(cropped.shape), (2, 1, 224, 224))
        torch.testing.assert_close(
            official_image_scores(cropped),
            torch.tensor([2.0, 3.0]),
        )

    def test_category_selection_rejects_unknown_names(self) -> None:
        self.assertEqual(len(selected_categories("all")), 15)
        with self.assertRaises(ValueError):
            selected_categories("bottle,missing")


if __name__ == "__main__":
    unittest.main()
