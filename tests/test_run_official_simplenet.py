from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.run_official_simplenet import (
    denormalize_imagenet,
    margin_discriminator_loss,
    normalize_imagenet,
    patch_logits_to_outputs,
)


class RunOfficialSimpleNetTests(unittest.TestCase):
    def test_imagenet_normalization_roundtrip(self) -> None:
        images = torch.rand(2, 3, 8, 8)
        restored = denormalize_imagenet(normalize_imagenet(images))
        torch.testing.assert_close(restored, images)

    def test_margin_loss_is_zero_outside_margin(self) -> None:
        true_scores = torch.tensor([[0.6], [1.0]])
        fake_scores = torch.tensor([[-0.6], [-1.0]])
        loss = margin_discriminator_loss(
            true_scores,
            fake_scores,
            margin=0.5,
        )
        torch.testing.assert_close(loss, torch.zeros(()))

    def test_patch_outputs_have_expected_shape_and_scores(self) -> None:
        logits = torch.tensor(
            [[-1.0], [0.0], [0.5], [0.25], [-0.2], [-0.4], [0.3], [0.1]]
        )
        scores, heatmaps = patch_logits_to_outputs(
            logits,
            batch_size=2,
            patch_shape=(2, 2),
            output_size=8,
            sigma=0,
        )
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(heatmaps.shape, (2, 8, 8))
        np.testing.assert_allclose(scores, np.asarray([1.0, 0.4]))


if __name__ == "__main__":
    unittest.main()
