from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.run_official_diffusionad import (
    BinaryFocalLoss,
    completed_output_matches_request,
    official_image_scores,
)


class RunOfficialDiffusionADTests(unittest.TestCase):
    @staticmethod
    def _request_kwargs(**overrides):
        values = {
            "epochs": 10,
            "max_train_batches": None,
            "amp_enabled": False,
            "activation_checkpointing": False,
            "offload_saved_tensors": False,
            "micro_batch_size": None,
            "batch_size": 16,
            "seed": 42,
            "synthetic_seeds": [7, 13, 23],
            "max_normal_images": 16,
            "synthetic_variants": 4,
            "max_normal_fpr": 0.005,
        }
        values.update(overrides)
        return values

    @staticmethod
    def _provenance(**configuration_overrides):
        configuration = {
            "epochs": 10,
            "batch_size": 16,
            "seed": 42,
            "max_train_batches": None,
            "amp_enabled": False,
            "activation_checkpointing": False,
            "offload_saved_tensors": False,
            "micro_batch_size": None,
        }
        configuration.update(configuration_overrides)
        return {
            "paper_eligible_full_training": False,
            "model_configuration": configuration,
            "synthetic_seeds": [7, 13, 23],
            "max_normal_images": 16,
            "synthetic_variants": 4,
            "max_normal_fpr": 0.005,
        }

    def test_official_image_score_is_top_50_mean(self) -> None:
        heatmaps = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
        score = official_image_scores(heatmaps)
        expected = torch.arange(14, 64, dtype=torch.float32).mean()
        torch.testing.assert_close(score, expected.reshape(1))

    def test_image_score_validates_shape(self) -> None:
        with self.assertRaises(ValueError):
            official_image_scores(torch.zeros(2, 8, 8))

    def test_focal_loss_is_finite(self) -> None:
        loss = BinaryFocalLoss()(
            torch.full((2, 1, 4, 4), 0.5),
            torch.zeros(2, 1, 4, 4),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0)

    def test_completed_output_requires_exact_training_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            path.write_text(
                json.dumps(self._provenance()),
                encoding="utf-8",
            )
            self.assertTrue(
                completed_output_matches_request(
                    path,
                    **self._request_kwargs(),
                )
            )
            self.assertFalse(
                completed_output_matches_request(
                    path,
                    **self._request_kwargs(epochs=3000),
                )
            )

    def test_paper_eligible_flag_must_match_full_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            path.write_text(
                json.dumps(
                    {
                        **self._provenance(epochs=3000),
                        "paper_eligible_full_training": False,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(
                completed_output_matches_request(
                    path,
                    **self._request_kwargs(epochs=3000),
                )
            )

    def test_completed_output_requires_exact_precision_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            path.write_text(
                json.dumps(
                    {
                        "paper_eligible_full_training": False,
                        "model_configuration": {
                            "epochs": 10,
                            "batch_size": 16,
                            "seed": 42,
                            "max_train_batches": None,
                            "amp_enabled": True,
                            "activation_checkpointing": False,
                            "offload_saved_tensors": False,
                            "micro_batch_size": None,
                        },
                        "synthetic_seeds": [7, 13, 23],
                        "max_normal_images": 16,
                        "synthetic_variants": 4,
                        "max_normal_fpr": 0.005,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                completed_output_matches_request(
                    path,
                    **self._request_kwargs(amp_enabled=True),
                )
            )
            self.assertFalse(
                completed_output_matches_request(
                    path,
                    **self._request_kwargs(amp_enabled=False),
                )
            )

    def test_completed_output_requires_exact_checkpointing_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            path.write_text(
                json.dumps(
                    {
                        "paper_eligible_full_training": False,
                        "model_configuration": {
                            "epochs": 10,
                            "batch_size": 16,
                            "seed": 42,
                            "max_train_batches": None,
                            "amp_enabled": True,
                            "activation_checkpointing": True,
                            "offload_saved_tensors": False,
                            "micro_batch_size": None,
                        },
                        "synthetic_seeds": [7, 13, 23],
                        "max_normal_images": 16,
                        "synthetic_variants": 4,
                        "max_normal_fpr": 0.005,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                completed_output_matches_request(
                    path,
                    **self._request_kwargs(
                        amp_enabled=True,
                        activation_checkpointing=True,
                    ),
                )
            )
            self.assertFalse(
                completed_output_matches_request(
                    path,
                    **self._request_kwargs(
                        amp_enabled=True,
                        activation_checkpointing=False,
                    ),
                )
            )

    def test_completed_output_requires_exact_offload_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            path.write_text(
                json.dumps(
                    {
                        "paper_eligible_full_training": False,
                        "model_configuration": {
                            "epochs": 10,
                            "batch_size": 16,
                            "seed": 42,
                            "max_train_batches": None,
                            "amp_enabled": True,
                            "activation_checkpointing": False,
                            "offload_saved_tensors": True,
                            "micro_batch_size": None,
                        },
                        "synthetic_seeds": [7, 13, 23],
                        "max_normal_images": 16,
                        "synthetic_variants": 4,
                        "max_normal_fpr": 0.005,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                completed_output_matches_request(
                    path,
                    **self._request_kwargs(
                        amp_enabled=True,
                        offload_saved_tensors=True,
                    ),
                )
            )
            self.assertFalse(
                completed_output_matches_request(
                    path,
                    **self._request_kwargs(
                        amp_enabled=True,
                        offload_saved_tensors=False,
                    ),
                )
            )

    def test_completed_output_requires_exact_micro_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "provenance.json"
            path.write_text(
                json.dumps(
                    {
                        "paper_eligible_full_training": False,
                        "model_configuration": {
                            "epochs": 10,
                            "batch_size": 16,
                            "seed": 42,
                            "max_train_batches": None,
                            "amp_enabled": True,
                            "activation_checkpointing": False,
                            "offload_saved_tensors": False,
                            "micro_batch_size": 4,
                        },
                        "synthetic_seeds": [7, 13, 23],
                        "max_normal_images": 16,
                        "synthetic_variants": 4,
                        "max_normal_fpr": 0.005,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                completed_output_matches_request(
                    path,
                    **self._request_kwargs(
                        amp_enabled=True,
                        micro_batch_size=4,
                    ),
                )
            )
            self.assertFalse(
                completed_output_matches_request(
                    path,
                    **self._request_kwargs(
                        amp_enabled=True,
                        micro_batch_size=8,
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
