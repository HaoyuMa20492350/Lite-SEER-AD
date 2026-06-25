from __future__ import annotations

import unittest
from pathlib import Path

import torch

from tools.run_reference_padim import (
    PADIM_CORE_FILES,
    load_padim_model_class,
)


class RunReferencePadimTests(unittest.TestCase):
    def test_core_file_manifest_exists(self) -> None:
        source_root = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "official_baselines"
            / "padim"
        )
        self.assertTrue(
            all((source_root / relative).is_file() for relative in PADIM_CORE_FILES)
        )

    def test_pinned_core_model_loads_without_cli_stack(self) -> None:
        source_root = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "official_baselines"
            / "padim"
        )
        model_class = load_padim_model_class(source_root)
        torch.manual_seed(42)
        model = model_class(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            pre_trained=False,
            n_features=8,
        )
        model.train()
        model(torch.zeros(2, 3, 64, 64))
        model.fit()
        model.eval()
        output = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.anomaly_map.shape), (1, 1, 64, 64))
        self.assertEqual(tuple(output.pred_score.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
