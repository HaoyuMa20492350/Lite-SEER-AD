from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.run_mvtec15_baselines import dataset_name, external_prediction_path


class RunMvtec15BaselinesTests(unittest.TestCase):
    def test_mvtec_config_uses_canonical_dataset_id(self) -> None:
        with patch(
            "tools.run_mvtec15_baselines._load_yaml",
            return_value={"dataset": {"name": "mvtec"}},
        ):
            self.assertEqual(dataset_name("unused.yaml"), "mvtec15")

    def test_canonical_external_prediction_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = (
                root
                / "mvtec15"
                / "patchcore"
                / "bottle"
                / "predictions.npz"
            )
            expected.parent.mkdir(parents=True)
            expected.touch()
            self.assertEqual(
                external_prediction_path(
                    root,
                    "mvtec15",
                    "patchcore",
                    "bottle",
                ),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
