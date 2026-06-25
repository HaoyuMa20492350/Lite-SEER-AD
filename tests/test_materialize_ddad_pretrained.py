from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.materialize_ddad_pretrained import (
    DDAD_MVTEC_FILES,
    DDAD_MVTEC_SETTINGS,
    download_with_retry,
    selected_categories,
)


class MaterializeDDADPretrainedTests(unittest.TestCase):
    def test_every_category_has_two_declared_files(self) -> None:
        expected = set()
        for category, settings in DDAD_MVTEC_SETTINGS.items():
            expected.add(f"{category}/{settings['unet_checkpoint']}")
            expected.add(f"{category}/{settings['feature_checkpoint']}")
        self.assertEqual(set(DDAD_MVTEC_FILES), expected)
        self.assertEqual(len(expected), 30)

    def test_selected_categories_rejects_unknown_names(self) -> None:
        self.assertEqual(len(selected_categories("all")), 15)
        with self.assertRaises(ValueError):
            selected_categories("bottle,missing")

    def test_download_retries_after_transient_failure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint"

            def fake_download(**_: object) -> str:
                if fake_download.calls == 0:
                    fake_download.calls += 1
                    raise OSError("transient")
                path.write_bytes(b"checkpoint")
                return str(path)

            fake_download.calls = 0
            with (
                patch(
                    "tools.materialize_ddad_pretrained.gdown.download",
                    side_effect=fake_download,
                ),
                patch("tools.materialize_ddad_pretrained.time.sleep"),
            ):
                download_with_retry(
                    "file-id",
                    path,
                    attempts=2,
                    initial_delay_seconds=0,
                )
            self.assertEqual(path.read_bytes(), b"checkpoint")


if __name__ == "__main__":
    unittest.main()
