from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.materialize_diffusionad_foregrounds import (
    normalize_drive_path,
    valid_image,
)


class MaterializeDiffusionADForegroundsTests(unittest.TestCase):
    def test_normalizes_author_metal_nut_directory(self) -> None:
        path = normalize_drive_path(
            r"metal_ nut\DISthresh\good\000.png"
        )
        self.assertEqual(
            path.as_posix(),
            "metal_nut/DISthresh/good/000.png",
        )

    def test_valid_image_rejects_non_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-image.png"
            path.write_text("not an image", encoding="utf-8")
            self.assertFalse(valid_image(path))

    def test_valid_image_accepts_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.new("L", (8, 8), color=255).save(path)
            self.assertTrue(valid_image(path))


if __name__ == "__main__":
    unittest.main()
