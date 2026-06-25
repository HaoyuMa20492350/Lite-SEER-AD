from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.audit_official_source_environments import parse_lfs_pointer
from tools.materialize_patchcore_pretrained import (
    media_url,
    selected_categories,
)


class OfficialSourceEnvironmentTests(unittest.TestCase):
    def test_parse_lfs_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.faiss"
            path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:" + "a" * 64 + "\n"
                "size 123\n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_lfs_pointer(path),
                {"sha256": "a" * 64, "size": 123},
            )

    def test_media_url_is_pinned(self) -> None:
        url = media_url(
            "https://github.com/amazon-science/patchcore-inspection",
            "f" * 40,
            Path("models") / "bundle name" / "index.faiss",
        )
        self.assertIn("/" + "f" * 40 + "/", url)
        self.assertIn("bundle%20name", url)

    def test_category_selection(self) -> None:
        self.assertEqual(selected_categories("bottle,cable"), ["bottle", "cable"])
        self.assertEqual(len(selected_categories("all")), 15)
        with self.assertRaises(ValueError):
            selected_categories("unknown")


if __name__ == "__main__":
    unittest.main()
