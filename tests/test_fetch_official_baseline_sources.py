from __future__ import annotations

import unittest
from pathlib import Path

from tools.fetch_official_baseline_sources import _repo_slug, _selected


class FetchOfficialBaselineSourceTests(unittest.TestCase):
    def test_repo_slug(self) -> None:
        self.assertEqual(
            _repo_slug("https://github.com/amazon-science/patchcore-inspection"),
            "amazon-science/patchcore-inspection",
        )

    def test_method_selection(self) -> None:
        available = {"patchcore", "draem"}
        self.assertEqual(_selected("all", available), ["draem", "patchcore"])
        self.assertEqual(_selected("patchcore", available), ["patchcore"])
        with self.assertRaises(ValueError):
            _selected("unknown", available)

    def test_workspace_root_exists(self) -> None:
        self.assertTrue(Path(__file__).resolve().parents[1].exists())


if __name__ == "__main__":
    unittest.main()
