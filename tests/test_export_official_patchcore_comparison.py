from __future__ import annotations

import unittest

from tools.export_official_patchcore_comparison import METRICS


class OfficialPatchCoreComparisonTests(unittest.TestCase):
    def test_comparison_includes_fixed_and_oracle_dice(self) -> None:
        self.assertIn("dice", METRICS)
        self.assertIn("oracle_dice", METRICS)
        self.assertIn("aupro", METRICS)


if __name__ == "__main__":
    unittest.main()
