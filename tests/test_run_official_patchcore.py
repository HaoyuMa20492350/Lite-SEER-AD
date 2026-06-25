from __future__ import annotations

import unittest

from tools.run_official_patchcore import split_ints, stable_seed


class RunOfficialPatchCoreTests(unittest.TestCase):
    def test_split_ints(self) -> None:
        self.assertEqual(split_ints("7,13,23"), [7, 13, 23])
        with self.assertRaises(ValueError):
            split_ints("")

    def test_stable_seed(self) -> None:
        first = stable_seed(7, "bottle", "image.png", 0)
        self.assertEqual(first, stable_seed(7, "bottle", "image.png", 0))
        self.assertNotEqual(first, stable_seed(7, "bottle", "image.png", 1))


if __name__ == "__main__":
    unittest.main()
