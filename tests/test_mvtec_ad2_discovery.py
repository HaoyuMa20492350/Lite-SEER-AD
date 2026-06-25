from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from seer_ad_v2.data.mvtec_ad2_discovery import (
    MVTEC_AD2_CATEGORIES,
    MVTEC_AD2_REQUIRED_PATHS,
    analyse_member_names,
    inspect_archive,
)


class MVTecAD2DiscoveryTests(unittest.TestCase):
    def test_complete_archive_layout_is_detected(self) -> None:
        names = [
            f"mvtec_ad_2/{category}/{required}/000.png"
            for category in MVTEC_AD2_CATEGORIES
            for required in MVTEC_AD2_REQUIRED_PATHS
        ]
        rows = analyse_member_names(names)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["ready"])
        self.assertEqual(rows[0]["root_prefix"], "mvtec_ad_2")

    def test_unrelated_archive_is_not_detected(self) -> None:
        rows = analyse_member_names(
            [
                "Dataset/New Asset/BlueBottle/input/photo.png",
                "Dataset/New Asset/BlueBottle/stage3/out/result.png",
            ]
        )
        self.assertEqual(rows, [])

    def test_missing_public_masks_prevents_readiness(self) -> None:
        names = [
            f"{category}/{required}/000.png"
            for category in MVTEC_AD2_CATEGORIES
            for required in MVTEC_AD2_REQUIRED_PATHS
            if required != "test_public/ground_truth/bad"
        ]
        rows = analyse_member_names(names)
        self.assertFalse(rows[0]["ready"])
        self.assertEqual(
            rows[0]["missing"]["can"],
            ["test_public/ground_truth/bad"],
        )

    def test_corrupt_archive_is_reported_without_raising(self) -> None:
        with patch(
            "seer_ad_v2.data.mvtec_ad2_discovery.archive_member_names",
            side_effect=EOFError("truncated"),
        ):
            result = inspect_archive(Path("broken.tar.gz"))
        self.assertFalse(result["ready"])
        self.assertIn("truncated", result["error"])


if __name__ == "__main__":
    unittest.main()
