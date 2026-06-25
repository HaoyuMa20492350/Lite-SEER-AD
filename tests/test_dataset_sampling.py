from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seer_ad_v2.data.datasets import ImageRecord, _limit_records, list_mvtec_ad2, list_visa


class DatasetSamplingTests(unittest.TestCase):
    def test_seeded_single_class_sampling_is_deterministic(self) -> None:
        records = [
            ImageRecord(Path(f"{index:03d}.png"), 0, None, "demo", "good")
            for index in range(20)
        ]
        first = _limit_records(records, 8, sample_seed=7)
        repeat = _limit_records(records, 8, sample_seed=7)
        different = _limit_records(records, 8, sample_seed=13)
        self.assertEqual([row.image_path for row in first], [row.image_path for row in repeat])
        self.assertNotEqual([row.image_path for row in first], [row.image_path for row in different])
        self.assertEqual(len(first), 8)

    def test_mvtec_ad2_public_and_private_layout(self) -> None:
        def fake_images(path: Path) -> list[Path]:
            return [path / "000.png"]

        expected_mask = (
            Path("dataset")
            / "can"
            / "test_public"
            / "ground_truth"
            / "bad"
            / "000_mask.png"
        )
        with (
            patch(
                "seer_ad_v2.data.datasets.list_images",
                side_effect=fake_images,
            ),
            patch.object(
                Path,
                "exists",
                autospec=True,
                side_effect=lambda path: path == expected_mask,
            ),
        ):
            public = list_mvtec_ad2(Path("dataset"), "can", "test_public")
            private = list_mvtec_ad2(Path("dataset"), "can", "test_private")
        self.assertEqual([row.label for row in public], [0, 1])
        self.assertEqual([row.defect_type for row in public], ["good", "bad"])
        self.assertEqual(public[1].mask_path, expected_mask)
        self.assertEqual(len(private), 1)
        self.assertEqual(private[0].defect_type, "private")

    def test_visa_uses_official_one_class_split_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "split_csv"
            split_dir.mkdir()
            rows = [
                {
                    "object": "candle",
                    "split": "train",
                    "label": "normal",
                    "image": "candle/Data/Images/Normal/train.JPG",
                    "mask": "",
                },
                {
                    "object": "candle",
                    "split": "test",
                    "label": "normal",
                    "image": "candle/Data/Images/Normal/test.JPG",
                    "mask": "",
                },
                {
                    "object": "candle",
                    "split": "test",
                    "label": "anomaly",
                    "image": "candle/Data/Images/Anomaly/bad.JPG",
                    "mask": "candle/Data/Masks/Anomaly/bad.png",
                },
                {
                    "object": "capsules",
                    "split": "test",
                    "label": "normal",
                    "image": "capsules/Data/Images/Normal/test.JPG",
                    "mask": "",
                },
            ]
            with (split_dir / "1cls.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["object", "split", "label", "image", "mask"],
                )
                writer.writeheader()
                writer.writerows(rows)

            train = list_visa(root, "candle", "train")
            test = list_visa(root, "candle", "test")

        self.assertEqual([row.image_path.name for row in train], ["train.JPG"])
        self.assertEqual([row.image_path.name for row in test], ["test.JPG", "bad.JPG"])
        self.assertEqual([row.label for row in test], [0, 1])
        self.assertIsNone(test[0].mask_path)
        self.assertEqual(test[1].mask_path.name, "bad.png")
        self.assertTrue(
            {row.image_path for row in train}.isdisjoint(
                {row.image_path for row in test}
            )
        )

    def test_visa_refuses_to_run_without_official_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "1cls.csv"):
                list_visa(Path(tmp), "candle", "test")


if __name__ == "__main__":
    unittest.main()
