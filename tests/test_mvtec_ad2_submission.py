from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from seer_ad_v2.evaluation.mvtec_ad2_submission import (
    assert_export_root_available,
    assert_path_outside_submission,
    assert_submission_root,
    create_submission_archive,
    default_metadata_dir,
    sha256_file,
)
from tools.export_mvtec_ad2_submission import prepare_submission_map


class MVTecAD2SubmissionTests(unittest.TestCase):
    def test_submission_map_can_preserve_model_resolution(self) -> None:
        heatmap = np.arange(16, dtype=np.float32).reshape(4, 4)
        output = prepare_submission_map(heatmap, (12, 8), "model")
        self.assertEqual(output.shape, (4, 4))
        np.testing.assert_array_equal(output, heatmap)

    def test_submission_map_can_expand_to_native_resolution(self) -> None:
        heatmap = np.arange(16, dtype=np.float32).reshape(4, 4)
        output = prepare_submission_map(heatmap, (12, 8), "native")
        self.assertEqual(output.shape, (8, 12))
        self.assertEqual(output.dtype, np.float32)

    def test_submission_map_rejects_unknown_resolution(self) -> None:
        with self.assertRaises(ValueError):
            prepare_submission_map(
                np.zeros((4, 4), dtype=np.float32),
                (12, 8),
                "unknown",
            )

    def test_metadata_is_kept_outside_submission_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission"
            (submission / "anomaly_images").mkdir(parents=True)
            (submission / "anomaly_images_thresholded").mkdir()
            metadata = default_metadata_dir(submission)
            metadata.mkdir()
            (metadata / "submission_protocol.json").write_text(
                "{}",
                encoding="utf-8",
            )
            assert_submission_root(submission)
            self.assertNotIn(
                "submission_protocol.json",
                {path.name for path in submission.iterdir()},
            )

    def test_extra_root_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission"
            (submission / "anomaly_images").mkdir(parents=True)
            (submission / "anomaly_images_thresholded").mkdir()
            (submission / "submission_protocol.json").write_text(
                "{}",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                assert_submission_root(submission)
            with self.assertRaises(ValueError):
                assert_export_root_available(submission)

    def test_metadata_and_archive_must_stay_outside_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            submission = Path(tmp) / "submission"
            submission.mkdir()
            with self.assertRaises(ValueError):
                assert_path_outside_submission(
                    submission,
                    submission / "metadata",
                    "metadata",
                )
            assert_path_outside_submission(
                submission,
                submission.with_name("submission_metadata"),
                "metadata",
            )

    def test_archive_has_single_submission_top_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            submission = root / "submission"
            continuous = submission / "anomaly_images" / "can" / "test_private"
            thresholded = (
                submission
                / "anomaly_images_thresholded"
                / "can"
                / "test_private"
            )
            continuous.mkdir(parents=True)
            thresholded.mkdir(parents=True)
            tifffile.imwrite(
                continuous / "000_regular.tiff",
                np.zeros((1, 1), dtype=np.float16),
            )
            Image.fromarray(np.zeros((1, 1), dtype=np.uint8)).save(
                thresholded / "000_regular.png"
            )
            archive = create_submission_archive(
                submission,
                root / "submission.tar.gz",
            )
            self.assertTrue(archive.is_file())

    @unittest.skipUnless(
        (
            Path(__file__).resolve().parents[1]
            / "submissions"
            / "mvtec_ad2_seed7_model256.tar.gz"
        ).is_file(),
        "local checker-passed MVTec AD 2 archive is unavailable",
    )
    def test_local_archive_hash_matches_protocol_and_documents(self) -> None:
        root = Path(__file__).resolve().parents[1]
        archive = root / "submissions" / "mvtec_ad2_seed7_model256.tar.gz"
        protocol = json.loads(
            (
                root
                / "submissions"
                / "mvtec_ad2_seed7_model256_metadata"
                / "submission_protocol.json"
            ).read_text(encoding="utf-8")
        )
        actual = sha256_file(archive)
        self.assertEqual(protocol["archive_sha256"], actual)
        for relative in (
            "docs/current_stage_and_next_plan_zh.md",
            "docs/mvtec_ad2_upload_status_zh.md",
            "docs/paper_protocol.md",
            "docs/results_limitations_draft.md",
        ):
            self.assertIn(
                actual,
                (root / relative).read_text(encoding="utf-8"),
                relative,
            )

    @unittest.skipUnless(
        (
            Path(__file__).resolve().parents[1]
            / "official_mvtec_ad2_utils"
            / "MVTecAD2_public_code_utils"
            / "check_and_prepare_data_for_upload.py"
        ).is_file(),
        "official MVTec AD 2 checker is not installed",
    )
    def test_official_checker_accepts_exact_structure(self) -> None:
        utils_dir = (
            Path(__file__).resolve().parents[1]
            / "official_mvtec_ad2_utils"
            / "MVTecAD2_public_code_utils"
        )
        sys.path.insert(0, str(utils_dir))
        try:
            import check_and_prepare_data_for_upload as checker
            import utils as official_utils

            with tempfile.TemporaryDirectory() as tmp:
                submission = Path(tmp) / "submission"
                original_counts = dict(official_utils.OBJECT_FILE_COUNTER)
                official_utils.OBJECT_FILE_COUNTER.update(
                    {category: 1 for category in original_counts}
                )
                try:
                    for output_dir, suffix, writer in (
                        (
                            "anomaly_images",
                            ".tiff",
                            lambda path: tifffile.imwrite(
                                path,
                                np.zeros((1, 1), dtype=np.float16),
                            ),
                        ),
                        (
                            "anomaly_images_thresholded",
                            ".png",
                            lambda path: Image.fromarray(
                                np.zeros((1, 1), dtype=np.uint8)
                            ).save(path),
                        ),
                    ):
                        for category in sorted(original_counts):
                            for split, name_suffix in (
                                ("test_private", "regular"),
                                ("test_private_mixed", "mixed"),
                            ):
                                folder = submission / output_dir / category / split
                                folder.mkdir(parents=True, exist_ok=True)
                                writer(folder / f"000_{name_suffix}{suffix}")
                    checker.check_submission(str(submission))
                finally:
                    official_utils.OBJECT_FILE_COUNTER.clear()
                    official_utils.OBJECT_FILE_COUNTER.update(original_counts)
        finally:
            sys.path.remove(str(utils_dir))


if __name__ == "__main__":
    unittest.main()
