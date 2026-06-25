from __future__ import annotations

import unittest

from baselines.official_sources import (
    load_official_source_manifest,
    validate_official_provenance,
)


class OfficialBaselineSourceTests(unittest.TestCase):
    def test_manifest_has_pinned_sources(self) -> None:
        manifest = load_official_source_manifest()
        self.assertEqual(
            set(manifest["sources"]),
            {
                "patchcore",
                "padim",
                "simplenet",
                "draem",
                "rd4ad",
                "uniad",
                "diffusionad",
                "ddad",
            },
        )

    def test_provenance_must_match_pinned_source(self) -> None:
        source = load_official_source_manifest()["sources"]["patchcore"]
        provenance = {
            "method": "patchcore",
            "dataset": "mvtec15",
            "category": "bottle",
            "source_kind": source["source_kind"],
            "source_repository": source["repository"],
            "source_commit": source["commit"],
            "official_implementation": True,
            "execution_command": "python bin/run_patchcore.py",
            "environment": "requirements.lock",
            "checkpoint_source": "trained_from_normal_data",
        }
        self.assertEqual(
            validate_official_provenance(
                provenance,
                source,
                method="patchcore",
                dataset="mvtec15",
                category="bottle",
            ),
            [],
        )
        provenance["source_commit"] = "0" * 40
        self.assertTrue(
            validate_official_provenance(
                provenance,
                source,
                method="patchcore",
                dataset="mvtec15",
                category="bottle",
            )
        )

    def test_maintained_reference_is_not_marked_author_official(self) -> None:
        source = load_official_source_manifest()["sources"]["padim"]
        provenance = {
            "method": "padim",
            "dataset": "mvtec15",
            "category": "bottle",
            "source_kind": source["source_kind"],
            "source_repository": source["repository"],
            "source_commit": source["commit"],
            "official_implementation": False,
            "execution_command": "anomalib train --model Padim",
            "environment": "uv.lock",
            "checkpoint_source": "trained_from_normal_data",
        }
        self.assertEqual(
            validate_official_provenance(
                provenance,
                source,
                method="padim",
                dataset="mvtec15",
                category="bottle",
            ),
            [],
        )

    def test_structured_environment_is_valid(self) -> None:
        source = load_official_source_manifest()["sources"]["patchcore"]
        provenance = {
            "method": "patchcore",
            "dataset": "mvtec15",
            "category": "bottle",
            "source_kind": source["source_kind"],
            "source_repository": source["repository"],
            "source_commit": source["commit"],
            "official_implementation": True,
            "execution_command": "python tools/run_official_patchcore.py",
            "environment": {"torch": "2.4.0", "faiss-cpu": "1.12.0"},
            "checkpoint_source": "pinned official Git LFS bundle",
        }
        self.assertEqual(
            validate_official_provenance(
                provenance,
                source,
                method="patchcore",
                dataset="mvtec15",
                category="bottle",
            ),
            [],
        )

    def test_truncated_training_is_not_paper_eligible(self) -> None:
        source = load_official_source_manifest()["sources"]["diffusionad"]
        provenance = {
            "method": "diffusionad",
            "dataset": "mvtec15",
            "category": "bottle",
            "source_kind": source["source_kind"],
            "source_repository": source["repository"],
            "source_commit": source["commit"],
            "official_implementation": True,
            "execution_command": "python tools/run_official_diffusionad.py",
            "environment": {"torch": "2.4.0"},
            "checkpoint_source": "one-step smoke checkpoint",
            "paper_eligible_full_training": False,
        }
        errors = validate_official_provenance(
            provenance,
            source,
            method="diffusionad",
            dataset="mvtec15",
            category="bottle",
        )
        self.assertIn(
            "artifact is explicitly marked as incomplete for paper use",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
