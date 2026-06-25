from __future__ import annotations

import unittest

from baselines.registry import BASELINES, LOCAL_BASELINES


class BaselineRegistryTests(unittest.TestCase):
    def test_local_baselines_are_not_marked_official(self) -> None:
        for method in LOCAL_BASELINES:
            spec = BASELINES[method]
            self.assertFalse(spec.official_implementation)
            self.assertIn(
                spec.implementation_variant,
                {"local_reimplementation", "lite_reimplementation"},
            )
            self.assertTrue(spec.source_path)

    def test_simplified_models_are_displayed_as_lite(self) -> None:
        for method in (
            "simplenet",
            "draem",
            "rd4ad",
            "uniad",
            "diffusionad",
            "ddad",
        ):
            self.assertTrue(BASELINES[method].display_name.endswith("-Lite"))


if __name__ == "__main__":
    unittest.main()
