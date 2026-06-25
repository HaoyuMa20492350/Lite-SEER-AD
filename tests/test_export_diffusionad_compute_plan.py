from __future__ import annotations

import unittest

from tools.export_diffusionad_compute_plan import (
    category_plan,
    greedy_walltime,
    measured_seconds_per_batch,
)


class ExportDiffusionADComputePlanTests(unittest.TestCase):
    def test_measured_seconds_per_batch_uses_measured_epoch_segment(self) -> None:
        history = {
            "resumed_from_epoch": 10,
            "epochs": 12,
            "training_seconds_this_invocation": 120.0,
            "history": [
                {"epoch": 10, "batches": 3},
                {"epoch": 11, "batches": 3},
                {"epoch": 12, "batches": 3},
            ],
        }
        self.assertEqual(measured_seconds_per_batch(history), 20.0)

    def test_category_plan_uses_drop_last_batches(self) -> None:
        plan = category_plan(
            "example",
            60,
            batch_size=16,
            epochs=3000,
            seconds_per_batch=20.0,
        )
        self.assertEqual(plan["batches_per_epoch"], 3)
        self.assertEqual(plan["optimizer_steps"], 9000)
        self.assertEqual(plan["estimated_gpu_hours"], 50.0)

    def test_greedy_walltime_assigns_largest_categories_first(self) -> None:
        rows = [
            {"estimated_gpu_hours": 10.0},
            {"estimated_gpu_hours": 8.0},
            {"estimated_gpu_hours": 5.0},
            {"estimated_gpu_hours": 3.0},
        ]
        self.assertEqual(greedy_walltime(rows, 2), 13.0)


if __name__ == "__main__":
    unittest.main()
