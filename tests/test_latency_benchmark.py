from __future__ import annotations

import unittest

from seer_ad_v2.evaluation.metrics_efficiency import benchmark_callable


class LatencyBenchmarkTests(unittest.TestCase):
    def test_benchmark_records_protocol_and_counts(self) -> None:
        calls = 0

        def work() -> None:
            nonlocal calls
            calls += 1

        result = benchmark_callable(
            work,
            warmups=2,
            repeats=3,
            batch_size=1,
        )
        self.assertEqual(calls, 5)
        self.assertEqual(
            result["latency_protocol"],
            "synchronized_batch_latency_v1",
        )
        self.assertEqual(result["latency_warmups"], 2)
        self.assertEqual(result["latency_repeats"], 3)
        self.assertGreaterEqual(result["latency_ms_p95"], 0.0)
        self.assertGreaterEqual(result["latency_ms_p99"], result["latency_ms_p95"])

    def test_invalid_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            benchmark_callable(lambda: None, repeats=0)


if __name__ == "__main__":
    unittest.main()
