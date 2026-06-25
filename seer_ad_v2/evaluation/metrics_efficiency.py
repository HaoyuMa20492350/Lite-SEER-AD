from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


@dataclass
class TimerResult:
    elapsed_ms: float = 0.0


@contextmanager
def timed_cuda(device: str = "cpu"):
    result = TimerResult()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield result
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    result.elapsed_ms = (time.perf_counter() - start) * 1000.0


def benchmark_callable(
    function: Callable[[], Any],
    *,
    device: str = "cpu",
    warmups: int = 50,
    repeats: int = 200,
    batch_size: int = 1,
) -> dict[str, Any]:
    if warmups < 0 or repeats < 1 or batch_size < 1:
        raise ValueError(
            "warmups >= 0, repeats >= 1, and batch_size >= 1 are required"
        )

    def synchronize() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    for _ in range(warmups):
        function()
    synchronize()

    elapsed_ms = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        function()
        synchronize()
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    values = np.asarray(elapsed_ms, dtype=np.float64) / float(batch_size)
    mean = float(values.mean())
    return {
        "latency_protocol": "synchronized_batch_latency_v1",
        "latency_batch_size": int(batch_size),
        "latency_warmups": int(warmups),
        "latency_repeats": int(repeats),
        "latency_ms_mean": mean,
        "latency_ms_std": float(values.std()),
        "latency_ms_p50": float(np.quantile(values, 0.50)),
        "latency_ms_p95": float(np.quantile(values, 0.95)),
        "latency_ms_p99": float(np.quantile(values, 0.99)),
        "fps": float(1000.0 / mean) if mean > 0.0 else 0.0,
    }
