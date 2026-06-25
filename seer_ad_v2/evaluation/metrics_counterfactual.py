from __future__ import annotations

import numpy as np


def score_drop_after_repair(before_scores: list[float], after_scores: list[float]) -> dict[str, float]:
    before = np.asarray(before_scores, dtype=np.float32)
    after = np.asarray(after_scores, dtype=np.float32)
    drops = np.maximum(0.0, before - after)
    return {
        "sdr_mean": float(drops.mean()) if drops.size else 0.0,
        "sdr_median": float(np.median(drops)) if drops.size else 0.0,
        "sdr_positive_rate": float((drops > 0).mean()) if drops.size else 0.0,
    }
