"""Paired BCa bootstrap and sign-test statistics for paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import binomtest, norm


def _read_paired_deltas(
    csv_path: Path,
    method_column: str,
    baseline_column: str,
    where_column: str | None = None,
    where_equals: str | None = None,
) -> np.ndarray:
    deltas: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {method_column, baseline_column}
        if where_column:
            required.add(where_column)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required column(s): {sorted(missing)}")
        for row in reader:
            if where_column and row.get(where_column) != where_equals:
                continue
            method_raw = row.get(method_column, "")
            baseline_raw = row.get(baseline_column, "")
            if method_raw == "" or baseline_raw == "":
                continue
            deltas.append(float(method_raw) - float(baseline_raw))
    if not deltas:
        raise ValueError("No paired numeric rows were found.")
    return np.asarray(deltas, dtype=np.float64)


def _clamp_probability(value: float) -> float:
    eps = 1e-9
    return min(1.0 - eps, max(eps, value))


def paired_bca_ci(
    deltas: np.ndarray,
    iterations: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Return the BCa confidence interval for the paired mean delta."""

    if deltas.ndim != 1:
        raise ValueError("deltas must be one-dimensional")
    if len(deltas) < 3:
        lower, upper = np.quantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lower), float(upper)

    rng = np.random.default_rng(seed)
    theta_hat = float(np.mean(deltas))
    boot = np.empty(iterations, dtype=np.float64)
    n = len(deltas)
    for i in range(iterations):
        indices = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(deltas[indices]))

    z0 = norm.ppf(_clamp_probability(float(np.mean(boot < theta_hat))))

    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack[i] = float(np.mean(np.delete(deltas, i)))
    jack_mean = float(np.mean(jack))
    centered = jack_mean - jack
    numerator = float(np.sum(centered**3))
    denominator = float(6.0 * (np.sum(centered**2) ** 1.5))
    acceleration = 0.0 if denominator == 0.0 else numerator / denominator

    quantiles: list[float] = []
    for percentile in (alpha / 2.0, 1.0 - alpha / 2.0):
        z_alpha = norm.ppf(percentile)
        adjusted = norm.cdf(
            z0 + (z0 + z_alpha) / (1.0 - acceleration * (z0 + z_alpha))
        )
        quantiles.append(_clamp_probability(float(adjusted)))

    lower, upper = np.quantile(boot, quantiles)
    return float(lower), float(upper)


def paired_sign_test(deltas: np.ndarray) -> dict[str, float | int]:
    wins = int(np.sum(deltas > 0))
    losses = int(np.sum(deltas < 0))
    ties = int(np.sum(deltas == 0))
    n = wins + losses
    p_value = 1.0 if n == 0 else float(binomtest(min(wins, losses), n, 0.5).pvalue)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "paired_sign_test_p": min(1.0, p_value),
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = list(p_values)
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0 for _ in values]
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        corrected = min(1.0, values[index] * (m - rank))
        running = max(running, corrected)
        adjusted[index] = running
    return adjusted


def summarize(
    deltas: np.ndarray,
    iterations: int,
    alpha: float,
    seed: int,
    metric_name: str,
    method_column: str,
    baseline_column: str,
) -> dict[str, object]:
    lower, upper = paired_bca_ci(deltas, iterations=iterations, alpha=alpha, seed=seed)
    summary: dict[str, object] = {
        "metric": metric_name,
        "method_column": method_column,
        "baseline_column": baseline_column,
        "delta_definition": "method_minus_baseline",
        "n_pairs": int(len(deltas)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "bca_ci": {
            "alpha": alpha,
            "lower": lower,
            "upper": upper,
            "iterations": iterations,
            "seed": seed,
        },
    }
    summary.update(paired_sign_test(deltas))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--method-column", required=True)
    parser.add_argument("--baseline-column", required=True)
    parser.add_argument("--metric-name", default="metric")
    parser.add_argument("--where-column")
    parser.add_argument("--where-equals")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if not math.isfinite(args.alpha) or not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in (0, 1)")
    if bool(args.where_column) != bool(args.where_equals):
        raise ValueError("--where-column and --where-equals must be supplied together")

    deltas = _read_paired_deltas(
        args.csv,
        args.method_column,
        args.baseline_column,
        where_column=args.where_column,
        where_equals=args.where_equals,
    )
    summary = summarize(
        deltas,
        iterations=args.iterations,
        alpha=args.alpha,
        seed=args.seed,
        metric_name=args.metric_name,
        method_column=args.method_column,
        baseline_column=args.baseline_column,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
