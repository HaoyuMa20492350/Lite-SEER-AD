import numpy as np

from scripts.eval.paired_bca_stats import (
    _read_paired_deltas,
    holm_adjust,
    paired_bca_ci,
    paired_sign_test,
)


def test_paired_sign_test_counts_wins_losses_and_ties():
    result = paired_sign_test(np.asarray([0.2, -0.1, 0.0, 0.3]))

    assert result["wins"] == 2
    assert result["losses"] == 1
    assert result["ties"] == 1
    assert 0.0 <= result["paired_sign_test_p"] <= 1.0


def test_bca_ci_contains_mean_for_simple_positive_deltas():
    deltas = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    lower, upper = paired_bca_ci(deltas, iterations=500, alpha=0.1, seed=7)

    assert lower < float(np.mean(deltas)) < upper


def test_holm_adjust_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted[0] <= adjusted[2] <= adjusted[1]
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_read_paired_deltas_supports_row_filter(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "method,lite,external\npadim,0.8,0.7\npatchcore,0.9,0.95\n",
        encoding="utf-8",
    )

    deltas = _read_paired_deltas(
        csv_path,
        method_column="lite",
        baseline_column="external",
        where_column="method",
        where_equals="padim",
    )

    assert np.allclose(deltas, [0.1])
