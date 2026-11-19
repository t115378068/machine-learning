"""Tests for the synthetic forecasting example."""

import numpy as np
import pytest

from machine_learning.demo import evaluate_demo, make_series, make_supervised


def test_series_is_reproducible() -> None:
    np.testing.assert_array_equal(make_series(30, seed=123), make_series(30, seed=123))


def test_lags_predict_only_the_next_observation() -> None:
    x, y = make_supervised(np.arange(6, dtype=float), n_lags=3)
    np.testing.assert_array_equal(x, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(y, [3, 4, 5])


def test_end_to_end_demo_produces_finite_scores() -> None:
    scores = evaluate_demo(n_samples=100, n_lags=5, test_size=0.2, seed=42)
    assert scores["train_samples"] == 76
    assert scores["test_samples"] == 19
    for name in ("model_mae", "model_rmse", "baseline_mae", "baseline_rmse"):
        assert np.isfinite(scores[name])
        assert scores[name] >= 0


def test_invalid_window_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_supervised(np.arange(4), n_lags=4)


def test_invalid_test_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_demo(test_size=1.0)
