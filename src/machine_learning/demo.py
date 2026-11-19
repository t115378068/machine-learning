"""Run a reproducible next-step time-series forecasting demo.

This example intentionally uses synthetic data, so no API keys, downloads,
or private datasets are needed. All evaluation uses a chronological split.

Usage:
    python -m machine_learning.demo
    python -m machine_learning.demo --samples 600 --lags 10 --test-size 0.2
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_series(n_samples: int = 500, seed: int = 42) -> np.ndarray:
    """Produce a deterministic, noisy, trend-and-seasonality example series."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=float)
    return 100.0 + 0.03 * t + 5.0 * np.sin(t / 8.0) + rng.normal(0, 0.3, n_samples)


def make_supervised(series: np.ndarray, n_lags: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Use observations through time t to predict the value at t+1.

    Each feature row contains [y_(t-n_lags+1), ..., y_t] and its target is
    y_(t+1). Nothing from the target or future enters its feature row.
    """
    values = np.asarray(series, dtype=float).reshape(-1)

    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    if len(values) <= n_lags:
        raise ValueError("series must contain more observations than n_lags")
    if not np.isfinite(values).all():
        raise ValueError("series must contain only finite values")

    features = np.stack(
        [values[i : i + n_lags] for i in range(len(values) - n_lags)]
    )
    targets = values[n_lags:]
    return features, targets


def evaluate_demo(
    n_samples: int = 500,
    n_lags: int = 10,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, float | int]:
    """Train on earlier observations and evaluate on later observations."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    series = make_series(n_samples, seed)
    x, y = make_supervised(series, n_lags)
    cutoff = int(len(x) * (1 - test_size))

    if not 1 <= cutoff < len(x):
        raise ValueError("train/test split must leave at least one sample in each set")

    x_train, x_test = x[:cutoff], x[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    # The last observed value is a useful baseline for next-step forecasts.
    persistence = x_test[:, -1]

    return {
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "model_mae": float(mean_absolute_error(y_test, predictions)),
        "model_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "baseline_mae": float(mean_absolute_error(y_test, persistence)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, persistence))),
    }


def main() -> None:
    """Parse CLI arguments and print an easy-to-read evaluation summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--lags", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        results = evaluate_demo(args.samples, args.lags, args.test_size, args.seed)
    except ValueError as exc:
        parser.error(str(exc))

    print("Synthetic next-step forecasting demo")
    print("------------------------------------")
    print(f"Training rows: {results['train_samples']}")
    print(f"Testing rows:  {results['test_samples']}")
    print(f"Random Forest MAE:  {results['model_mae']:.4f}")
    print(f"Random Forest RMSE: {results['model_rmse']:.4f}")
    print(f"Last-value MAE:     {results['baseline_mae']:.4f}")
    print(f"Last-value RMSE:    {results['baseline_rmse']:.4f}")
    print("Lower error is better. This synthetic example is not a stock forecast.")


if __name__ == "__main__":
    main()
