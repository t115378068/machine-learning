# Source code

This package contains reusable code for machine-learning experiments.

## Try the sample forecasting program

Install dependencies from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m machine_learning.demo
```

Change sample count and lag window:

```bash
python -m machine_learning.demo --samples 600 --lags 10 --test-size 0.2
```

The demo generates **synthetic** time-series observations, creates lag features,
trains a Random Forest using only earlier observations, and reports test MAE
and RMSE alongside a last-observation baseline. No downloaded data is needed.

Run its tests with:

```bash
python -m pytest tests/test_demo.py
```

For a real experiment, replace the synthetic series with actual data and
preserve chronological evaluation; don't interpret the synthetic metrics as
evidence of financial forecasting performance.
