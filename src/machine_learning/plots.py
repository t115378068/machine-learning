"""Reusable plotting helpers."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_series(series: pd.Series):
    """Plot a numeric series and return the matplotlib axes."""
    _, ax = plt.subplots()
    series.plot(ax=ax)
    ax.set_xlabel("Observation")
    ax.set_ylabel(series.name or "Value")
    return ax
