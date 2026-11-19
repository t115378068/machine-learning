"""Dataset loading utilities."""

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file with pandas."""
    return pd.read_csv(path)
