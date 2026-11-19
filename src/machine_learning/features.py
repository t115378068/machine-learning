"""Feature engineering helpers."""

import pandas as pd


def select_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return selected input columns without modifying the original frame."""
    return frame.loc[:, columns].copy()
