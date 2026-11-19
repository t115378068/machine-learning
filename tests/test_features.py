"""Small smoke test for the project package."""

import pandas as pd

from machine_learning.features import select_features


def test_select_features_returns_requested_columns() -> None:
    frame = pd.DataFrame({"close": [10, 11], "volume": [100, 200]})
    selected = select_features(frame, ["close"])
    assert selected.columns.tolist() == ["close"]
    assert selected["close"].tolist() == [10, 11]
