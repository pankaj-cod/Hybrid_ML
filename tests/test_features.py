import numpy as np
import pandas as pd

from pjme_forecast.features import (
    LOOKBACK,
    base_design,
    build_training_frame,
    calendar_features,
    feature_columns,
)


def test_features_use_only_past_values(series):
    """Changing y[t] or anything after it must not change the features for time t."""
    y = series.to_numpy().copy()
    p = np.array([500])
    before = build_training_frame(y, series.index, p)
    y[500:] += 1e6
    after = build_training_frame(y, series.index, p)
    pd.testing.assert_frame_equal(before, after)


def test_lag_values_match_definition(series):
    y = series.to_numpy()
    p = 1000
    X = build_training_frame(y, series.index, np.array([p]))
    for lag in (1, 24, 168):
        assert X[f"lag_{lag}"].iloc[0] == y[p - lag]
    assert np.isclose(X["roll_mean_24"].iloc[0], y[p - 24 : p].mean())
    assert list(X.columns) == feature_columns(False)


def test_holidays_flagged():
    idx = pd.DatetimeIndex(["2017-07-04 12:00", "2017-07-05 12:00", "2017-07-12 12:00",
                            "2017-11-24 09:00"])
    cal = calendar_features(idx)
    assert cal["is_holiday"].tolist() == [1, 0, 0, 0]
    assert cal["is_holiday_adjacent"].tolist() == [0, 1, 0, 1]  # day after July 4th / Thanksgiving


def test_base_design_is_deterministic_and_future_safe():
    idx = pd.date_range("2030-01-01", periods=48, freq="h")
    a, b = base_design(idx), base_design(idx)
    assert a.shape == b.shape and (a != b).nnz == 0
    assert np.isfinite(a.toarray()).all()
