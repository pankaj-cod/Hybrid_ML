import numpy as np
import pandas as pd
import pytest

from pjme_forecast.data import DataValidationError, clean_series, split_series, validate_history


def _raw(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h"), dtype=float)


def test_duplicates_are_averaged_and_output_is_sorted():
    s = _raw([30000, 31000, 32000])
    dup = pd.Series([33000.0], index=[s.index[1]])
    out = clean_series(pd.concat([s, dup]).iloc[::-1])
    assert out.index.is_monotonic_increasing
    assert len(out) == 3
    assert out.iloc[1] == 32000.0  # mean of 31000 and 33000


def test_short_gaps_are_interpolated():
    s = _raw([30000, 31000, 32000, 33000]).drop(pd.Timestamp("2020-01-01 01:00"))
    out = clean_series(s)
    assert len(out) == 4
    assert out.iloc[1] == pytest.approx(31000)


def test_long_gap_raises():
    s = _raw(np.full(30, 30000.0))
    s = s.drop(s.index[5:20])
    with pytest.raises(DataValidationError, match="gap"):
        clean_series(s, max_interpolate_hours=6)


def test_implausible_values_are_replaced():
    out = clean_series(_raw([30000, -5, 32000]))
    assert out.iloc[1] == pytest.approx(31000)


def test_validate_history_rejects_short_and_irregular():
    with pytest.raises(DataValidationError):
        validate_history(_raw(np.ones(10) * 3e4), min_length=168)
    s = _raw(np.ones(200) * 3e4).drop(pd.Timestamp("2020-01-02 00:00"))
    with pytest.raises(DataValidationError, match="contiguous"):
        validate_history(s, min_length=168)


def test_split_is_chronological_and_disjoint(series):
    tr, va, te = split_series(series, "2016-03-01", "2016-04-15")
    assert tr.index[-1] < va.index[0] and va.index[-1] < te.index[0]
    assert len(tr) + len(va) + len(te) == len(series)
