"""Feature engineering shared by training and inference.

Every feature for target hour ``t`` is computed from a *window* of the ``LOOKBACK``
values strictly before ``t`` plus the calendar of ``t``. Training and recursive
forecasting both go through :func:`window_features`, so the model sees exactly the
same feature definitions in both modes (no train/serve skew).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy import sparse

LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48, 72, 168)
LOOKBACK = 168  # hours of history needed to build one feature row
TREND_EPOCH = pd.Timestamp("2000-01-01")

CALENDAR_FEATURES = [
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "weekofyear",
    "quarter",
    "is_weekend",
    "is_holiday",
    "is_holiday_adjacent",
]
WINDOW_FEATURES = [f"lag_{lag}" for lag in LAGS] + [
    "roll_mean_24",
    "roll_std_24",
    "roll_min_24",
    "roll_max_24",
    "roll_mean_168",
]
BASE_FEATURE = "base"


def feature_columns(with_base: bool) -> list[str]:
    cols = CALENDAR_FEATURES + WINDOW_FEATURES
    return cols + [BASE_FEATURE] if with_base else cols


@lru_cache(maxsize=32)
def _holidays(start_year: int, end_year: int) -> pd.DatetimeIndex:
    return USFederalHolidayCalendar().holidays(
        pd.Timestamp(start_year - 1, 1, 1), pd.Timestamp(end_year + 1, 12, 31)
    )


def holiday_flags(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(is_holiday, is_day_before_or_after_holiday) for each timestamp."""
    days = index.normalize()
    hol = _holidays(int(index.year.min()), int(index.year.max()))
    is_hol = days.isin(hol)
    adjacent = (days - pd.Timedelta(days=1)).isin(hol) | (days + pd.Timedelta(days=1)).isin(hol)
    return is_hol.astype(np.int8), (adjacent & ~is_hol).astype(np.int8)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    index = pd.DatetimeIndex(index)
    is_hol, adjacent = holiday_flags(index)
    return pd.DataFrame(
        {
            "hour": index.hour,
            "dayofweek": index.dayofweek,
            "month": index.month,
            "dayofyear": index.dayofyear,
            "weekofyear": index.isocalendar().week.to_numpy(dtype=np.int32),
            "quarter": index.quarter,
            "is_weekend": (index.dayofweek >= 5).astype(np.int8),
            "is_holiday": is_hol,
            "is_holiday_adjacent": adjacent,
        },
        index=index,
    )


def window_features(windows: np.ndarray) -> dict[str, np.ndarray]:
    """Lag / rolling features from windows of shape (n, LOOKBACK).

    ``windows[:, -1]`` is the value one hour before the target, ``windows[:, -k]`` is lag k.
    """
    if windows.ndim != 2 or windows.shape[1] != LOOKBACK:
        raise ValueError(f"windows must have shape (n, {LOOKBACK}), got {windows.shape}")
    feats = {f"lag_{lag}": windows[:, -lag] for lag in LAGS}
    last24 = windows[:, -24:]
    feats["roll_mean_24"] = last24.mean(axis=1)
    feats["roll_std_24"] = last24.std(axis=1, ddof=1)
    feats["roll_min_24"] = last24.min(axis=1)
    feats["roll_max_24"] = last24.max(axis=1)
    feats["roll_mean_168"] = windows.mean(axis=1)
    return feats


def assemble(
    target_index: pd.DatetimeIndex,
    windows: np.ndarray,
    base: np.ndarray | None = None,
) -> pd.DataFrame:
    """Full feature frame for the given target timestamps and their history windows."""
    X = calendar_features(target_index).reset_index(drop=True)
    for name, values in window_features(windows).items():
        X[name] = values
    if base is not None:
        X[BASE_FEATURE] = base
    return X[feature_columns(base is not None)]


def build_training_frame(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    positions: np.ndarray,
    base: np.ndarray | None = None,
    chunk_size: int = 20_000,
) -> pd.DataFrame:
    """Teacher-forced features for ``values[positions]`` (each needs LOOKBACK prior values)."""
    positions = np.asarray(positions)
    if positions.size and positions.min() < LOOKBACK:
        raise ValueError(f"positions must be >= LOOKBACK ({LOOKBACK})")
    views = sliding_window_view(values, LOOKBACK)  # views[i] == values[i : i + LOOKBACK]
    parts = []
    for start in range(0, len(positions), chunk_size):
        pos = positions[start : start + chunk_size]
        parts.append(
            assemble(index[pos], views[pos - LOOKBACK], None if base is None else base[pos])
        )
    return pd.concat(parts, ignore_index=True)


def base_design(
    index: pd.DatetimeIndex, fourier_order: int = 3, include_trend: bool = True
) -> sparse.csr_matrix:
    """Sparse deterministic design matrix for the seasonal base model.

    Columns: 168 hour-of-week dummies, annual Fourier terms (plain and interacted with
    hour of day, so the daily profile can change shape between summer and winter),
    holiday / holiday-adjacent dummies interacted with hour, and an optional linear trend.
    Nothing here depends on observed load, so it can be evaluated for any future time.
    """
    index = pd.DatetimeIndex(index)
    n = len(index)
    hour = index.hour.to_numpy()
    how = index.dayofweek.to_numpy() * 24 + hour
    doy = 2 * np.pi * (index.dayofyear.to_numpy() - 1) / 365.25
    is_hol, adjacent = holiday_flags(index)

    rows, cols, vals = [], [], []
    rows_all = np.arange(n)

    def add(col_idx: np.ndarray, value: np.ndarray, mask: np.ndarray | None = None) -> None:
        if mask is not None:
            rows.append(rows_all[mask]); cols.append(col_idx[mask]); vals.append(value[mask])
        else:
            rows.append(rows_all); cols.append(col_idx); vals.append(value)

    offset = 0
    add(offset + how, np.ones(n)); offset += 168
    for k in range(1, fourier_order + 1):
        for wave in (np.sin(k * doy), np.cos(k * doy)):
            add(np.full(n, offset), wave); offset += 1
            add(offset + hour, wave); offset += 24
    for flag in (is_hol.astype(bool), adjacent.astype(bool)):
        add(offset + hour, np.ones(n), flag); offset += 24
    if include_trend:
        years = ((index - TREND_EPOCH) / pd.Timedelta(days=365.25)).to_numpy(dtype=float)
        add(np.full(n, offset), years); offset += 1

    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, offset)
    )
