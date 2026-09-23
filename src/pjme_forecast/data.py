"""Loading, cleaning and validating the hourly load series."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Physically plausible bounds for PJM East hourly load (MW). Values outside are data errors.
MIN_PLAUSIBLE_MW = 1_000.0
MAX_PLAUSIBLE_MW = 100_000.0


class DataValidationError(ValueError):
    """Raised when an input series cannot be safely used for training or forecasting."""


def load_series(
    path: str | Path,
    timestamp_col: str = "Datetime",
    target_col: str = "PJME_MW",
) -> pd.Series:
    """Read the raw CSV and return a raw (uncleaned) float series indexed by timestamp."""
    df = pd.read_csv(path)
    missing = {timestamp_col, target_col} - set(df.columns)
    if missing:
        raise DataValidationError(f"{path}: missing column(s) {sorted(missing)}")
    index = pd.to_datetime(df[timestamp_col], errors="raise")
    values = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    return pd.Series(values, index=pd.DatetimeIndex(index), name=target_col)


def clean_series(series: pd.Series, max_interpolate_hours: int = 6) -> pd.Series:
    """Return a strictly hourly, gap-free, sorted series.

    * Duplicate timestamps (DST fall-back hour is reported twice) are averaged.
    * Missing hours (DST spring-forward, sporadic outages) are time-interpolated,
      but only for gaps up to ``max_interpolate_hours``; longer gaps raise.
    * Implausible values are treated as missing before interpolation.
    """
    if series.empty:
        raise DataValidationError("Series is empty")
    s = series.astype(float).copy()
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)

    bad = (s < MIN_PLAUSIBLE_MW) | (s > MAX_PLAUSIBLE_MW)
    if bad.any():
        log.warning("Treating %d implausible value(s) as missing", int(bad.sum()))
        s[bad] = np.nan

    n_dup = int(s.index.duplicated().sum())
    s = s.groupby(level=0).mean().sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    n_missing = len(full) - len(s)
    s = s.reindex(full)

    gap_len = _longest_nan_run(s.to_numpy())
    if gap_len > max_interpolate_hours:
        raise DataValidationError(
            f"Found a gap of {gap_len} consecutive missing hours "
            f"(limit {max_interpolate_hours}); refusing to interpolate"
        )
    s = s.interpolate(method="time", limit_direction="both")
    s.name = series.name
    log.info(
        "Cleaned series: %d rows, %d duplicate(s) averaged, %d missing hour(s) filled",
        len(s), n_dup, n_missing + int(bad.sum()),
    )
    return s


def validate_history(series: pd.Series, min_length: int) -> None:
    """Checks a cleaned series must pass before it is used as forecast input."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise DataValidationError("History must have a DatetimeIndex")
    if len(series) < min_length:
        raise DataValidationError(
            f"Need at least {min_length} hours of history, got {len(series)}"
        )
    if series.isna().any():
        raise DataValidationError("History contains missing values after cleaning")
    steps = np.diff(series.index.values)
    if len(steps) and not np.all(steps == np.timedelta64(1, "h")):
        raise DataValidationError("History is not a contiguous hourly series")


def load_clean_series(cfg: dict) -> pd.Series:
    d = cfg["data"]
    raw = load_series(d["path"], d["timestamp_col"], d["target_col"])
    return clean_series(raw, d["max_interpolate_hours"])


def split_series(
    series: pd.Series, val_start: str, test_start: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Chronological train / validation / test split."""
    val_ts, test_ts = pd.Timestamp(val_start), pd.Timestamp(test_start)
    if not series.index[0] < val_ts < test_ts < series.index[-1]:
        raise DataValidationError(
            f"Split dates must satisfy start < val_start < test_start < end "
            f"({series.index[0]} .. {series.index[-1]})"
        )
    train = series[series.index < val_ts]
    val = series[(series.index >= val_ts) & (series.index < test_ts)]
    test = series[series.index >= test_ts]
    return train, val, test


def _longest_nan_run(values: np.ndarray) -> int:
    isnan = np.isnan(values).astype(np.int8)
    if not isnan.any():
        return 0
    edges = np.diff(np.concatenate([[0], isnan, [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max())
