"""Forecasting models.

All forecasters share one interface:

* ``fit(train, val=None)``   – fit (LightGBM uses ``val`` for early stopping)
* ``refit(series)``          – refit on more data, keeping the chosen hyper-parameters
* ``predict_one_step(series, start)``        – 1-hour-ahead predictions (actuals as lags)
* ``forecast_from_origins(series, origins, horizon)`` – recursive multi-step backtest
* ``forecast(history, horizon)``             – production forecast after the last observation

The hybrid model is a :class:`RecursiveForecaster` with a :class:`SeasonalBase`.
It forecasts the residual ``z = y - base(t)`` and *every lag feature is computed on
that residual series*. That is the fix for the original notebook, where the residual
model was trained on residual targets but raw-load lags, so any drift in the base
model went straight into the final prediction.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from .data import validate_history
from .features import LOOKBACK, assemble, base_design, build_training_frame, feature_columns

log = logging.getLogger(__name__)
HOUR = pd.Timedelta(hours=1)


class SeasonalBase:
    """Ridge regression on deterministic calendar terms (weekly profile, annual cycle, holidays, trend).

    Annual Fourier terms and the linear trend are only identifiable from multi-year data;
    fitted on a shorter window they extrapolate wildly, so they are switched off below
    ``MIN_YEARS_FOR_ANNUAL`` years of training data.
    """

    MIN_YEARS_FOR_ANNUAL = 2.0

    def __init__(self, fourier_order: int = 3, ridge_alpha: float = 1.0, include_trend: bool = True):
        self.fourier_order = fourier_order
        self.ridge_alpha = ridge_alpha
        self.include_trend = include_trend
        self.model_: Ridge | None = None
        self.fourier_order_ = fourier_order
        self.include_trend_ = include_trend

    def _design(self, index: pd.DatetimeIndex):
        return base_design(index, self.fourier_order_, self.include_trend_)

    def fit(self, series: pd.Series) -> "SeasonalBase":
        years = (series.index[-1] - series.index[0]) / pd.Timedelta(days=365.25)
        long_enough = years >= self.MIN_YEARS_FOR_ANNUAL
        self.fourier_order_ = self.fourier_order if long_enough else 0
        self.include_trend_ = self.include_trend and long_enough
        if not long_enough and (self.fourier_order or self.include_trend):
            log.warning(
                "Only %.1f years of data: base model drops annual/trend terms (needs %.0f)",
                years, self.MIN_YEARS_FOR_ANNUAL,
            )
        self.model_ = Ridge(alpha=self.ridge_alpha).fit(self._design(series.index), series.to_numpy(float))
        return self

    def predict(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("SeasonalBase is not fitted")
        return self.model_.predict(self._design(pd.DatetimeIndex(index)))


class Forecaster:
    name = "forecaster"

    def fit(self, train: pd.Series, val: pd.Series | None = None) -> "Forecaster":
        return self

    def refit(self, series: pd.Series) -> "Forecaster":
        return self.fit(series)

    def predict_one_step(self, series: pd.Series, start: pd.Timestamp) -> pd.Series:
        raise NotImplementedError

    def forecast_from_origins(
        self, series: pd.Series, origins: pd.DatetimeIndex, horizon: int
    ) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, history: pd.Series, horizon: int = 24) -> pd.DataFrame:
        validate_history(history, LOOKBACK)
        origin = pd.DatetimeIndex([history.index[-1] + HOUR])
        values = self.forecast_from_origins(_extend(history, horizon), origin, horizon)[0]
        index = pd.date_range(origin[0], periods=horizon, freq="h", name="timestamp")
        return pd.DataFrame({"forecast": values}, index=index)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__}


class SeasonalNaive(Forecaster):
    """y(t) = y(t - period); the standard sanity-check baseline."""

    def __init__(self, period: int = 24):
        self.period = period
        self.name = f"naive_{period}h"

    def predict_one_step(self, series, start):
        pos = _positions_from(series.index, start, self.period)
        return pd.Series(series.to_numpy(float)[pos - self.period], index=series.index[pos])

    def forecast_from_origins(self, series, origins, horizon):
        y = series.to_numpy(float)
        p = _origin_positions(series.index, origins, self.period)
        h = np.arange(horizon)
        lag = self.period * np.ceil((h + 1) / self.period).astype(int)  # stay within observed data
        return y[p[:, None] + h[None, :] - lag[None, :]]


class RecursiveForecaster(Forecaster):
    """One-step-ahead regressor applied recursively for multi-step forecasts.

    ``base=None``  -> direct model on the raw load (the notebook's "tree model").
    ``base=SeasonalBase(...)`` -> hybrid: seasonal base + regressor on the residual.
    """

    def __init__(
        self,
        name: str,
        regressor: str = "lightgbm",
        lgb_params: dict | None = None,
        early_stopping_rounds: int = 200,
        base: SeasonalBase | None = None,
    ):
        if regressor not in {"lightgbm", "linear"}:
            raise ValueError(f"Unknown regressor {regressor!r}")
        self.name = name
        self.regressor = regressor
        self.lgb_params = dict(lgb_params or {})
        self.early_stopping_rounds = early_stopping_rounds
        self.base = base
        self.model_ = None
        self.n_estimators_: int | None = None
        self.trained_range_: tuple[str, str] | None = None

    @property
    def is_hybrid(self) -> bool:
        return self.base is not None

    # ------------------------------------------------------------------ training
    def _to_target_space(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray | None]:
        y = series.to_numpy(float)
        if self.base is None:
            return y, None
        b = self.base.predict(series.index)
        return y - b, b

    def _new_regressor(self, n_estimators: int | None = None):
        if self.regressor == "linear":
            return LinearRegression()
        params = {"verbose": -1, **self.lgb_params}
        if n_estimators is not None:
            params["n_estimators"] = n_estimators
        return lgb.LGBMRegressor(**params)

    def fit(self, train: pd.Series, val: pd.Series | None = None) -> "RecursiveForecaster":
        if self.base is not None:
            self.base.fit(train)
        if val is not None and len(val):
            if val.index[0] != train.index[-1] + HOUR:
                raise ValueError("Validation series must start right after the training series")
            full = pd.concat([train, val])
        else:
            full, val = train, None
        z, b = self._to_target_space(full)
        tr_pos = np.arange(LOOKBACK, len(train))
        X_tr = build_training_frame(z, full.index, tr_pos, b)
        model = self._new_regressor()
        if self.regressor == "lightgbm" and val is not None:
            va_pos = np.arange(len(train), len(full))
            X_va = build_training_frame(z, full.index, va_pos, b)
            _fit_lgbm_early_stopping(model, X_tr, z[tr_pos], X_va, z[va_pos], self.early_stopping_rounds)
            self.n_estimators_ = int(model.best_iteration_ or model.n_estimators)
            log.info("%s: early stopping chose %d trees", self.name, self.n_estimators_)
        else:
            model.fit(X_tr, z[tr_pos])
            if self.regressor == "lightgbm":
                self.n_estimators_ = int(model.n_estimators)
        self.model_ = model
        self.trained_range_ = (str(train.index[0]), str(train.index[-1]))
        return self

    def refit(self, series: pd.Series) -> "RecursiveForecaster":
        """Refit on ``series`` with the tree count found by early stopping."""
        if self.base is not None:
            self.base.fit(series)
        z, b = self._to_target_space(series)
        pos = np.arange(LOOKBACK, len(series))
        model = self._new_regressor(self.n_estimators_)
        model.fit(build_training_frame(z, series.index, pos, b), z[pos])
        self.model_ = model
        self.trained_range_ = (str(series.index[0]), str(series.index[-1]))
        return self

    # ---------------------------------------------------------------- inference
    def _check_fitted(self) -> None:
        if self.model_ is None:
            raise RuntimeError(f"{self.name} is not fitted")

    def predict_one_step(self, series, start):
        self._check_fitted()
        pos = _positions_from(series.index, start, LOOKBACK)
        z, b = self._to_target_space(series)
        pred = self.model_.predict(build_training_frame(z, series.index, pos, b))
        if b is not None:
            pred = pred + b[pos]
        return pd.Series(pred, index=series.index[pos])

    def forecast_from_origins(self, series, origins, horizon):
        """Recursive forecasts for many origins at once (vectorised over origins).

        ``origins`` are the first forecast timestamps; only data before each origin is used.
        """
        self._check_fitted()
        p = _origin_positions(series.index, origins, LOOKBACK)
        z, _ = self._to_target_space(series)
        windows = np.stack([z[i - LOOKBACK : i] for i in p])
        return self._recursive(windows, pd.DatetimeIndex(origins), horizon)

    def _recursive(self, windows: np.ndarray, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
        n = len(origins)
        steps = np.arange(horizon) * np.timedelta64(1, "h")
        target_times = pd.DatetimeIndex((origins.values[:, None] + steps[None, :]).ravel())
        base = None
        if self.base is not None:
            base = self.base.predict(target_times).reshape(n, horizon)
        buf = np.concatenate([windows, np.empty((n, horizon))], axis=1)
        for h in range(horizon):
            X = assemble(
                target_times[h::horizon],
                buf[:, h : h + LOOKBACK],
                None if base is None else base[:, h],
            )
            buf[:, LOOKBACK + h] = self.model_.predict(X)
        pred = buf[:, LOOKBACK:]
        return pred + base if base is not None else pred

    def forecast(self, history, horizon=24):
        """Forecast ``horizon`` hours after the last observation in ``history``."""
        self._check_fitted()
        validate_history(history, LOOKBACK)
        origin = history.index[-1] + HOUR
        index = pd.date_range(origin, periods=horizon, freq="h", name="timestamp")
        z, _ = self._to_target_space(history.iloc[-LOOKBACK:])
        values = self._recursive(z[None, :], pd.DatetimeIndex([origin]), horizon)[0]
        out = pd.DataFrame({"forecast": values}, index=index)
        if self.base is not None:
            out["base"] = self.base.predict(index)
            out["residual"] = out["forecast"] - out["base"]
        return out

    def describe(self):
        return {
            **super().describe(),
            "regressor": self.regressor,
            "hybrid": self.is_hybrid,
            "n_estimators": self.n_estimators_,
            "features": feature_columns(self.is_hybrid),
            "trained_range": self.trained_range_,
            "lightgbm_params": self.lgb_params if self.regressor == "lightgbm" else None,
            "base_model": None
            if self.base is None
            else {
                "fourier_order": self.base.fourier_order_,
                "ridge_alpha": self.base.ridge_alpha,
                "include_trend": self.base.include_trend_,
            },
        }


class EnsembleForecaster(Forecaster):
    """Weighted average of already-configured forecasters."""

    def __init__(self, name: str, members: list[Forecaster], weights: list[float] | None = None):
        self.name = name
        self.members = members
        w = np.asarray(weights if weights is not None else [1.0] * len(members), dtype=float)
        self.weights = w / w.sum()

    def fit(self, train, val=None):
        for m in self.members:
            m.fit(train, val)
        return self

    def refit(self, series):
        for m in self.members:
            m.refit(series)
        return self

    def predict_one_step(self, series, start):
        preds = [m.predict_one_step(series, start) for m in self.members]
        return sum(w * p for w, p in zip(self.weights, preds))

    def forecast_from_origins(self, series, origins, horizon):
        preds = [m.forecast_from_origins(series, origins, horizon) for m in self.members]
        return sum(w * p for w, p in zip(self.weights, preds))

    def forecast(self, history, horizon=24):
        parts = [m.forecast(history, horizon)["forecast"] for m in self.members]
        out = pd.DataFrame({"forecast": sum(w * p for w, p in zip(self.weights, parts))})
        for m, p in zip(self.members, parts):
            out[f"forecast_{m.name}"] = p
        return out

    def describe(self):
        return {
            **super().describe(),
            "weights": self.weights.tolist(),
            "members": [m.describe() for m in self.members],
        }


# ---------------------------------------------------------------------- factory
def build_model(kind: str, cfg: dict) -> Forecaster:
    """Create an unfitted model from the config. ``kind`` is one of MODEL_KINDS."""
    lgb_cfg = dict(cfg["lightgbm"])
    rounds = lgb_cfg.pop("early_stopping_rounds", 200)
    base_cfg = cfg["base_model"]
    if kind == "naive_24h":
        return SeasonalNaive(24)
    if kind == "naive_168h":
        return SeasonalNaive(168)
    if kind == "linear":
        return RecursiveForecaster("linear", regressor="linear")
    if kind == "lightgbm":
        return RecursiveForecaster("lightgbm", lgb_params=lgb_cfg, early_stopping_rounds=rounds)
    if kind == "hybrid":
        return RecursiveForecaster(
            "hybrid", lgb_params=lgb_cfg, early_stopping_rounds=rounds, base=SeasonalBase(**base_cfg)
        )
    if kind == "ensemble":
        return EnsembleForecaster("ensemble", [build_model("lightgbm", cfg), build_model("hybrid", cfg)])
    raise ValueError(f"Unknown model kind {kind!r}; expected one of {MODEL_KINDS}")


MODEL_KINDS = ("naive_24h", "naive_168h", "linear", "lightgbm", "hybrid", "ensemble")


# ---------------------------------------------------------------------- helpers
def _fit_lgbm_early_stopping(model, X_tr, y_tr, X_va, y_va, rounds: int) -> None:
    callbacks = [lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(0)]
    if "eval_X" in inspect.signature(model.fit).parameters:  # lightgbm >= 4.7
        model.fit(X_tr, y_tr, eval_X=(X_va,), eval_y=(y_va,), callbacks=callbacks)
    else:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=callbacks)


def _positions_from(index: pd.DatetimeIndex, start, min_pos: int) -> np.ndarray:
    first = max(int(index.searchsorted(pd.Timestamp(start))), min_pos)
    if first >= len(index):
        raise ValueError(f"No predictable timestamps at or after {start}")
    return np.arange(first, len(index))


def _origin_positions(index: pd.DatetimeIndex, origins, min_pos: int) -> np.ndarray:
    p = index.get_indexer(pd.DatetimeIndex(origins))
    if (p < 0).any():
        raise ValueError("Some origins are not in the series index")
    if (p < min_pos).any():
        raise ValueError(f"Each origin needs at least {min_pos} hours of history before it")
    return p


def _extend(history: pd.Series, horizon: int) -> pd.Series:
    """Append NaN placeholders so origin timestamps exist in the index."""
    future = pd.date_range(history.index[-1] + HOUR, periods=horizon, freq="h")
    return pd.concat([history, pd.Series(np.nan, index=future)])
