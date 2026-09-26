"""Metrics and backtesting."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import LOOKBACK
from .models import Forecaster

# Fixed colour per model so every chart uses the same legend.
COLORS = {
    "naive_24h": "#9e9e9e", "naive_168h": "#cfcfcf", "linear": "#8c6d31",
    "lightgbm": "#1f77b4", "hybrid": "#d62728", "ensemble": "#2ca02c",
}


def metrics(actual, predicted) -> dict[str, float]:
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    err = a - p
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(100 * np.mean(np.abs(err / a))),
        "R2": float(1 - np.sum(err**2) / np.sum((a - a.mean()) ** 2)),
        "bias": float(np.mean(p - a)),
    }


def day_ahead_origins(
    series: pd.Series, start: pd.Timestamp, horizon: int, origin_hour: int = 0
) -> pd.DatetimeIndex:
    """One origin per day at ``origin_hour`` whose whole horizon lies inside the series."""
    idx = series.index
    first_allowed = idx[LOOKBACK]
    last_allowed = idx[-1] - pd.Timedelta(hours=horizon - 1)
    mask = (idx >= max(pd.Timestamp(start), first_allowed)) & (idx <= last_allowed) & (idx.hour == origin_hour)
    return idx[mask]


@dataclass
class BacktestResult:
    name: str
    one_step: pd.Series                 # 1-hour-ahead predictions over the test period
    day_ahead: pd.DataFrame             # rows = origins, cols = horizon step (1..H)
    actual_day_ahead: pd.DataFrame
    summary: dict[str, float] = field(default_factory=dict)


def backtest(
    model: Forecaster,
    series: pd.Series,
    test_start: pd.Timestamp,
    horizon: int = 24,
    origin_hour: int = 0,
) -> BacktestResult:
    """Evaluate on everything from ``test_start`` on. ``series`` must include prior history."""
    one = model.predict_one_step(series, test_start)
    origins = day_ahead_origins(series, test_start, horizon, origin_hour)
    pred = model.forecast_from_origins(series, origins, horizon)
    y = series.to_numpy(float)
    pos = series.index.get_indexer(origins)
    actual = np.stack([y[p : p + horizon] for p in pos])
    cols = pd.RangeIndex(1, horizon + 1, name="h")
    res = BacktestResult(
        name=model.name,
        one_step=one,
        day_ahead=pd.DataFrame(pred, index=origins, columns=cols),
        actual_day_ahead=pd.DataFrame(actual, index=origins, columns=cols),
    )
    res.summary = {
        **{f"1h_{k}": v for k, v in metrics(series.loc[one.index], one).items()},
        **{f"day_ahead_{k}": v for k, v in metrics(actual, pred).items()},
        "n_one_step": int(len(one)),
        "n_day_ahead_origins": int(len(origins)),
    }
    return res


def summary_table(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({r.name: r.summary for r in results}).T


def mae_by_horizon(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame(
        {r.name: (r.actual_day_ahead - r.day_ahead).abs().mean(axis=0) for r in results}
    )


def mae_by_month(results: list[BacktestResult]) -> pd.DataFrame:
    out = {}
    for r in results:
        err = (r.actual_day_ahead - r.day_ahead).abs().mean(axis=1)
        out[r.name] = err.groupby(err.index.to_period("M")).mean()
    return pd.DataFrame(out)


def backtest_frame(results: list[BacktestResult]) -> pd.DataFrame:
    """Hourly table of actuals and each model's day-ahead forecast (one row per forecast hour)."""
    first = results[0]
    horizon = first.day_ahead.shape[1]
    times = pd.DatetimeIndex(
        (first.day_ahead.index.values[:, None] + np.arange(horizon) * np.timedelta64(1, "h")).ravel(),
        name="timestamp",
    )
    out = pd.DataFrame({"actual": first.actual_day_ahead.to_numpy().ravel()}, index=times)
    for r in results:
        out[r.name] = r.day_ahead.to_numpy().ravel()
    return out


def plot_reports(results: list[BacktestResult], series: pd.Series, out_dir) -> list[str]:
    """Save diagnostic PNGs; returns their paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    models = [r for r in results if not r.name.startswith("naive")]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, col in mae_by_horizon(results).items():
        ax.plot(col.index, col.values, marker="o", ms=3, label=name, color=COLORS.get(name))
    ax.set(xlabel="Hours ahead", ylabel="MAE (MW)", title="Day-ahead MAE by forecast horizon (test set)")
    ax.grid(alpha=0.3); ax.legend()
    paths.append(_save(fig, out_dir / "mae_by_horizon.png"))

    # A week from the middle of the test period.
    ref = models[-1]
    week_start = ref.day_ahead.index[len(ref.day_ahead) // 2]
    week_origins = ref.day_ahead.index[(ref.day_ahead.index >= week_start)][:7]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    times = pd.DatetimeIndex(np.concatenate(
        [pd.date_range(o, periods=ref.day_ahead.shape[1], freq="h") for o in week_origins]
    ))
    ax.plot(times, series.loc[times].values, color="black", lw=2, label="actual")
    for r in models:
        ax.plot(times, r.day_ahead.loc[week_origins].to_numpy().ravel(), lw=1.2, label=r.name,
                color=COLORS.get(r.name))
    ax.set(ylabel="MW", title="Day-ahead forecasts vs actual (one test week)")
    ax.grid(alpha=0.3); ax.legend()
    paths.append(_save(fig, out_dir / "test_week.png"))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    by_month = mae_by_month(models)
    by_month.index = by_month.index.astype(str)
    for name, col in by_month.items():
        ax.plot(col.index, col.values, marker="o", ms=3, label=name, color=COLORS.get(name))
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.set(xlabel="Month", ylabel="Day-ahead MAE (MW)", title="Day-ahead MAE by month (test set)")
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out_dir / "mae_by_month.png"))
    return paths


def _save(fig, path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)
