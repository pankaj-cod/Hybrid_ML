"""Demo dashboard: an HTML page at ``/dashboard`` plus the JSON endpoints it reads.

Accuracy views use the *out-of-sample* test predictions written by ``pjme train``
(``artifacts/reports/backtest_day_ahead.csv``), never the production model, which has
seen the test period during its final refit.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).parent / "static"
MIN_DAYS_PER_MONTH = 20
router = APIRouter()


def _reports_dir() -> Path:
    return Path(os.environ.get("PJME_REPORTS_DIR", "artifacts/reports"))


@lru_cache(maxsize=1)
def _report() -> dict:
    path = _reports_dir() / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path} not found; run `pjme train` first")
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def _backtest() -> pd.DataFrame:
    path = _reports_dir() / "backtest_day_ahead.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path} not found; run `pjme train` first")
    return pd.read_csv(path, index_col="timestamp", parse_dates=True)


def clear_caches() -> None:
    _report.cache_clear()
    _backtest.cache_clear()


@router.get("/dashboard", include_in_schema=False)
def dashboard_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html", media_type="text/html")


@router.get("/api/summary", tags=["dashboard"])
def summary() -> dict:
    from .service import get_model  # late import avoids a cycle

    _, meta = get_model()
    report = _report()
    days = _backtest().index.normalize().unique()
    # Drop months with only a few forecast days (e.g. the data ends on 3 Aug 2018).
    days_per_month = pd.Series(1, index=days).groupby(days.to_period("M")).sum()
    full_months = {str(p) for p, n in days_per_month.items() if n >= MIN_DAYS_PER_MONTH}
    return {
        "model_kind": meta["model_kind"],
        "trained_on": meta["trained_on"],
        "created_at": meta.get("created_at"),
        "split": report["split"],
        "metrics": report["metrics"],
        "mae_by_horizon": report["mae_by_horizon"],
        "mae_by_month": {k: v for k, v in report.get("mae_by_month", {}).items() if k in full_months},
        "backtest_days": [str(days[0].date()), str(days[-1].date())],
    }


@router.get("/api/backtest", tags=["dashboard"])
def backtest_day(day: str = Query(..., description="YYYY-MM-DD within the test period")) -> dict:
    df = _backtest()
    try:
        start = pd.Timestamp(day).normalize()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date {day!r}") from exc
    rows = df[(df.index >= start) & (df.index < start + pd.Timedelta(days=1))]
    if rows.empty:
        first, last = df.index[0].date(), df.index[-1].date()
        raise HTTPException(status_code=404, detail=f"No test forecast for {day}; pick {first} to {last}")
    models = [c for c in rows.columns if c != "actual"]
    actual = rows["actual"].to_numpy()
    day_metrics = {
        m: {
            "MAE": float(np.mean(np.abs(actual - rows[m].to_numpy()))),
            "MAPE": float(100 * np.mean(np.abs((actual - rows[m].to_numpy()) / actual))),
            "peak_error": float(rows[m].max() - actual.max()),
        }
        for m in models
    }
    return {
        "day": str(start.date()),
        "timestamps": [t.isoformat() for t in rows.index],
        "series": {c: [round(float(v), 1) for v in rows[c]] for c in rows.columns},
        "metrics": day_metrics,
    }


@router.get("/api/backtest/daily", tags=["dashboard"])
def backtest_daily() -> dict:
    """Mean absolute error per test day, for every model."""
    df = _backtest()
    err = df.drop(columns="actual").sub(df["actual"], axis=0).abs()
    daily = err.groupby(err.index.normalize()).mean()
    return {
        "days": [str(d.date()) for d in daily.index],
        "mae": {c: [round(float(v), 1) for v in daily[c]] for c in daily.columns},
    }


@router.get("/api/forecast", tags=["dashboard"])
def live_forecast(horizon: int = Query(24, ge=1, le=168), context_hours: int = Query(72, ge=24, le=336)) -> dict:
    from .service import get_demo_history, get_model

    history = get_demo_history()
    if history is None:
        raise HTTPException(status_code=404, detail="PJME_HISTORY_PATH is not configured")
    model, _ = get_model()
    fc = model.forecast(history, horizon=horizon)
    recent = history.iloc[-context_hours:]
    out = {
        "history": {"timestamps": [t.isoformat() for t in recent.index],
                    "values": [round(float(v), 1) for v in recent]},
        "timestamps": [t.isoformat() for t in fc.index],
    }
    for col in fc.columns:
        out[col] = [round(float(v), 1) for v in fc[col]]
    return out
