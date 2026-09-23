"""HTTP API. Run with:

    PJME_MODEL_PATH=artifacts/model.joblib uvicorn pjme_forecast.service:app --port 8000

Environment variables:
    PJME_MODEL_PATH    saved model bundle (default: artifacts/model.joblib)
    PJME_HISTORY_PATH  optional CSV used by GET /forecast/latest (demo endpoint)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import __version__
from .data import DataValidationError, clean_series, load_series
from .features import LOOKBACK
from .pipeline import load_bundle

MAX_HORIZON = 168
log = logging.getLogger("uvicorn.error")  # shows up in uvicorn / Render logs


class Observation(BaseModel):
    timestamp: datetime
    value: float


class ForecastRequest(BaseModel):
    history: list[Observation] = Field(..., description=f"At least {LOOKBACK} recent hourly observations")
    horizon: int = Field(24, ge=1, le=MAX_HORIZON)


class ForecastPoint(BaseModel):
    timestamp: datetime
    forecast: float


class ForecastResponse(BaseModel):
    model_kind: str
    trained_on: list[str]
    forecast: list[ForecastPoint]


@lru_cache(maxsize=1)
def get_model():
    return load_bundle(os.environ.get("PJME_MODEL_PATH", "artifacts/model.joblib"))


@lru_cache(maxsize=1)
def get_demo_history() -> pd.Series | None:
    path = os.environ.get("PJME_HISTORY_PATH")
    if not path:
        return None
    return clean_series(load_series(path)).iloc[-(LOOKBACK + 24):]


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load at startup so a missing or broken model fails the deploy instead of the first request.
    _, meta = get_model()
    log.info("Loaded %s model trained on %s", meta["model_kind"], meta["trained_on"])
    yield


app = FastAPI(title="PJME load forecast", version=__version__, lifespan=lifespan)


@app.get("/")
def root() -> dict:
    return {"service": "PJME hourly load forecast", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict:
    _, meta = get_model()
    return {"status": "ok", "model_kind": meta["model_kind"], "trained_on": meta["trained_on"],
            "test_metrics": meta.get("test_metrics")}


def _run(model, meta, history: pd.Series, horizon: int) -> ForecastResponse:
    try:
        fc = model.forecast(history, horizon=horizon)
    except (DataValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ForecastResponse(
        model_kind=meta["model_kind"],
        trained_on=meta["trained_on"],
        forecast=[ForecastPoint(timestamp=t, forecast=float(v)) for t, v in fc["forecast"].items()],
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    model, meta = get_model()
    series = pd.Series(
        [o.value for o in req.history],
        index=pd.DatetimeIndex([o.timestamp for o in req.history]),
        name=meta.get("target", "PJME_MW"),
    )
    try:
        history = clean_series(series)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _run(model, meta, history, req.horizon)


@app.get("/forecast/latest", response_model=ForecastResponse)
def forecast_latest(horizon: int = Query(24, ge=1, le=MAX_HORIZON)) -> ForecastResponse:
    """Forecast after the end of the bundled history file (demo; needs PJME_HISTORY_PATH)."""
    history = get_demo_history()
    if history is None:
        raise HTTPException(status_code=404, detail="PJME_HISTORY_PATH is not configured")
    model, meta = get_model()
    return _run(model, meta, history, horizon)
