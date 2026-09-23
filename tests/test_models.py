import numpy as np
import pandas as pd
import pytest

from pjme_forecast.data import split_series
from pjme_forecast.evaluation import backtest, metrics
from pjme_forecast.features import LOOKBACK
from pjme_forecast.models import EnsembleForecaster, build_model
from pjme_forecast.pipeline import load_bundle, save_bundle

VAL, TEST = "2016-03-15", "2016-04-20"


@pytest.fixture(scope="module")
def fitted(series, cfg):
    tr, va, _ = split_series(series, VAL, TEST)
    out = {k: build_model(k, cfg).fit(tr, va).refit(pd.concat([tr, va]))
           for k in ("naive_24h", "linear", "lightgbm", "hybrid")}
    out["ensemble"] = EnsembleForecaster("ensemble", [out["lightgbm"], out["hybrid"]])
    return out


@pytest.mark.parametrize("kind", ["lightgbm", "hybrid", "linear"])
def test_first_recursive_step_equals_one_step_prediction(fitted, series, kind):
    """Recursive inference must build exactly the same features as training (no skew)."""
    m = fitted[kind]
    origins = series.index[[LOOKBACK + 10, 2000, 3000]]
    rec = m.forecast_from_origins(series, origins, horizon=6)[:, 0]
    one = m.predict_one_step(series, series.index[LOOKBACK]).loc[origins].to_numpy()
    np.testing.assert_allclose(rec, one, rtol=1e-9)


def test_forecast_output_shape_and_index(fitted, series):
    fc = fitted["hybrid"].forecast(series.iloc[-400:], horizon=24)
    assert len(fc) == 24
    assert fc.index[0] == series.index[-1] + pd.Timedelta(hours=1)
    assert (np.diff(fc.index.values) == np.timedelta64(1, "h")).all()
    assert {"forecast", "base", "residual"} <= set(fc.columns)
    np.testing.assert_allclose(fc["base"] + fc["residual"], fc["forecast"])
    assert np.isfinite(fc.to_numpy()).all()


def test_production_forecast_matches_backtest_path(fitted, series):
    """forecast(history) must equal the backtest forecast from the same origin, and never see the future."""
    cut = 3000
    fc = fitted["lightgbm"].forecast(series.iloc[:cut], 12)["forecast"].to_numpy()
    bt = fitted["lightgbm"].forecast_from_origins(series, series.index[[cut]], 12)[0]
    np.testing.assert_allclose(fc, bt, rtol=1e-9)


def test_models_beat_naive(fitted, series):
    res = {k: backtest(m, series, pd.Timestamp(TEST), 24).summary for k, m in fitted.items()}
    for k in ("lightgbm", "hybrid", "ensemble"):
        assert res[k]["1h_MAE"] < res["naive_24h"]["1h_MAE"]
        assert res[k]["day_ahead_MAE"] < res["naive_24h"]["day_ahead_MAE"]


def test_hybrid_absorbs_base_model_drift(fitted, series, monkeypatch):
    """If the seasonal base is off by a constant, residual lags must compensate.

    This is the original notebook's failure mode: its residual model used raw-load lags,
    so a base error of +c moved the final forecast by exactly +c.
    """
    hybrid = fitted["hybrid"]
    before = hybrid.predict_one_step(series, TEST)
    offset = 500.0
    original = hybrid.base.predict
    monkeypatch.setattr(hybrid.base, "predict", lambda idx: original(idx) + offset)
    after = hybrid.predict_one_step(series, TEST)
    moved = float((after - before).mean())
    assert abs(moved) < 0.7 * offset  # notebook-style design: moved == offset


def test_save_load_roundtrip(fitted, series, tmp_path):
    path = save_bundle(fitted["hybrid"], {"model_kind": "hybrid"}, tmp_path / "m.joblib")
    model, meta = load_bundle(path)
    assert meta["model_kind"] == "hybrid"
    assert (tmp_path / "m.json").exists()
    pd.testing.assert_frame_equal(model.forecast(series, 24), fitted["hybrid"].forecast(series, 24))


def test_unfitted_model_raises(cfg, series):
    with pytest.raises(RuntimeError):
        build_model("hybrid", cfg).forecast(series, 24)
