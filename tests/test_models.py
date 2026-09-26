import numpy as np
import pandas as pd
import pytest

from pjme_forecast.data import split_series
from pjme_forecast.evaluation import backtest, metrics
from pjme_forecast.features import LOOKBACK
from pjme_forecast.models import BiasCorrected, EnsembleForecaster, SeasonalNaive, build_model, estimate_bias_table
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


class _KnownBias(SeasonalNaive):
    """Perfect-foresight forecaster plus a planted error of 100 MW per step (+10 per origin hour)."""

    name = "known_bias"

    def forecast_from_origins(self, series, origins, horizon):
        y = series.to_numpy(float)
        pos = series.index.get_indexer(origins)
        truth = np.stack([y[p : p + horizon] for p in pos])
        return truth + 100.0 * np.arange(1, horizon + 1) + 10.0 * pd.DatetimeIndex(origins).hour.to_numpy()[:, None]


def test_bias_table_recovers_planted_error(series):
    tr, va, _ = split_series(series, VAL, TEST)
    table = estimate_bias_table(_KnownBias(), tr, va, horizon=24)
    assert table.shape == (24, 24)
    expected = 100.0 * np.arange(1, 25)[None, :] + 10.0 * np.arange(24)[:, None]
    np.testing.assert_allclose(table, expected)

    corrected = BiasCorrected(_KnownBias(), table)
    origins = series.index[[3000, 3013]]
    out = corrected.forecast_from_origins(series, origins, 24)
    truth = np.stack([series.to_numpy()[p : p + 24] for p in series.index.get_indexer(origins)])
    np.testing.assert_allclose(out, truth)
    # Steps beyond the table are left as-is rather than extrapolated.
    long = corrected.forecast_from_origins(series, origins[:1], 30)
    np.testing.assert_allclose(long[0, 24:], _KnownBias().forecast_from_origins(series, origins[:1], 30)[0, 24:])


def test_bias_corrected_model_fit_and_forecast(fitted, series, cfg):
    tr, va, _ = split_series(series, VAL, TEST)
    m = build_model("hybrid_bc", cfg).fit(tr, va)
    assert m.table.shape == (24, 24)
    fc = m.forecast(series.iloc[-400:], 24)
    np.testing.assert_allclose(fc["base"] + fc["residual"], fc["forecast"])
    raw = m.inner.forecast(series.iloc[-400:], 24)
    np.testing.assert_allclose(raw["forecast"] - fc["forecast"], m.table[fc.index[0].hour])
