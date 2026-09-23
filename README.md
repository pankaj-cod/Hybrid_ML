# PJME Hourly Load Forecasting: Hybrid Model

Forecasts hourly electricity demand for PJM East (`PJME_hourly.csv`, 2002–2018) with a
**hybrid model**: a seasonal linear base plus LightGBM on the residual. It is evaluated
against naive, linear, and direct LightGBM baselines on a held-out test period.

## Results (test set: 2017-01-01 → 2018-08-03, never used for fitting or tuning)

| Model | 1-hour-ahead MAE | Day-ahead MAE | Day-ahead MAPE | Day-ahead R² |
|---|---|---|---|---|
| Naive (same hour yesterday) | 2,300 MW | 2,300 MW | 7.32 % | 0.74 |
| Linear regression on lags | 386 MW | 2,033 MW | 6.48 % | 0.80 |
| LightGBM (direct "tree model") | 210 MW | 1,382 MW | 4.23 % | 0.886 |
| **Hybrid (seasonal base + LightGBM residual)** | **195 MW** | **1,329 MW** | **4.08 %** | **0.891** |
| Ensemble (mean of LightGBM + hybrid) | 187 MW | 1,311 MW | 4.01 % | 0.896 |

*Day-ahead* means one forecast every midnight for the next 24 hours, produced recursively
(each prediction feeds the next hour's lags), which is how the model runs in production.
The hybrid beats the direct tree model by about 7 % at 1 hour and 4 % day-ahead.
Full tables and plots are written to `artifacts/reports/` by `pjme train`.

## What was wrong with the original hybrid (`notebooks/00_original_exploration.ipynb`)

1. **The residual model worked in the wrong space.** It was trained to predict
   `y − trend`, but its lag features were lags of raw `y`, and it never saw the trend.
   So any error in the extrapolated trend passed 1:1 into the final forecast. That is
   why folds 1–2 had MAE of about 1,000–1,500 MW. **Fix:** lags and rolling statistics
   are computed on the *residual* series (and the base value is passed as a feature).
   If the base drifts by +c, the residual lags shift by −c and cancel it. Measured on the
   trained model, an injected +1,000 MW base error moves the forecast by only about 16 MW.
   Re-running the old design on the same split gives 1-hour MAE ≈ 500 MW, versus 195 for the fix.
2. **The multi-scale "trend" targets were features.** `trend_24` was
   `shift(1).rolling(24).mean()`, which is exactly the `rolling_mean_24` feature, so the
   "trend models" learned an identity mapping (MAE 26). The trend weights were then
   fitted on out-of-fold predictions but replaced by hard-coded 0.4/0.35/0.25 at inference.
3. **Train/serve skew.** Several forecast cells froze all lag features at the last
   observed value (`create_future_features`), so every hour of the 24-hour forecast saw
   the same `lag_1`.
4. **The "final test" compared against the wrong day.** The forecast for
   2018-08-03 01:00 → 08-04 00:00 was scored against actuals for 08-02 01:00 → 08-03 00:00.
   That, plus (3), produced the MAE ≈ 6,300 MW and R² < 0 in the last cells.
5. **Optimistic evaluation.** Hyper-parameters were searched on the same CV folds used
   to report scores, and all scores were one-step-ahead (lag_1 known). The recursive
   24-hour forecast that was actually being used was never backtested.

## Method

```
y(t) = base(t) + z(t)
base(t): Ridge on deterministic terms: 168 hour-of-week dummies, annual Fourier
         terms (k=1..3) × hour-of-day, US federal holidays (+ adjacent days) × hour,
         linear trend. Needs no load data, so it is known for any future hour.
z(t):    LightGBM on calendar features + lags {1,2,3,6,12,24,48,72,168} and rolling
         24h/168h stats *of z*, plus base(t).
```

* **Split:** train 2002–2014 · validation 2015–2016 (early stopping) · test 2017–2018-08.
  After early stopping, models are refit on train+val with the chosen tree count, then
  scored once on test. The production model is refit on all data.
* **One feature code path.** `features.window_features` builds features from the 168
  values before the target, both for training and for each recursive step. A test
  asserts that the first recursive step equals the teacher-forced prediction exactly.
* **Data cleaning:** DST duplicate hours are averaged, and missing hours are
  time-interpolated only for gaps of 6 h or less (longer gaps raise an error).
  Implausible values are treated as missing.
* **Guard:** the base model drops annual/trend terms when trained on less than 2 years
  of data, because they extrapolate badly from short windows.

## Usage

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[serve,dev]"
# macOS only: LightGBM needs OpenMP ->  brew install libomp
```

Train, evaluate, and save the production model (about 2–3 min on a laptop):

```bash
.venv/bin/pjme train                      # uses config/default.yaml
```

Outputs are written to `artifacts/`: `model.joblib` and `model.json` (metadata and test
metrics), and `reports/metrics.{md,json}` with `mae_by_horizon.png`, `test_week.png`, and
`mae_by_month.png`.

Forecast the 24 hours after the end of any history CSV (same columns as the training data,
at least 168 h of it):

```bash
.venv/bin/pjme forecast --history data/raw/PJME_hourly.csv --horizon 24 --output forecast.csv
```

The output has `forecast` plus the `base` and `residual` components.

HTTP API (optional):

```bash
PJME_MODEL_PATH=artifacts/model.joblib .venv/bin/uvicorn pjme_forecast.service:app --port 8000
# GET /health    POST /forecast  {"history":[{"timestamp":"2018-07-27T01:00","value":31000}, ...], "horizon":24}
```

Python:

```python
from pjme_forecast.pipeline import load_bundle
model, meta = load_bundle("artifacts/model.joblib")
model.forecast(history_series, horizon=24)   # history: hourly pd.Series, >= 168 h
```

Tests: `.venv/bin/python -m pytest` (23 tests; they cover leakage, train/serve parity,
drift absorption, save/load, CLI, and the API; runtime is a few seconds).

## Deployment

Step-by-step Render guide: [docs/DEPLOY_RENDER.md](docs/DEPLOY_RENDER.md). The container is built
from `Dockerfile`, and CI builds and smoke-tests it on every push.

## Configuration

`config/default.yaml` holds the split dates, LightGBM params, base-model terms, horizon,
and `production_model` (`hybrid` by default; set it to `ensemble` for the most accurate
option, or to `lightgbm`).

## Layout

```
config/default.yaml         settings
data/raw/PJME_hourly.csv    dataset
src/pjme_forecast/
  data.py        load, clean, validate, split
  features.py    calendar/holiday, lag & rolling features, base-model design matrix
  models.py      SeasonalBase, RecursiveForecaster (direct / hybrid), naive, ensemble
  evaluation.py  metrics, 1h + day-ahead backtests, plots
  pipeline.py    train → evaluate → refit → save bundle
  cli.py         `pjme train`, `pjme forecast`
  service.py     FastAPI app
tests/                      pytest suite (synthetic data, fast)
notebooks/                  original exploration + results notebook
```

## Limitations and next steps

* **No weather data.** PJM load is driven mostly by temperature. Day-ahead error is
  largest for the afternoon/evening peak hours and in summer/winter extremes (see
  `mae_by_horizon.png`, `mae_by_month.png`). Adding temperature
  forecasts as features is the biggest remaining improvement available.
* Forecasts are point estimates. Quantile LightGBM models would add prediction intervals.
* Model bundles are pickles (joblib). Only load bundles you created yourself.
