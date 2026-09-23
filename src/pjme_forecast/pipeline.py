"""End-to-end training: fit candidates, evaluate on the held-out test period, save the production model."""
from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from . import __version__
from .data import load_clean_series, split_series
from .evaluation import backtest, mae_by_horizon, plot_reports, summary_table
from .features import LOOKBACK
from .models import EnsembleForecaster, Forecaster, build_model

log = logging.getLogger(__name__)

CANDIDATES = ("naive_24h", "naive_168h", "linear", "lightgbm", "hybrid")
BUNDLE_NAME = "model.joblib"


def run_training(cfg: dict, save_production: bool = True) -> dict[str, Any]:
    out_dir = Path(cfg["output"]["artifacts_dir"])
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    series = load_clean_series(cfg)
    train, val, test = split_series(series, cfg["split"]["val_start"], cfg["split"]["test_start"])
    test_start = test.index[0]
    horizon = int(cfg["forecast"]["horizon"])
    origin_hour = int(cfg["forecast"]["eval_origin_hour"])
    log.info("train %s..%s (%d h)", train.index[0], train.index[-1], len(train))
    log.info("val   %s..%s (%d h)", val.index[0], val.index[-1], len(val))
    log.info("test  %s..%s (%d h)", test.index[0], test.index[-1], len(test))

    # 1) Fit each candidate on train (early stopping on val), then refit on train+val
    #    with the chosen tree count, and evaluate once on the untouched test period.
    train_val = pd.concat([train, val])
    fitted: dict[str, Forecaster] = {}
    results = []
    for kind in CANDIDATES:
        t0 = time.time()
        model = build_model(kind, cfg).fit(train, val).refit(train_val)
        fitted[kind] = model
        results.append(backtest(model, series, test_start, horizon, origin_hour))
        log.info("%-10s fitted + evaluated in %.0fs", kind, time.time() - t0)
    fitted["ensemble"] = EnsembleForecaster("ensemble", [fitted["lightgbm"], fitted["hybrid"]])
    results.append(backtest(fitted["ensemble"], series, test_start, horizon, origin_hour))

    summary = summary_table(results)
    by_h = mae_by_horizon(results)
    split_info = {
        "train": [str(train.index[0]), str(train.index[-1])],
        "val": [str(val.index[0]), str(val.index[-1])],
        "test": [str(test.index[0]), str(test.index[-1])],
    }
    report = {
        "split": split_info,
        "horizon": horizon,
        "metrics": summary.to_dict(orient="index"),
        "mae_by_horizon": {k: v.round(2).tolist() for k, v in by_h.items()},
    }
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (report_dir / "metrics.md").write_text(_markdown_report(summary, by_h, split_info))
    plot_reports(results, series, report_dir)
    log.info("\n%s", _console_table(summary))

    # 2) Refit the production model on *all* data and save it.
    if save_production:
        kind = cfg["production_model"]
        if kind not in fitted:
            raise ValueError(f"production_model {kind!r} is not one of {list(fitted)}")
        prod = fitted[kind].refit(series)
        metadata = {
            "package_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "model_kind": kind,
            "model": prod.describe(),
            "trained_on": [str(series.index[0]), str(series.index[-1])],
            "lookback_hours": LOOKBACK,
            "default_horizon": horizon,
            "target": cfg["data"]["target_col"],
            "test_metrics": summary.loc[kind].to_dict(),
            "evaluation_split": split_info,
        }
        path = save_bundle(prod, metadata, out_dir / BUNDLE_NAME)
        report["production_model"] = str(path)
        log.info("Saved production model (%s) to %s", kind, path)
    return report


def save_bundle(model: Forecaster, metadata: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))
    return path


def load_bundle(path: str | Path) -> tuple[Forecaster, dict]:
    """Load a saved model. Only load bundles you created: joblib files can execute code."""
    bundle = joblib.load(path)
    return bundle["model"], bundle["metadata"]


def _console_table(summary: pd.DataFrame) -> str:
    cols = ["1h_MAE", "1h_MAPE", "day_ahead_MAE", "day_ahead_RMSE", "day_ahead_MAPE", "day_ahead_R2"]
    return summary[cols].astype(float).round(3).to_string()


def _markdown_report(summary: pd.DataFrame, by_h: pd.DataFrame, split: dict) -> str:
    cols = {
        "1h_MAE": "1h MAE (MW)", "1h_RMSE": "1h RMSE", "1h_MAPE": "1h MAPE %",
        "day_ahead_MAE": "Day-ahead MAE (MW)", "day_ahead_RMSE": "Day-ahead RMSE",
        "day_ahead_MAPE": "Day-ahead MAPE %", "day_ahead_R2": "Day-ahead R²",
    }
    t = summary[list(cols)].astype(float).rename(columns=cols)
    lines = [
        "# Test-set evaluation",
        "",
        f"Train {split['train'][0]} → {split['train'][1]} · "
        f"Validation {split['val'][0]} → {split['val'][1]} · "
        f"Test {split['test'][0]} → {split['test'][1]}",
        "",
        "* **1h**: one-hour-ahead (all lags are actual observations).",
        "* **Day-ahead**: one forecast per day at midnight for the next 24 h, produced recursively.",
        "",
        _to_markdown(t.round(2)),
        "",
        "## Day-ahead MAE by horizon (MW)",
        "",
        _to_markdown(by_h.loc[[1, 2, 3, 6, 12, 18, 24]].round(0)),
        "",
    ]
    return "\n".join(lines)


def _to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [f"{v:,.2f}" if isinstance(v, float) else str(v) for v in r]) + " |"
            for i, r in zip(df.index, df.to_numpy())]
    return "\n".join([header, sep, *rows])
