"""Configuration loading."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

_DEFAULTS: dict[str, Any] = {
    "data": {
        "path": "data/raw/PJME_hourly.csv",
        "timestamp_col": "Datetime",
        "target_col": "PJME_MW",
        "max_interpolate_hours": 6,
    },
    "split": {"val_start": "2015-01-01", "test_start": "2017-01-01"},
    "forecast": {"horizon": 24, "eval_origin_hour": 0},
    "lightgbm": {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_estimators": 5000,
        "early_stopping_rounds": 200,
        "random_state": 42,
    },
    "base_model": {"fourier_order": 3, "ridge_alpha": 1.0, "include_trend": True},
    "ensemble": {"weight_hybrid": 0.5},
    "production_model": "ensemble_bc",
    "output": {"artifacts_dir": "artifacts"},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config, filling anything missing from the built-in defaults."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        if path == DEFAULT_CONFIG_PATH:
            return copy.deepcopy(_DEFAULTS)
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    return _merge(_DEFAULTS, user)
