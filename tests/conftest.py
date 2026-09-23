import numpy as np
import pandas as pd
import pytest

from pjme_forecast.config import load_config


def make_series(days: int = 150, start: str = "2016-01-01", seed: int = 0) -> pd.Series:
    """Synthetic load with daily, weekly and annual cycles plus AR(1) noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    h, dow, doy = idx.hour.to_numpy(), idx.dayofweek.to_numpy(), idx.dayofyear.to_numpy()
    daily = 4000 * np.sin(2 * np.pi * (h - 6) / 24)
    weekly = np.where(dow >= 5, -2500, 0)
    annual = 3000 * np.cos(2 * np.pi * (doy - 200) / 365.25)
    noise = np.zeros(len(idx))
    eps = rng.normal(0, 300, len(idx))
    for i in range(1, len(idx)):
        noise[i] = 0.9 * noise[i - 1] + eps[i]
    return pd.Series(30000 + daily + weekly + annual + noise, index=idx, name="PJME_MW")


@pytest.fixture(scope="session")
def series() -> pd.Series:
    return make_series()


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> dict:
    cfg = load_config()
    cfg["lightgbm"].update(n_estimators=150, learning_rate=0.1, num_leaves=15, early_stopping_rounds=20)
    cfg["output"]["artifacts_dir"] = str(tmp_path_factory.mktemp("artifacts"))
    return cfg
