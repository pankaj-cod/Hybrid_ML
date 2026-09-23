"""End-to-end: CLI train on a small CSV, CLI forecast, and the HTTP service."""
import json
from pathlib import Path

import pandas as pd
import pytest

from pjme_forecast.cli import main
from tests.conftest import make_series


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    s = make_series(days=160, start="2016-01-01", seed=1)
    csv = root / "load.csv"
    s.rename_axis("Datetime").reset_index().to_csv(csv, index=False)
    cfg = root / "cfg.yaml"
    cfg.write_text(
        f"""
data: {{path: "{csv}"}}
split: {{val_start: "2016-03-20", test_start: "2016-05-01"}}
lightgbm: {{n_estimators: 100, learning_rate: 0.1, num_leaves: 15, early_stopping_rounds: 20}}
output: {{artifacts_dir: "{root / 'artifacts'}"}}
"""
    )
    assert main(["train", "--config", str(cfg)]) == 0
    return root, csv, cfg


def test_train_writes_artifacts(trained):
    root, _, _ = trained
    art = root / "artifacts"
    assert (art / "model.joblib").exists()
    meta = json.loads((art / "model.json").read_text())
    assert meta["model_kind"] == "hybrid"
    report = json.loads((art / "reports" / "metrics.json").read_text())
    assert {"naive_24h", "lightgbm", "hybrid", "ensemble"} <= set(report["metrics"])
    for png in ("mae_by_horizon.png", "test_week.png", "mae_by_month.png"):
        assert (art / "reports" / png).stat().st_size > 0


def test_cli_forecast(trained):
    root, csv, cfg = trained
    out = root / "fc.csv"
    rc = main(["forecast", "--model", str(root / "artifacts" / "model.joblib"),
               "--history", str(csv), "--horizon", "24", "--output", str(out), "--config", str(cfg)])
    assert rc == 0
    fc = pd.read_csv(out, index_col=0, parse_dates=True)
    assert len(fc) == 24 and fc["forecast"].notna().all()


def test_cli_rejects_bad_history(trained, tmp_path):
    root, _, cfg = trained
    bad = tmp_path / "short.csv"
    pd.DataFrame({"Datetime": pd.date_range("2020-01-01", periods=10, freq="h"),
                  "PJME_MW": 30000.0}).to_csv(bad, index=False)
    rc = main(["forecast", "--model", str(root / "artifacts" / "model.joblib"),
               "--history", str(bad), "--config", str(cfg)])
    assert rc == 2


def test_http_service(trained, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pjme_forecast import service

    root, csv, _ = trained
    monkeypatch.setenv("PJME_MODEL_PATH", str(root / "artifacts" / "model.joblib"))
    monkeypatch.setenv("PJME_HISTORY_PATH", str(csv))
    service.get_model.cache_clear()
    service.get_demo_history.cache_clear()
    with TestClient(service.app) as client:  # runs the startup model load
        assert client.get("/health").json()["status"] == "ok"

        df = pd.read_csv(csv).tail(200)
        body = {"history": [{"timestamp": t, "value": v} for t, v in zip(df.Datetime, df.PJME_MW)],
                "horizon": 12}
        r = client.post("/forecast", json=body)
        assert r.status_code == 200, r.text
        assert len(r.json()["forecast"]) == 12

        body["history"] = body["history"][:50]
        assert client.post("/forecast", json=body).status_code == 422

        latest = client.get("/forecast/latest", params={"horizon": 6})
        assert latest.status_code == 200, latest.text
        assert len(latest.json()["forecast"]) == 6
