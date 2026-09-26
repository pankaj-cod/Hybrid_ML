"""Command-line interface: ``pjme train``, ``pjme forecast`` and ``pjme serve``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .data import DataValidationError, clean_series, load_series
from .features import LOOKBACK


def _train(args) -> int:
    from .pipeline import run_training

    cfg = load_config(args.config)
    if args.data:
        cfg["data"]["path"] = args.data
    if args.artifacts_dir:
        cfg["output"]["artifacts_dir"] = args.artifacts_dir
    report = run_training(cfg, save_production=not args.no_save)
    print(json.dumps({k: report[k] for k in ("split", "production_model") if k in report}, indent=2))
    return 0


def _forecast(args) -> int:
    from .pipeline import load_bundle

    model, meta = load_bundle(args.model)
    cfg = load_config(args.config)
    raw = load_series(args.history, cfg["data"]["timestamp_col"], cfg["data"]["target_col"])
    history = clean_series(raw, cfg["data"]["max_interpolate_hours"])
    # Only the last LOOKBACK hours are needed; keep a margin so interpolation has context.
    history = history.iloc[-(LOOKBACK + 24):]
    horizon = args.horizon or meta.get("default_horizon", 24)
    fc = model.forecast(history, horizon=horizon)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fc.to_csv(args.output, float_format="%.1f")
        logging.getLogger(__name__).info("Wrote %d rows to %s", len(fc), args.output)
    else:
        print(fc.round(1).to_string())
    return 0


def _serve(args) -> int:
    import os

    import uvicorn

    for name, value in (("PJME_MODEL_PATH", args.model), ("PJME_HISTORY_PATH", args.history),
                        ("PJME_REPORTS_DIR", args.reports)):
        if not Path(value).exists():
            raise FileNotFoundError(f"{value} not found (run `pjme train` first?)")
        os.environ[name] = str(value)
    print(f"Dashboard: http://{args.host}:{args.port}/dashboard   API docs: http://{args.host}:{args.port}/docs")
    uvicorn.run("pjme_forecast.service:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pjme", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Evaluate all models on the test period and save the production model")
    p.add_argument("--config", default=None, help="YAML config (default: config/default.yaml)")
    p.add_argument("--data", default=None, help="Override data.path from the config")
    p.add_argument("--artifacts-dir", default=None, help="Override output.artifacts_dir")
    p.add_argument("--no-save", action="store_true", help="Only evaluate; do not refit/save a model")
    p.set_defaults(func=_train)

    p = sub.add_parser("forecast", help="Forecast the hours after the end of a history CSV")
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--history", required=True, help="CSV with the same columns as the training data")
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--output", default=None, help="Write CSV here instead of printing")
    p.add_argument("--config", default=None)
    p.set_defaults(func=_forecast)

    p = sub.add_parser("serve", help="Run the API and demo dashboard locally")
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--history", default="data/raw/PJME_hourly.csv")
    p.add_argument("--reports", default="artifacts/reports")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        logging.getLogger("pjme").error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
