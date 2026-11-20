"""Rolling-origin backtest of baselines and models (plan v2 section 9.1).

    python -m nowcaster.features.build          # first: feature panels
    python scripts/backtest_models.py [--bas SOCO,TVA] [--start 2025-01-01 --end 2026-11-20]
    python scripts/report_models.py             # then: tables and tests

Refits every model at each month start on labels published before it. Saves all
predictions to <data>/backtest/pred_<BA>.parquet.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from nowcaster.baselines.simple import BASELINES
from nowcaster.config import data_dir, load_bas
from nowcaster.eval.rolling import rolling_origin
from nowcaster.features.build import FULL_WEATHER_START, load_panel, panel_path
from nowcaster.models.gbm import MODELS

KEEP = ["ba", "issue_utc", "target_utc", "op_day", "horizon", "hour", "da_posted", "actual",
        "da", "wx_temperature_2m"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bas", default=None)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    ap.add_argument("--train-start", default=FULL_WEATHER_START)
    a = ap.parse_args(argv)
    codes = a.bas.split(",") if a.bas else [c for c in load_bas() if panel_path(c).exists()]
    models = {**BASELINES, **MODELS}
    out_dir = data_dir() / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in codes:
        t0 = time.time()
        panel = load_panel(code)
        res = rolling_origin(panel, models, a.start, a.end, train_start=a.train_start)
        res[KEEP + ["origin", *models]].to_parquet(out_dir / f"pred_{code}.parquet", index=False)
        print(f"{code}: {len(res)} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
