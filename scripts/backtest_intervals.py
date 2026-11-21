"""Rolling-origin backtest of forecast ranges (M4 and the reference Q0).

    python scripts/backtest_intervals.py [--bas SOCO] [--start 2025-01-01 --end 2026-11-19]
    python scripts/report_intervals.py

Only issues after the day's DA is posted (both methods are built on B4b). Saves
<data>/backtest/int_<BA>.parquet.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from nowcaster.config import data_dir, load_bas
from nowcaster.eval.rolling import rolling_origin
from nowcaster.features.build import FULL_WEATHER_START, load_panel, panel_path
from nowcaster.models.intervals import INTERVAL_MODELS

KEEP = ["ba", "issue_utc", "target_utc", "op_day", "horizon", "hour", "actual", "da", "origin"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bas", default=None)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    ap.add_argument("--train-start", default=FULL_WEATHER_START)
    a = ap.parse_args(argv)
    codes = a.bas.split(",") if a.bas else [c for c in load_bas() if panel_path(c).exists()]
    out_dir = data_dir() / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    for code in codes:
        t0 = time.time()
        panel = load_panel(code)
        panel = panel[panel["da_posted"]].reset_index(drop=True)
        res = rolling_origin(panel, INTERVAL_MODELS, a.start, a.end, train_start=a.train_start)
        cols = KEEP + [c for c in res.columns if c.startswith(("M4_", "Q0_"))]
        res[cols].to_parquet(out_dir / f"int_{code}.parquet", index=False)
        print(f"{code}: {len(res)} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
