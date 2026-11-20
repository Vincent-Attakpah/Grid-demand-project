"""Full model panel for one BA: as-of demand/DA features + calendar + point-in-time weather.

    python -m nowcaster.features.build [--start 2024-01-21 --end 2026-11-20]

Writes <data>/features/panel_<BA>.parquet (one file per BA, rebuilt each call).
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ..config import data_dir, load_bas
from ..storage import query
from .calendar import add_calendar
from .panel import Lags, build_panel_for_ba
from .weather_join import attach_weather

log = logging.getLogger(__name__)

FULL_WEATHER_START = "2024-01-21"   # gfs_seamless Previous Runs start for all variables (docs/decisions.md)


def load_eia() -> pd.DataFrame:
    eia = query("select ba, period_utc, demand_adjusted, da_forecast, demand_imputed "
                "from read_parquet('{clean}/eia_hourly/**/*.parquet', hive_partitioning=true)")
    eia["period_utc"] = pd.to_datetime(eia["period_utc"], utc=True)
    return eia


def load_weather(ba: str) -> pd.DataFrame:
    w = query("select * exclude (n_cities, code_version, month) "
              "from read_parquet('{clean}/weather_ba_hourly/**/*.parquet', hive_partitioning=true) "
              f"where ba = '{ba}'")
    w["period_utc"] = pd.to_datetime(w["period_utc"], utc=True)
    return w


def build_features_for_ba(eia: pd.DataFrame, wba: pd.DataFrame, code: str, tz: str,
                          start, end, lags: Lags | None = None) -> pd.DataFrame:
    p = build_panel_for_ba(eia, code, tz, start, end, lags)
    p = add_calendar(p, tz)
    return attach_weather(p, wba)


def panel_path(code: str):
    return data_dir() / "features" / f"panel_{code}.parquet"


def load_panel(code: str) -> pd.DataFrame:
    return pd.read_parquet(panel_path(code))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=FULL_WEATHER_START)
    ap.add_argument("--end", default=None, help="default: today (UTC)")
    a = ap.parse_args(argv)
    end = a.end or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    eia = load_eia()
    lags = Lags.from_config()
    for code, ba in load_bas().items():
        wba = load_weather(code)
        if wba.empty:
            log.info("skip %s: no weather", code)
            continue
        p = build_features_for_ba(eia, wba, code, ba.tz, a.start, end, lags)
        out = panel_path(code)
        out.parent.mkdir(parents=True, exist_ok=True)
        p.to_parquet(out, index=False)
        log.info("%s: %d rows, %d columns -> %s", code, len(p), p.shape[1], out)


if __name__ == "__main__":
    main()
