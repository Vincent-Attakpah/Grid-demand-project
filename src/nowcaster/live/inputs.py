"""Live inputs in exactly the shapes the backtest feature code reads.

EIA: the API's hourly demand (type D) equals the bulk files' adjusted demand (checked on
2026-11-15..19: identical for SOCO, TVA, FPL; DUK within 3 MW), and type DF is the
day-ahead forecast. The bulk "imputed" flag is not in the API, so live hours are treated
as reported.

Weather: one Previous Runs API call per BA with the day-1 and day-2 values (the same
`prev_day1` / `prev_day2` sources the backtest used for future hours) plus the latest
temperature for recent past hours (the thermal-lag inputs, standing in for
`hist_forecast`, which the backtest reads only up to issue time - 6 h).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import requests

from ..config import BA, load_bas, weather_settings
from ..logger import eia as eia_api
from ..logger.common import get_json
from ..weather.backfill import parse_payload
from ..weather.client import ENDPOINTS
from ..weather.weighting import weight_ba

EIA_DAYS = 65          # 60-day error library + a few days for lags


def eia_frame(rows: list[dict]) -> pd.DataFrame:
    """Logger-style EIA rows -> eia_hourly shape (ba, period_utc, demand_adjusted, da_forecast)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ba", "period_utc", "demand_adjusted", "da_forecast", "demand_imputed"])
    df["period_utc"] = pd.to_datetime(df["period_utc"], utc=True)
    w = (df.drop_duplicates(["ba", "type", "period_utc"], keep="last")
           .pivot_table(index=["ba", "period_utc"], columns="type", values="value", aggfunc="last")
           .reset_index())
    w = w.rename(columns={"D": "demand_adjusted", "DF": "da_forecast"})
    for c in ("demand_adjusted", "da_forecast"):
        if c not in w:
            w[c] = float("nan")
    w["demand_imputed"] = False
    return w[["ba", "period_utc", "demand_adjusted", "da_forecast", "demand_imputed"]]


def fetch_eia(now: dt.datetime, bas, days: int = EIA_DAYS, session=None) -> pd.DataFrame:
    rows = eia_api.fetch(now, bas=tuple(bas), types=("D", "DF"), window_hours=24 * days, session=session)
    return eia_frame(rows)


def weather_params(ba: BA, base_vars: list[str], model: str) -> dict:
    hourly = [f"{v}_previous_day{d}" for v in base_vars for d in (1, 2)] + ["temperature_2m"]
    return {
        "latitude": ",".join(f"{c.lat:.4f}" for c in ba.cities),
        "longitude": ",".join(f"{c.lon:.4f}" for c in ba.cities),
        "hourly": ",".join(hourly), "models": model, "timezone": "GMT",
        "past_days": 3, "forecast_days": 2,
    }


def weather_frame(code: str, ba: BA, response, base_vars: list[str]) -> pd.DataFrame:
    """API response (list of locations) -> one BA's weather_ba_hourly rows."""
    resp = response if isinstance(response, list) else [response]
    meta = {"ba": code, "start": "", "cities": [c.name for c in ba.cities]}
    city = pd.concat([parse_payload({"meta": {**meta, "source": "previous_runs_full"}, "response": resp}, base_vars),
                      parse_payload({"meta": {**meta, "source": "historical_forecast"}, "response": resp}, base_vars)],
                     ignore_index=True)
    city = city.dropna(subset=base_vars, how="all")
    return weight_ba(city, {code: ba}, base_vars)


def fetch_weather(code: str, ba: BA, session=None) -> pd.DataFrame:
    ws = weather_settings()
    session = session or requests.Session()
    resp = get_json(session, ENDPOINTS["previous_runs"], weather_params(ba, ws["variables"], ws["model"]))
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(f"Open-Meteo error for {code}: {resp.get('reason')}")
    return weather_frame(code, ba, resp, ws["variables"])


def core_bas() -> dict[str, BA]:
    return load_bas()
