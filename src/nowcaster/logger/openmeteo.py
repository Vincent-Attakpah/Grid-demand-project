"""Open-Meteo Forecast API logger client: one request per BA, all cities in it.

Time stamps are logged exactly as Open-Meteo returns them (``timezone=GMT``) in
``time_utc``. Open-Meteo state variables (temperature, dew point, wind, cloud, humidity) are
instantaneous at that time; ``shortwave_radiation`` is the mean over the preceding hour.
Mapping onto the project's hour-ending convention is left to the feature builder so the
raw log stays a faithful copy of what was served.

The model run in use is recorded from Open-Meteo's per-model metadata endpoint
(``/data/<model>/static/meta.json``); the forecast response itself carries no run time.
``gfs_seamless`` blends several NCEP products, so metadata for each is recorded.
"""

from __future__ import annotations

import datetime as dt

import requests

from ..config import load_bas, weather_settings
from .common import UTC, get_json, iso

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
META_URL = "https://api.open-meteo.com/data/{model}/static/meta.json"
# Underlying products of gfs_seamless (best effort; unknown ones simply fail and are noted).
SEAMLESS_COMPONENTS = {"gfs_seamless": ("ncep_gfs013", "ncep_gfs025", "ncep_hrrr_conus")}


def _active(city, now: dt.datetime) -> bool:
    return city.active_from is None or dt.date.fromisoformat(city.active_from) <= now.date()


def fetch_ba(ba_code: str, cities, now: dt.datetime, *, model: str, variables, session,
             forecast_days: int = 2, past_days: int = 1) -> list[dict]:
    cities = [c for c in cities if _active(c, now)]
    params = {
        "latitude": ",".join(f"{c.lat}" for c in cities),
        "longitude": ",".join(f"{c.lon}" for c in cities),
        "hourly": ",".join(variables),
        "models": model,
        "forecast_days": forecast_days,
        "past_days": past_days,
        "timezone": "GMT",
        "timeformat": "iso8601",
    }
    payload = get_json(session, FORECAST_URL, params)
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason')}")
    locs = payload if isinstance(payload, list) else [payload]
    if len(locs) != len(cities):
        raise RuntimeError(f"{ba_code}: expected {len(cities)} locations, got {len(locs)}")
    rows = []
    for city, loc in zip(cities, locs):
        hourly = loc["hourly"]
        for i, t in enumerate(hourly["time"]):
            ts = dt.datetime.fromisoformat(t).replace(tzinfo=UTC)
            row = {"ba": ba_code, "city": city.name, "time_utc": iso(ts), "model": model}
            for v in variables:
                row[v] = hourly.get(v, [None] * len(hourly["time"]))[i]
            rows.append(row)
    return rows


def fetch(now: dt.datetime, *, bas=None, session=None) -> tuple[list[dict], dict]:
    """All configured BAs -> (rows, per-BA errors). Raises only if every BA fails."""
    session = session or requests.Session()
    ws = weather_settings()
    cfg = load_bas()
    codes = bas or list(cfg)
    rows, errors = [], {}
    for code in codes:
        try:
            rows += fetch_ba(code, cfg[code].cities, now, model=ws["model"],
                             variables=ws["variables"], session=session)
        except Exception as e:  # keep other BAs going
            errors[code] = f"{type(e).__name__}: {e}"
    if errors and len(errors) == len(codes):
        raise RuntimeError(f"all Open-Meteo requests failed: {errors}")
    return rows, errors


def fetch_meta(model: str, session=None) -> dict:
    session = session or requests.Session()
    out = {}
    for comp in SEAMLESS_COMPONENTS.get(model, (model,)):
        try:
            m = get_json(session, META_URL.format(model=comp), None, retries=1, timeout=10)
            out[comp] = {k: (iso(dt.datetime.fromtimestamp(m[k], UTC)) if isinstance(m.get(k), (int, float)) else m.get(k))
                         for k in ("last_run_initialisation_time", "last_run_availability_time",
                                   "last_run_modification_time") if k in m}
        except Exception as e:
            out[comp] = {"error": f"{type(e).__name__}: {e}"}
    return out


def key_and_sig(row: dict, variables) -> tuple[str, str, str]:
    sig = repr(tuple(row.get(v) for v in variables))
    return f"{row['ba']}|{row['city']}|{row['time_utc']}", sig, row["time_utc"]
