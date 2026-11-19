"""Attach point-in-time weather to the issue-time panel (plan v2 section 7.1).

Future target hours: Previous Runs `prev_dayN` with N = asof.safe_offset(horizon), never
hist_forecast or era5 (test_weather_future_source).
Past hours (thermal lag): hist_forecast only up to floor(tau - availability), because a
stitched run's first hours are usable only after the run is published
(test_weather_past_delay).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import lag_settings
from ..weather.asof import safe_offset

H = pd.Timedelta(hours=1)
FUTURE_VARS = ["temperature_2m", "dew_point_2m", "wind_speed_10m", "cloud_cover",
               "shortwave_radiation", "relative_humidity_2m"]
THERMAL_WINDOWS = (3, 6, 24, 48)
BASE_C = 18.0  # degree-hour base temperature, deg C


def _series(wba: pd.DataFrame, source: str, var: str) -> pd.Series:
    w = wba[wba["source"] == source]
    s = pd.Series(w[var].to_numpy(dtype=float), index=pd.DatetimeIndex(w["period_utc"]).tz_convert("UTC"))
    return s[~s.index.duplicated(keep="last")].sort_index()


def attach_weather(panel: pd.DataFrame, wba: pd.DataFrame, availability_h: float | None = None,
                   variables=FUTURE_VARS) -> pd.DataFrame:
    """`wba`: one BA's weather_ba_hourly rows (columns source, period_utc, variables)."""
    if availability_h is None:
        availability_h = float(lag_settings()["weather_run_availability_hours"])
    p = panel.copy()
    target = pd.DatetimeIndex(p["target_utc"])
    offsets = np.array([safe_offset(k, availability_h) for k in p["horizon"].to_numpy()])
    p["wx_offset"] = offsets
    for var in [v for v in variables if v in wba.columns]:
        out = np.full(len(p), np.nan)
        for n in np.unique(offsets):
            m = offsets == n
            out[m] = _series(wba, f"prev_day{n}", var).reindex(target[m]).to_numpy()
        p[f"wx_{var}"] = out
    if "wx_temperature_2m" in p:
        t = p["wx_temperature_2m"]
        p["hdh"] = np.maximum(0.0, BASE_C - t)
        p["cdh"] = np.maximum(0.0, t - BASE_C)

    # Thermal lag from hist_forecast, anchored at the last hour usable at issue time.
    if "temperature_2m" in wba.columns and (wba["source"] == "hist_forecast").any():
        hist = _series(wba, "hist_forecast", "temperature_2m")
        hist = hist.reindex(pd.date_range(hist.index.min(), hist.index.max(), freq="h", tz="UTC"))
        anchor = (pd.DatetimeIndex(p["issue_utc"]) - pd.Timedelta(hours=availability_h)).floor("h")
        for w in THERMAL_WINDOWS:
            roll = hist.rolling(w, min_periods=max(1, w // 2)).mean()
            p[f"temp_mean_{w}h"] = roll.reindex(anchor).to_numpy()
    return p
