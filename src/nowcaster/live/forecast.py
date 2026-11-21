"""One live issue for one BA, built with the backtest's own functions.

Hourly forecasts: build_panel -> add_calendar -> attach_weather for the single issue time,
then B4b (Trick 2), M3 and the M4 ranges from the monthly bundle.

Peak warning: the issue is appended to the forecast library (bundle seed + live history),
and the backtest's peak code is run on it unchanged: 60-day whole-day error paths at the
same local hour and number of remaining hours, a stretch factor chosen at the month start
from the previous 90 days, and the high-day line from the two previous calendar years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..baselines.simple import b4b_predict
from ..features.calendar import add_calendar
from ..features.panel import ISSUE_MINUTE, Lags, build_panel, clean_hourly
from ..features.weather_join import attach_weather
from ..models.gbm import m3_predict
from ..models.intervals import LEVELS, m4_predict
from ..models.peaks import (actual_peak, inflation_factors, issue_arrays, peak_ensembles, stretch,
                            yearly_thresholds)

KEEP = ["ba", "issue_utc", "target_utc", "op_day", "horizon", "da_posted", "da"]
Q_COLS = [f"M4_{k}" for k in LEVELS]
HIST_COLS = KEEP + ["B4b", "M3", *Q_COLS]
LIB_COLS = ["issue_utc", "target_utc", "op_day", "horizon", "da_posted", "da", "B4b", "M3"]
LIB_WINDOW_DAYS = 155        # 90-day stretch window + 60-day library + slack


def issue_time(now) -> pd.Timestamp:
    """The latest :17 issue time at or before `now` (a late run keeps its nominal hour)."""
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    m = pd.Timedelta(minutes=ISSUE_MINUTE)
    return (now - m).floor("h") + m


def forecast_issue(models: dict, eia_ba: pd.DataFrame, wba: pd.DataFrame, code: str, tz: str,
                   tau: pd.Timestamp, lags: Lags | None = None) -> pd.DataFrame:
    hourly = clean_hourly(eia_ba)
    p = build_panel(hourly, tz, pd.DatetimeIndex([tau]), lags, ba=code)
    p = add_calendar(p, tz)
    p = attach_weather(p, wba)
    out = p[KEEP].copy()
    m3 = models.get("m3")
    out["B4b"] = b4b_predict(m3[0], p) if m3 else np.nan
    out["M3"] = m3_predict(m3, p)
    q = m4_predict(models.get("m4"), p)
    for k in LEVELS:
        out[f"M4_{k}"] = q[k].to_numpy()
    return out


def library(seed: pd.DataFrame, history: pd.DataFrame, actual: pd.Series, since) -> pd.DataFrame:
    """Past forecasts with their actuals: live history where present, else the backtest seed."""
    parts = [seed.assign(_src=0)]
    if history is not None and not history.empty:
        parts.append(history.assign(_src=1))
    lib = pd.concat([x[[*LIB_COLS, "_src"]] for x in parts], ignore_index=True)
    lib["issue_utc"] = pd.to_datetime(lib["issue_utc"], utc=True)
    lib["target_utc"] = pd.to_datetime(lib["target_utc"], utc=True)
    lib["op_day"] = pd.to_datetime(lib["op_day"]).dt.normalize()
    since = pd.Timestamp(since)
    lib = lib[lib["op_day"] >= (since.tz_localize(None) if since.tzinfo else since)]
    lib = lib.sort_values("_src").drop_duplicates(["issue_utc", "horizon"], keep="last")
    # actual by target hour: live/archived labels first, then the seed's copy of that hour
    seed_act = (seed.assign(target_utc=pd.to_datetime(seed["target_utc"], utc=True))
                .dropna(subset=["actual"]).groupby("target_utc")["actual"].first())
    act = actual.combine_first(seed_act) if len(actual) else seed_act
    lib["actual"] = act.reindex(pd.DatetimeIndex(lib["target_utc"])).to_numpy()
    return lib.drop(columns="_src").sort_values(["issue_utc", "horizon"]).reset_index(drop=True)


def peak_now(rows: pd.DataFrame, lib: pd.DataFrame, daily_peaks: pd.Series, tz: str) -> dict | None:
    """Rest-of-day peak forecast and high-day chance for the issue in `rows`."""
    if rows.empty or not rows["da_posted"].any() or not np.isfinite(rows["M3"]).all():
        return None
    tau = pd.Timestamp(rows["issue_utc"].iloc[0])
    cur = rows[LIB_COLS].copy()
    cur["op_day"] = pd.to_datetime(cur["op_day"]).dt.normalize()
    cur["actual"] = np.nan
    pred = pd.concat([lib[lib["issue_utc"] < tau], cur], ignore_index=True)
    meta, arrs = issue_arrays(pred, ["M3", "B4b", "da"], tz)
    i = int(np.flatnonzero(meta["issue_utc"] == tau)[0])
    raw = peak_ensembles(meta, arrs, "M3")
    if raw[i] is None:
        return None
    c = inflation_factors(meta, raw, actual_peak(arrs))[i]
    ens = stretch([raw[i]], np.array([c]))[0]
    day = pd.Timestamp(meta.at[i, "op_day"])
    thr_tab = yearly_thresholds(daily_peaks, [day.year])
    thr = float(thr_tab.loc[day.year, day.month])
    pres = arrs["present"][i]
    lo, med, hi = np.quantile(ens, [0.05, 0.5, 0.95])
    return {"ba": rows["ba"].iloc[0], "issue_utc": tau, "op_day": day.date().isoformat(),
            "K": int(pres.sum()), "local_hour": int(meta.at[i, "local_hour"]),
            "threshold": thr, "stretch": float(c), "n_lib": int(len(ens)),
            "peak_M5": float(med), "lo_M5": float(lo), "hi_M5": float(hi),
            "p_M5": float(np.mean(ens > thr)),
            "peak_point_M3": float(np.nanmax(arrs["M3"][i][pres])),
            "peak_point_B4b": float(np.nanmax(arrs["B4b"][i][pres])),
            "peak_point_da": float(np.nanmax(arrs["da"][i][pres]))}
