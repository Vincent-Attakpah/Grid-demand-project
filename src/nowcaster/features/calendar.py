"""Calendar features for target hours (always allowed: known in advance)."""

from __future__ import annotations

import datetime as dt

import holidays
import numpy as np
import pandas as pd

H = pd.Timedelta(hours=1)


def _holiday_set(years) -> set[dt.date]:
    return set(holidays.US(years=sorted(set(years))).keys())


def add_calendar(panel: pd.DataFrame, tz: str, col: str = "target_utc") -> pd.DataFrame:
    """Add hour/day/holiday/Christmas-distance/Fourier features for `col` in BA local time."""
    p = panel.copy()
    t = pd.DatetimeIndex(p[col])
    local = t.tz_convert(tz) - H                      # start of the hour, local
    day = local.tz_localize(None).normalize()
    p["hour"] = (local.hour + 1).astype(int)          # hour-ending 1..24
    p["dow"] = local.dayofweek.astype(int)
    p["month"] = local.month.astype(int)
    hol = _holiday_set(range(day.year.min() - 1, day.year.max() + 2))
    dates = day.date
    p["holiday"] = np.fromiter((d in hol for d in dates), bool, len(dates))
    prev = np.fromiter(((d - dt.timedelta(days=1)) in hol for d in dates), bool, len(dates))
    nxt = np.fromiter(((d + dt.timedelta(days=1)) in hol for d in dates), bool, len(dates))
    dow = p["dow"].to_numpy()
    # Bridge day: a weekday squeezed between a holiday and a weekend.
    p["bridge"] = ((dow == 0) & nxt) | ((dow == 4) & prev)
    xmas = pd.to_datetime([f"{y}-12-25" for y in day.year])
    p["days_to_xmas"] = (xmas - day).days.to_numpy()          # negative after Christmas
    ny = pd.to_datetime([f"{y + (m == 12)}-01-01" for y, m in zip(day.year, day.month)])
    p["days_to_newyear"] = (ny - day).days.to_numpy()
    # 18 Dec .. 3 Jan. Days after Christmas measure from the *previous* year's Christmas in Jan.
    after_prev_xmas = (day - pd.to_datetime([f"{y - 1}-12-25" for y in day.year])).days.to_numpy()
    p["xmas_window"] = (np.abs(p["days_to_xmas"].to_numpy()) <= 7) | ((day.month == 1) & (after_prev_xmas <= 9))
    doy = local.dayofyear.to_numpy()
    for k in (1, 2):
        p[f"doy_sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        p[f"doy_cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return p
