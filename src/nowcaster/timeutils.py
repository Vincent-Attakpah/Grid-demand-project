"""Time handling. Storage convention: UTC, hour-ending, tz-aware pandas Timestamps.

An hour-ending stamp H covers the interval (H-1h, H]. The operating day of a BA is its
local calendar day; local hour-ending 00:00 belongs to the *previous* day (it is HE24).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

UTC = "UTC"


def to_utc(ts) -> pd.Timestamp | pd.DatetimeIndex:
    """Coerce timestamps to tz-aware UTC. Naive inputs are assumed to already be UTC."""
    if isinstance(ts, (pd.DatetimeIndex, pd.Series)):
        idx = pd.DatetimeIndex(ts)
        return idx.tz_localize(UTC) if idx.tz is None else idx.tz_convert(UTC)
    t = pd.Timestamp(ts)
    return t.tz_localize(UTC) if t.tzinfo is None else t.tz_convert(UTC)


def floor_hour(ts) -> pd.Timestamp:
    return to_utc(ts).floor("h")


def operating_day(he_utc, tz: str):
    """Local operating date for hour-ending UTC stamp(s)."""
    local = to_utc(he_utc).tz_convert(tz) - pd.Timedelta(hours=1)
    if isinstance(local, pd.DatetimeIndex):
        return local.date
    return local.date()


def operating_day_hours(day: dt.date, tz: str) -> pd.DatetimeIndex:
    """All hour-ending UTC stamps of a local operating day (23/24/25 on DST days)."""
    start = pd.Timestamp(day).tz_localize(tz)  # local midnight (beginning of HE1)
    end = pd.Timestamp(day + dt.timedelta(days=1)).tz_localize(tz)
    return pd.date_range(start + pd.Timedelta(hours=1), end, freq="h").tz_convert(UTC)


def remaining_day_hours(issue_utc, tz: str) -> pd.DatetimeIndex:
    """Target hours h > issue time through the end of the issue's operating day.

    Issue time tau is an instant; the first target is the next full hour-ending after it.
    """
    tau = to_utc(issue_utc)
    day = (tau.tz_convert(tz)).date()
    hours = operating_day_hours(day, tz)
    return hours[hours > tau]


def horizon_hours(issue_utc, target_he_utc) -> int:
    """Integer horizon in hours, ceil((target - issue)/1h)."""
    delta = to_utc(target_he_utc) - to_utc(issue_utc)
    return int(-(-delta // pd.Timedelta(hours=1)))
