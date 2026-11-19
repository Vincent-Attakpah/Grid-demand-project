"""EIA API v2 client for electricity/rto/region-data (Form EIA-930 hourly series).

Period convention
-----------------
We request ``frequency=hourly``, for which the API returns ``period`` as ``YYYY-MM-DDTHH``
in UTC (``frequency=local-hourly`` would return local time with an offset suffix, e.g.
``2025-01-15T10-06``). Following the EIA-930 bulk files ("UTC time at end of hour") and the
gridstatus library, ``period`` is treated as the END of the hour, so it maps directly onto
this project's UTC hour-ending convention. This is an inference, not a documented guarantee:
cross-check a few hours against the six-month bulk files (ingest agent) before relying on it.

Types pulled: D (demand), DF (day-ahead demand forecast), NG (net generation),
TI (total interchange).
"""

from __future__ import annotations

import datetime as dt
import os

import requests

from .common import UTC, floor_hour, get_json, iso

BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
DEFAULT_BAS = ("SOCO", "TVA", "DUK", "FPL", "ERCO")
DEFAULT_TYPES = ("D", "DF", "NG", "TI")
PAGE = 5000


def parse_period(period: str) -> dt.datetime:
    """'2025-01-15T16' (UTC, hourly frequency) -> tz-aware UTC hour-ending datetime."""
    if len(period) == 13:  # YYYY-MM-DDTHH
        return dt.datetime.strptime(period, "%Y-%m-%dT%H").replace(tzinfo=UTC)
    # local-hourly style 'YYYY-MM-DDTHH-05' / '+05' — accepted defensively.
    base, sign, off = period[:13], period[13], int(period[14:16])
    t = dt.datetime.strptime(base, "%Y-%m-%dT%H")
    tz = dt.timezone(dt.timedelta(hours=off if sign == "+" else -off))
    return t.replace(tzinfo=tz).astimezone(UTC)


def _params(api_key: str, bas, types, start: dt.datetime, offset: int):
    p = [("api_key", api_key), ("frequency", "hourly"), ("data[0]", "value"),
         ("start", start.strftime("%Y-%m-%dT%H")),
         ("sort[0][column]", "period"), ("sort[0][direction]", "asc"),
         ("offset", str(offset)), ("length", str(PAGE))]
    p += [("facets[respondent][]", b) for b in bas]
    p += [("facets[type][]", t) for t in types]
    return p


def fetch(now: dt.datetime, *, api_key: str | None = None, bas=DEFAULT_BAS, types=DEFAULT_TYPES,
          window_hours: int = 72, session=None) -> list[dict]:
    """Pull the last `window_hours` hours plus anything stamped later than now (no `end`
    is sent, so forward-dated DF values are included). Returns normalised rows."""
    api_key = api_key or os.environ.get("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("EIA_API_KEY is not set")
    session = session or requests.Session()
    start = floor_hour(now) - dt.timedelta(hours=window_hours)
    rows: list[dict] = []
    offset = 0
    while True:
        payload = get_json(session, BASE_URL, _params(api_key, bas, types, start, offset))
        if "response" not in payload:
            raise RuntimeError(f"EIA error: {str(payload)[:300]}")
        resp = payload["response"]
        data = resp.get("data", [])
        for d in data:
            v = d.get("value")
            he = parse_period(d["period"])
            rows.append({
                "ba": d["respondent"],
                "type": d["type"],
                "period_utc": iso(he),
                "value": None if v in (None, "") else float(v),
                "units": d.get("value-units"),
                "is_future": he > now,
            })
        offset += len(data)
        total = int(resp.get("total", 0) or 0)
        if not data or offset >= total:
            break
    return rows


def key_and_sig(row: dict) -> tuple[str, str, str]:
    """(dedup key, value signature, key time) for the latest-value index."""
    return f"{row['ba']}|{row['type']}|{row['period_utc']}", repr(row["value"]), row["period_utc"]
