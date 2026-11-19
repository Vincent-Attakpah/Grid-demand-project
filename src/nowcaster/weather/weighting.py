"""Population-weighted BA weather from per-city rows (plan v2 section 6.1).

For each BA, source and hour, weights are the populations of the cities active at that
hour (active_from <= hour < active_until), renormalised per variable over the cities with
a non-null value, so one missing city does not bias the average toward zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BA


def city_weights(ba: BA, periods: pd.Series) -> pd.DataFrame:
    """Long frame (period_utc, city, w) of raw population weights for active cities."""
    out = []
    p = pd.DatetimeIndex(periods.unique())
    for c in ba.cities:
        mask = np.ones(len(p), dtype=bool)
        if c.active_from:
            mask &= p >= pd.Timestamp(c.active_from, tz="UTC")
        if c.active_until:
            mask &= p < pd.Timestamp(c.active_until, tz="UTC")
        out.append(pd.DataFrame({"period_utc": p[mask], "city": c.name, "w": float(c.pop)}))
    return pd.concat(out, ignore_index=True)


def weight_ba(city_df: pd.DataFrame, bas: dict[str, BA], variables: list[str]) -> pd.DataFrame:
    frames = []
    for code, g in city_df.groupby("ba"):
        if code not in bas:
            continue
        w = city_weights(bas[code], g["period_utc"])
        m = g.merge(w, on=["period_utc", "city"], how="inner")  # inactive cities drop out
        res = m[["ba", "source", "period_utc"]].drop_duplicates()
        keys = ["ba", "source", "period_utc"]
        agg = None
        for v in variables:
            ok = m[v].notna()
            num = (m[v].where(ok, 0) * m["w"].where(ok, 0)).groupby([m[k] for k in keys]).sum()
            den = m["w"].where(ok, 0).groupby([m[k] for k in keys]).sum()
            s = (num / den.replace(0, np.nan)).rename(v)
            agg = s.to_frame() if agg is None else agg.join(s)
        n = m.groupby(keys)["city"].nunique().rename("n_cities")
        agg = agg.join(n).reset_index()
        frames.append(res.merge(agg, on=keys, how="left"))
    if not frames:
        return pd.DataFrame(columns=["ba", "source", "period_utc", *variables, "n_cities"])
    return pd.concat(frames, ignore_index=True).sort_values(["ba", "source", "period_utc"]).reset_index(drop=True)
