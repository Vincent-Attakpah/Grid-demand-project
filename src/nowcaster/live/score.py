"""Live scoring: how the forecasts issued so far did once actual demand was published.

Same measures as the backtest: average miss (MAE, MW) of our forecast (M3), Trick 2 (B4b)
and the utility's day-ahead forecast on the same hours; how often the 90% and 50% ranges
caught the actual; and, for peaks, the miss on the rest-of-day peak and the warning
outcomes (a warning = chance >= 50%; an issue is scored only once every remaining hour of
its day has been published).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {"last_30_days": 30, "since_start": None}
BUCKETS = {"1-6 h ahead": (1, 6), "all hours": (1, 25)}


def _f(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def hourly_scores(hist: pd.DataFrame, actual: pd.Series, now: pd.Timestamp) -> dict:
    h = hist.copy()
    h["actual"] = actual.reindex(pd.DatetimeIndex(h["target_utc"])).to_numpy()
    h = h.dropna(subset=["actual", "M3", "B4b", "da"])
    out = {}
    for wname, days in WINDOWS.items():
        g0 = h if days is None else h[pd.DatetimeIndex(h["issue_utc"]) >= now - pd.Timedelta(days=days)]
        out[wname] = {}
        for bname, (lo, hi) in BUCKETS.items():
            g = g0[(g0["horizon"] >= lo) & (g0["horizon"] <= hi)]
            r = {"hours_scored": int(len(g)), "days": int(pd.Series(g["op_day"]).nunique())}
            for m in ("M3", "B4b", "da"):
                r[f"mae_{m}"] = _f((g[m] - g["actual"]).abs().mean()) if len(g) else None
            if len(g) and r["mae_B4b"]:
                r["skill_vs_trick2"] = _f(1 - r["mae_M3"] / r["mae_B4b"])
                r["skill_vs_utility"] = _f(1 - r["mae_M3"] / r["mae_da"])
            ok = g.dropna(subset=["M4_q05", "M4_q95"])
            r["cover90"] = _f(((ok["actual"] >= ok["M4_q05"]) & (ok["actual"] <= ok["M4_q95"])).mean()) if len(ok) else None
            r["cover50"] = _f(((ok["actual"] >= ok["M4_q25"]) & (ok["actual"] <= ok["M4_q75"])).mean()) if len(ok) else None
            out[wname][bname] = r
    return out


def peak_scores(peaks: pd.DataFrame, hist: pd.DataFrame, actual: pd.Series, now: pd.Timestamp) -> dict:
    if peaks is None or peaks.empty:
        return {}
    h = hist[["issue_utc", "target_utc"]].copy()
    h["actual"] = actual.reindex(pd.DatetimeIndex(h["target_utc"])).to_numpy()
    agg = h.groupby("issue_utc")["actual"].agg(["max", "count", "size"])
    p = peaks.merge(agg, left_on="issue_utc", right_index=True, how="left")
    p = p[p["count"] == p["size"]].rename(columns={"max": "peak_actual"})     # day fully published
    out = {}
    for wname, days in WINDOWS.items():
        g = p if days is None else p[pd.DatetimeIndex(p["issue_utc"]) >= now - pd.Timedelta(days=days)]
        if g.empty:
            out[wname] = {"issues_scored": 0}
            continue
        ev = g["peak_actual"] > g["threshold"]
        warn = g["p_M5"] >= 0.5
        out[wname] = {
            "issues_scored": int(len(g)), "days": int(g["op_day"].nunique()),
            "high_days": int(g.loc[ev, "op_day"].nunique()),
            "peak_mae_M5": _f((g["peak_M5"] - g["peak_actual"]).abs().mean()),
            "peak_mae_trick2": _f((g["peak_point_B4b"] - g["peak_actual"]).abs().mean()),
            "peak_mae_utility": _f((g["peak_point_da"] - g["peak_actual"]).abs().mean()),
            "peak_cover90": _f(((g["peak_actual"] >= g["lo_M5"]) & (g["peak_actual"] <= g["hi_M5"])).mean()),
            "brier": _f(((g["p_M5"] - ev.astype(float)) ** 2).mean()),
            "hits": int((warn & ev).sum()), "misses": int((~warn & ev).sum()),
            "false_alarms": int((warn & ~ev).sum()),
        }
    return out
