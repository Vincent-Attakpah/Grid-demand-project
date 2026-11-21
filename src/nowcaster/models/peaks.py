"""Peak forecasts and high-peak warnings (plan v2 section 8.4, M5).

Target, for each issue time: the highest demand over the remaining hours of the operating
day (the panel's target hours). Hours already past are not included: by the time the DA is
posted (10:00 ET) a winter morning peak is usually over, so this measures what is still ahead.

Method (analog error paths): a point forecast for each remaining hour is turned into an
ensemble by adding whole-day error paths from earlier days. For an issue on day d at local
hour h, the library is the issues on days d-LIB_DAYS .. d-1 at the same local hour with the
same number of remaining hours; each library day contributes its vector of errors
(actual - forecast) across those hours. Adding whole paths keeps the hour-to-hour
correlation of errors, which is what decides how high the peak gets. Every library day is
fully published before the issue (its last hour + 3 h is before 03:00 of day d).

  M5 = M3 point forecasts + M3 error paths
  R  = B4b (Trick 2) point forecasts + B4b error paths   (reference: same method, Trick 2)

High-peak event: remaining peak > the 90th percentile of daily peaks in that calendar month
over the two calendar years before the issue's year (refreshed every January, so demand
growth does not make the line stale).

Calibration (`inflation_factors`): the raw ensembles were a little too narrow (90% ranges
caught 85-88%). At each month start, a stretch factor c is chosen from the previous
CAL_DAYS days of issues (all fully published) so that their 5-95% ranges would have caught
90%; every ensemble in the month is then stretched around its median by c. The same rule
is applied to M5 and R.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

LIB_DAYS = 60
MIN_LIB = 20
H = 25
MIN_ACTUAL_SHARE = 0.8
CAL_DAYS = 90
GRID = np.round(np.arange(0.8, 3.01, 0.05), 2)


def issue_arrays(pred: pd.DataFrame, cols: list[str], tz: str) -> tuple[pd.DataFrame, dict]:
    """One row per issue; arrays [n_issue, H] by horizon (NaN where the hour is not in the day)."""
    p = pred[pred["da_posted"]].copy()
    meta = (p.groupby("issue_utc").agg(op_day=("op_day", "first"), K=("horizon", "size"))
            .reset_index().sort_values("issue_utc").reset_index(drop=True))
    meta["op_day"] = pd.to_datetime(meta["op_day"])
    meta["local_hour"] = pd.DatetimeIndex(meta["issue_utc"]).tz_convert(tz).hour
    pos = pd.Series(np.arange(len(meta)), index=meta["issue_utc"])
    r = pos.reindex(p["issue_utc"]).to_numpy()
    c = p["horizon"].to_numpy() - 1
    arrs = {}
    for col in cols + ["actual"]:
        a = np.full((len(meta), H), np.nan)
        a[r, c] = p[col].to_numpy(float)
        arrs[col] = a
    present = np.zeros((len(meta), H), bool)
    present[r, c] = True
    arrs["present"] = present
    return meta, arrs


def actual_peak(arrs) -> np.ndarray:
    a, pres = arrs["actual"], arrs["present"]
    share = np.isfinite(a).sum(1) / np.maximum(pres.sum(1), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pk = np.nanmax(np.where(pres, a, np.nan), axis=1)
    return np.where(share >= MIN_ACTUAL_SHARE, pk, np.nan)


def peak_ensembles(meta: pd.DataFrame, arrs: dict, model: str) -> list[np.ndarray | None]:
    """Per issue: array of ensemble peaks (one per library day), or None."""
    f, a, pres = arrs[model], arrs["actual"], arrs["present"]
    err = np.nan_to_num(a - f, nan=0.0)          # missing actual: that hour uses the point forecast
    out: list[np.ndarray | None] = [None] * len(meta)
    days = meta["op_day"].to_numpy()
    for (hr, K), g in meta.groupby(["local_hour", "K"]):
        idx = g.index.to_numpy()
        d = days[idx]
        order = np.argsort(d)
        idx, d = idx[order], d[order]
        for j, i in enumerate(idx):
            lo = d[j] - np.timedelta64(LIB_DAYS, "D")
            lib = idx[:j][(d[:j] >= lo) & (d[:j] < d[j])]
            if len(lib) < MIN_LIB or not np.isfinite(f[i][pres[i]]).all():
                continue
            paths = f[i][None, :] + err[lib]                 # [n_lib, H]
            out[i] = np.max(np.where(pres[i][None, :], paths, -np.inf), axis=1)
    return out


def monthly_thresholds(daily_peaks: pd.Series, q: float = 0.9) -> pd.Series:
    """daily_peaks indexed by date -> threshold per calendar month (1..12)."""
    s = daily_peaks.dropna()
    return s.groupby(pd.DatetimeIndex(s.index).month).quantile(q)


def yearly_thresholds(daily_peaks: pd.Series, years, q: float = 0.9, back: int = 2) -> pd.DataFrame:
    """Threshold table: index = year Y, columns = month; from daily peaks in Y-back .. Y-1."""
    s = daily_peaks.dropna()
    yr = pd.DatetimeIndex(s.index).year
    rows = {}
    for y in years:
        sel = s[(yr >= y - back) & (yr < y)]
        rows[y] = monthly_thresholds(sel, q).reindex(range(1, 13))
    return pd.DataFrame(rows).T


def threshold_for(op_day: pd.Series, thresholds) -> np.ndarray:
    """Per-issue threshold from a month Series or a (year x month) table."""
    d = pd.DatetimeIndex(op_day)
    if isinstance(thresholds, pd.DataFrame):
        t = thresholds.reindex(index=d.year)
        return np.array([t.iat[i, t.columns.get_loc(m)] if m in t.columns else np.nan
                         for i, m in enumerate(d.month)], float)
    return thresholds.reindex(d.month).to_numpy(float)


def best_factor(ens, idx, actual, target=0.9) -> float:
    """Smallest stretch in GRID whose past coverage reaches the target (1.0 if too few issues)."""
    idx = [i for i in idx if ens[i] is not None and np.isfinite(actual[i])]
    if len(idx) < 200:
        return 1.0
    s = [ens[i] for i in idx]
    a = actual[idx]
    med = np.array([np.median(x) for x in s])
    q = np.array([np.quantile(x, [0.05, 0.95]) for x in s])
    for c in GRID:
        lo, hi = med + c * (q[:, 0] - med), med + c * (q[:, 1] - med)
        if np.mean((a >= lo) & (a <= hi)) >= target:
            return float(c)
    return float(GRID[-1])


def inflation_factors(meta: pd.DataFrame, ens: list, actual: np.ndarray, days: int = CAL_DAYS) -> np.ndarray:
    """Per issue: the stretch factor chosen at its month start from the previous `days` days."""
    od = pd.DatetimeIndex(meta["op_day"])
    month = od.to_period("M")
    out = np.ones(len(meta))
    for mon in month.unique():
        m0 = mon.to_timestamp()
        past = np.flatnonzero((od >= m0 - pd.Timedelta(days=days)) & (od < m0))
        out[np.asarray(month == mon)] = best_factor(ens, past, actual)
    return out


def stretch(ens: list, c: np.ndarray) -> list:
    return [None if s is None else np.median(s) + c[i] * (s - np.median(s)) for i, s in enumerate(ens)]


def summarize(meta, arrs, ens: dict[str, list], thresholds: pd.Series) -> pd.DataFrame:
    """Per issue: actual peak, point/median peaks, 5-95% range, P(exceed), event outcome."""
    out = meta.copy()
    out["peak_actual"] = actual_peak(arrs)
    pres = arrs["present"]
    for col in ("da", "B4b", "M3"):
        if col in arrs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[f"peak_point_{col}"] = np.nanmax(np.where(pres, arrs[col], np.nan), axis=1)
    out["threshold"] = threshold_for(out["op_day"], thresholds)
    for name, e in ens.items():
        med, lo, hi, pr = (np.full(len(out), np.nan) for _ in range(4))
        for i, s in enumerate(e):
            if s is None:
                continue
            med[i], lo[i], hi[i] = np.quantile(s, [0.5, 0.05, 0.95])
            pr[i] = np.mean(s > out["threshold"].iat[i])
        out[f"peak_{name}"], out[f"lo_{name}"], out[f"hi_{name}"], out[f"p_{name}"] = med, lo, hi, pr
    out["event"] = np.where(out["peak_actual"].notna(), out["peak_actual"] > out["threshold"], np.nan)
    return out
