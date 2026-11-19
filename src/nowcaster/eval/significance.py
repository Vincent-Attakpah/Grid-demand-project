"""Significance tests and resampling intervals (plan v2 section 9.3).

Diebold-Mariano
---------------
`dm_test(loss_a, loss_b, h)` tests H0: E[loss_a - loss_b] = 0 on one series of loss
differentials, using a Newey-West (Bartlett) long-run variance with lag h-1 by default (the
MA(h-1) structure of optimal h-step errors) and the Harvey-Leybourne-Newbold (1997)
small-sample correction with Student-t(T-1) reference. Negative statistic = model a better.

Headline test: `dm_daily`
-------------------------
The nowcast panel has many rows per day (every issue hour x every remaining target hour),
and those rows share the same weather and demand shocks, so treating them as independent
grossly overstates significance. `dm_daily` therefore averages the loss differential per
block (default: operating day of issue) and runs DM on that daily series with h = 1 and a
Newey-West lag from the automatic rule floor(4 * (T/100)^(2/9)) (at least 1), because daily
error differences are themselves persistent (weather regimes last several days). This is
the recommended headline test per BA x horizon bucket.

Holm
----
`holm` adjusts the ~32 (BA x bucket x reference) p-values for family-wise error.

Block bootstrap
---------------
`block_bootstrap_ci` resamples whole days (or runs of days) to keep within-day dependence,
and returns a percentile interval for any statistic computed from a DataFrame.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

LOSSES = {
    "abs": lambda e: np.abs(e),
    "sq": lambda e: e ** 2,
}


def newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Long-run variance of a series (Bartlett kernel): gamma0 + 2 sum (1 - k/(L+1)) gamma_k."""
    d = np.asarray(d, float)
    T = len(d)
    u = d - d.mean()
    lrv = u @ u / T
    for k in range(1, min(lag, T - 1) + 1):
        w = 1 - k / (lag + 1)
        lrv += 2 * w * (u[k:] @ u[:-k]) / T
    return float(lrv)


def auto_lag(T: int) -> int:
    return max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))


def dm_test(loss_a, loss_b, h: int = 1, lag: int | None = None, hln: bool = True) -> dict:
    """Diebold-Mariano test on aligned loss series. Returns stat, pvalue (two-sided),
    mean_diff (a - b), T, lag. NaN pairs are dropped."""
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    T = len(d)
    lag = max(h - 1, 0) if lag is None else lag
    out = {"mean_diff": float(d.mean()) if T else np.nan, "T": T, "lag": lag,
           "stat": np.nan, "pvalue": np.nan}
    if T < 3:
        return out
    lrv = newey_west_lrv(d, lag)
    if not np.isfinite(lrv) or lrv <= 0:
        return out
    stat = d.mean() / np.sqrt(lrv / T)
    if hln:
        factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        stat *= factor
        p = 2 * stats.t.sf(abs(stat), df=T - 1)
    else:
        p = 2 * stats.norm.sf(abs(stat))
    out.update(stat=float(stat), pvalue=float(p))
    return out


def dm_daily(df: pd.DataFrame, a: str, b: str, actual: str = "actual",
             block: str | None = None, loss: str = "abs", lag: int | None = None) -> dict:
    """DM on per-block mean loss differentials (headline test).

    `block` names a column identifying the day (e.g. 'op_day'); if None, uses 'op_day' when
    present, else the UTC date of 'issue_utc'. Rows missing the actual or either prediction
    are dropped. Lag defaults to auto_lag(number of days).
    """
    d = df.dropna(subset=[actual, a, b])
    if block is None:
        block = "op_day" if "op_day" in d.columns else None
    keys = d[block] if block else pd.to_datetime(d["issue_utc"], utc=True).dt.floor("D")
    f = LOSSES[loss]
    diff = f(d[actual].to_numpy(float) - d[a].to_numpy(float)) - \
        f(d[actual].to_numpy(float) - d[b].to_numpy(float))
    daily = pd.Series(diff, index=d.index).groupby(keys.to_numpy()).mean().sort_index()
    T = len(daily)
    res = dm_test(daily.to_numpy(), np.zeros(T), h=1, lag=auto_lag(T) if lag is None else lag)
    res["n_rows"] = len(d)
    res["n_days"] = T
    return res


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as input; NaN stays NaN)."""
    p = np.asarray(pvalues, float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    adj = np.maximum.accumulate([(m - i) * p[j] for i, j in enumerate(order)])
    out[order] = np.minimum(adj, 1.0)
    return out


def block_bootstrap_ci(df: pd.DataFrame, stat_fn: Callable[[pd.DataFrame], float],
                       block_col: str, n_boot: int = 1000, alpha: float = 0.05,
                       block_len: int = 1, method: str = "moving", seed: int = 0) -> dict:
    """Percentile CI for stat_fn(df) resampling blocks identified by `block_col` (days).

    method='moving': runs of `block_len` consecutive blocks with random starts (circular).
    method='stationary': Politis-Romano, geometric run lengths with mean `block_len`.
    block_len=1 with either method is the plain i.i.d. day bootstrap.
    """
    rng = np.random.default_rng(seed)
    keys = np.sort(df[block_col].unique())
    K = len(keys)
    groups = df.groupby(block_col).indices
    idx_by_block = [groups[k] for k in keys]
    est = float(stat_fn(df))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked: list[int] = []
        while len(picked) < K:
            start = int(rng.integers(K))
            L = block_len if method == "moving" else int(rng.geometric(1 / max(block_len, 1)))
            picked.extend(((start + np.arange(L)) % K).tolist())
        rows = np.concatenate([idx_by_block[j] for j in picked[:K]])
        draws[i] = stat_fn(df.iloc[rows])
    lo, hi = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"estimate": est, "lo": float(lo), "hi": float(hi), "n_blocks": K,
            "n_boot": n_boot, "se": float(np.nanstd(draws, ddof=1))}
