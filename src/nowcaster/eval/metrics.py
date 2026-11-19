"""Point, probabilistic and event metrics (plan v2 section 9.2).

All functions are NaN-aware: a row with a missing actual or prediction is ignored. When
models are compared (`skill`, `score_table`) the rows are aligned first so every model is
scored on exactly the same (issue, target) pairs.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

HORIZON_BUCKETS = [(1, 3, "1-3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 24, "13-24")]
BUCKET_ORDER = [b[2] for b in HORIZON_BUCKETS] + ["25+"]


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _mask(*xs) -> np.ndarray:
    m = np.ones(len(_arr(xs[0])), dtype=bool)
    for x in xs:
        m &= np.isfinite(_arr(x))
    return m


def horizon_bucket(h) -> pd.Series:
    """Map integer horizons (hours) to the plan's buckets: 1-3, 4-6, 7-12, 13-24, 25+."""
    h = pd.Series(h).astype(float)
    out = pd.Series(pd.NA, index=h.index, dtype="object")
    for lo, hi, lab in HORIZON_BUCKETS:
        out[(h >= lo) & (h <= hi)] = lab
    out[h > 24] = "25+"
    return pd.Categorical(out, categories=BUCKET_ORDER, ordered=True)


# ---------------------------------------------------------------- point metrics
def mae(y, yhat) -> float:
    y, yhat = _arr(y), _arr(yhat)
    m = _mask(y, yhat)
    return float(np.mean(np.abs(y[m] - yhat[m]))) if m.any() else np.nan


def mape(y, yhat) -> float:
    """Mean absolute percentage error as a fraction (0.025 = 2.5%). Ignores y <= 0."""
    y, yhat = _arr(y), _arr(yhat)
    m = _mask(y, yhat) & (np.nan_to_num(y) > 0)
    return float(np.mean(np.abs(y[m] - yhat[m]) / y[m])) if m.any() else np.nan


def skill(y, yhat, yref) -> float:
    """1 - MAE(model)/MAE(reference), on rows where all three are present."""
    y, yhat, yref = _arr(y), _arr(yhat), _arr(yref)
    m = _mask(y, yhat, yref)
    if not m.any():
        return np.nan
    ref = np.mean(np.abs(y[m] - yref[m]))
    return float(1 - np.mean(np.abs(y[m] - yhat[m])) / ref) if ref > 0 else np.nan


# -------------------------------------------------------- probabilistic metrics
def pinball(y, q_pred, tau: float) -> float:
    """Mean pinball (quantile) loss for quantile level tau."""
    y, q = _arr(y), _arr(q_pred)
    m = _mask(y, q)
    if not m.any():
        return np.nan
    u = y[m] - q[m]
    return float(np.mean(np.maximum(tau * u, (tau - 1) * u)))


def pinball_table(y, quantiles: dict[float, Sequence[float]]) -> pd.Series:
    """Pinball loss per quantile level, plus their mean ('mean')."""
    s = pd.Series({tau: pinball(y, q, tau) for tau, q in sorted(quantiles.items())})
    s["mean"] = s.mean()
    return s


def coverage(y, lo, hi) -> float:
    """Share of finite rows with lo <= y <= hi."""
    y, lo, hi = _arr(y), _arr(lo), _arr(hi)
    m = _mask(y, lo, hi)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))) if m.any() else np.nan


def mean_width(lo, hi) -> float:
    lo, hi = _arr(lo), _arr(hi)
    m = _mask(lo, hi)
    return float(np.mean(hi[m] - lo[m])) if m.any() else np.nan


# ---------------------------------------------------------------- event metrics
def brier(y, p) -> float:
    """Brier score for binary outcomes y in {0,1} and probabilities p."""
    y, p = _arr(y), _arr(p)
    m = _mask(y, p)
    return float(np.mean((p[m] - y[m]) ** 2)) if m.any() else np.nan


def brier_skill(y, p, base_rate) -> float:
    """1 - BS(model)/BS(reference). `base_rate` is a scalar or per-row climatology
    (e.g. the BA's seasonal base rate from the calibration year)."""
    y, p = _arr(y), _arr(p)
    base = np.broadcast_to(_arr(base_rate), y.shape)
    m = _mask(y, p, base)
    if not m.any():
        return np.nan
    ref = np.mean((base[m] - y[m]) ** 2)
    return float(1 - np.mean((p[m] - y[m]) ** 2) / ref) if ref > 0 else np.nan


def reliability_bins(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Reliability-diagram table: equal-width probability bins with mean forecast,
    observed frequency and count. Empty bins are kept with n = 0."""
    y, p = _arr(y), _arr(p)
    m = _mask(y, p)
    y, p = y[m], np.clip(p[m], 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": n,
                     "mean_p": float(p[sel].mean()) if n else np.nan,
                     "obs_freq": float(y[sel].mean()) if n else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- score table
def score_table(df: pd.DataFrame, pred_cols: Iterable[str], ref_cols: Iterable[str] = ("B3", "B4"),
                by: Sequence[str] = ("ba", "bucket"), actual: str = "actual",
                horizon: str = "horizon") -> pd.DataFrame:
    """MAE / MAPE / skill vs each reference, per group, on identical rows.

    `df` is a long panel with columns ba, issue_utc, target_utc, horizon, actual and one
    column per model. Rows where the actual or ANY compared column is missing are dropped
    before grouping, so every model in a group is scored on the same rows. A `bucket`
    column is derived from `horizon` if `by` asks for it and it is absent.

    Returns one row per (group, model): n, mae, mape, skill_vs_<ref> for each ref.
    """
    pred_cols, ref_cols = list(pred_cols), list(ref_cols)
    models = list(dict.fromkeys(pred_cols + ref_cols))
    d = df.copy()
    if "bucket" in by and "bucket" not in d.columns:
        d["bucket"] = horizon_bucket(d[horizon].to_numpy())
    d = d.dropna(subset=[actual, *models])
    rows = []
    for key, g in d.groupby(list(by), observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        y = g[actual].to_numpy(float)
        for mdl in models:
            r = dict(zip(by, key))
            r.update(model=mdl, n=len(g), mae=mae(y, g[mdl]), mape=mape(y, g[mdl]))
            for ref in ref_cols:
                r[f"skill_vs_{ref}"] = skill(y, g[mdl], g[ref])
            rows.append(r)
    return pd.DataFrame(rows)
