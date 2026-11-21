"""Forecast ranges (plan v2 section 8.3): M4 quantile model + conformal calibration, and the
simple reference Q0.

M4: LightGBM quantile regressions on the error left after B4b (same inputs as M3), for
    levels 5/25/50/75/95%. Calibration (conformalized quantile regression, CQR): the
    quantile models are fit on the training window minus its last CAL_DAYS days; on those
    held-out days we measure how far the actual falls outside each interval, per horizon
    bucket, and widen (or narrow) the 50% and 90% intervals by the finite-sample
    (1 - miss) quantile of that overshoot. The quantile models are then used as fit.
Q0: B4b plus fixed quantiles of B4b's recent errors (last Q0_DAYS days of training, per
    horizon). The "no model" range a fancy method must beat.

Both return a DataFrame with columns q05, q25, q50, q75, q95 (MW), so rolling_origin
stores them as M4_q05 ... and Q0_q05 ...
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..baselines.simple import b4b_fit, b4b_predict
from .gbm import M3_COLS, _cols, _X

LEVELS = {"q05": 0.05, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q95": 0.95}
PAIRS = {0.90: ("q05", "q95"), 0.50: ("q25", "q75")}
BUCKET_EDGES = [(1, 3), (4, 6), (7, 12), (13, 25)]
CAL_DAYS = 28
Q0_DAYS = 90
MIN_ROWS = 24 * 90
QPARAMS = dict(objective="quantile", learning_rate=0.05, num_leaves=31, min_data_in_leaf=300,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
               seed=11, deterministic=True, force_row_wise=True, num_threads=4, verbose=-1)
ROUNDS = 400


def _bucket(h: np.ndarray) -> np.ndarray:
    out = np.zeros(len(h), int)
    for i, (lo, hi) in enumerate(BUCKET_EDGES):
        out[(h >= lo) & (h <= hi)] = i
    return out


def _empty(n: int) -> pd.DataFrame:
    return pd.DataFrame({k: np.full(n, np.nan) for k in LEVELS})


def _posted(train: pd.DataFrame, coefs) -> pd.DataFrame:
    t = train[train["da_posted"].astype(bool)].dropna(subset=["actual", "da"]).copy()
    t["_base"] = b4b_predict(coefs, t)
    return t.dropna(subset=["_base"])


def _sort(q: pd.DataFrame) -> pd.DataFrame:
    """No crossing: sort each row's quantiles."""
    return pd.DataFrame(np.sort(q[list(LEVELS)].to_numpy(), axis=1), columns=list(LEVELS), index=q.index)


def conformal_shift(y, lo, hi, miss: float) -> float:
    """CQR margin: finite-sample (1-miss) quantile of max(lo - y, y - hi)."""
    s = np.maximum(np.asarray(lo) - y, y - np.asarray(hi))
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return 0.0
    k = min(len(s), int(np.ceil((len(s) + 1) * (1 - miss))))
    return float(np.sort(s)[k - 1])


# ---- M4 --------------------------------------------------------------------------
def m4_fit(train: pd.DataFrame):
    coefs = b4b_fit(train)
    if not coefs:
        return None
    t = _posted(train, coefs)
    if len(t) < MIN_ROWS:
        return None
    cols = _cols(t, M3_COLS)
    tt = pd.DatetimeIndex(t["target_utc"])
    cal_days = min(CAL_DAYS, (tt.max() - tt.min()).days / 4)   # short windows (tests) keep 3/4 to fit
    cal = np.asarray(tt > tt.max() - pd.Timedelta(days=cal_days))
    fit_rows = ~cal
    X = _X(t, cols)
    r = (t["actual"] - t["_base"]).to_numpy(float)
    models = {k: lgb.train({**QPARAMS, "alpha": a}, lgb.Dataset(X[fit_rows], r[fit_rows]), ROUNDS)
              for k, a in LEVELS.items()}
    # calibration on held-out days, per horizon bucket and interval
    qc = _sort(pd.DataFrame({k: m.predict(X[cal]) for k, m in models.items()}))
    bc = _bucket(t["horizon"].to_numpy()[cal])
    shifts = {}
    for cov, (lo, hi) in PAIRS.items():
        for b in range(len(BUCKET_EDGES)):
            m = bc == b
            shifts[(cov, b)] = conformal_shift(r[cal][m], qc[lo].to_numpy()[m], qc[hi].to_numpy()[m],
                                               1 - cov) if m.sum() >= 50 else 0.0
    return coefs, cols, models, shifts


def m4_predict(fitted, p: pd.DataFrame) -> pd.DataFrame:
    if fitted is None:
        return _empty(len(p))
    coefs, cols, models, shifts = fitted
    base = b4b_predict(coefs, p)
    q = _sort(pd.DataFrame({k: m.predict(_X(p, cols)) for k, m in models.items()}))
    b = _bucket(p["horizon"].to_numpy())
    for cov, (lo, hi) in PAIRS.items():
        s = np.array([shifts[(cov, i)] for i in b])
        q[lo] = q[lo] - s
        q[hi] = q[hi] + s
    q = _sort(q)
    out = q.add(base, axis=0)
    out.loc[~np.isfinite(base)] = np.nan
    return out.reset_index(drop=True)


# ---- Q0: fixed error quantiles around B4b ----------------------------------------
def q0_fit(train: pd.DataFrame):
    coefs = b4b_fit(train)
    if not coefs:
        return None
    t = _posted(train, coefs)
    tt = pd.DatetimeIndex(t["target_utc"])
    t = t[np.asarray(tt > tt.max() - pd.Timedelta(days=Q0_DAYS))]
    r = t["actual"] - t["_base"]
    qs = r.groupby(t["horizon"]).quantile(list(LEVELS.values())).unstack()
    qs.columns = list(LEVELS)
    return coefs, qs


def q0_predict(fitted, p: pd.DataFrame) -> pd.DataFrame:
    if fitted is None:
        return _empty(len(p))
    coefs, qs = fitted
    base = b4b_predict(coefs, p)
    h = np.clip(p["horizon"].to_numpy(), qs.index.min(), qs.index.max())
    q = qs.reindex(h).ffill().bfill().reset_index(drop=True)
    out = q.add(base, axis=0)
    out.loc[~np.isfinite(base)] = np.nan
    return out


INTERVAL_MODELS = {"M4": (m4_fit, m4_predict), "Q0": (q0_fit, q0_predict)}
