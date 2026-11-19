"""Baselines B0-B4b (plan v2 section 8.1).

Every baseline reads only panel features, so it obeys the same lag rules as the model.
Each baseline is a (fit, predict) pair: fit(train_panel) -> params; predict(params, panel) -> array.
Fitted baselines (B2, B4, B4b) are refit inside the rolling-origin loop, on data before
the origin only (test_no_global_fit).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---- B0, B1, B3: no fitting --------------------------------------------------------
def _nofit(train):
    return None


def b0_predict(_, p):  # persistence: latest published demand held flat
    return p["y_last"].to_numpy(dtype=float)


def b1_predict(_, p):  # same hour last week
    return p["lag168"].to_numpy(dtype=float)


def b3_predict(_, p):  # operator day-ahead
    return p["da"].to_numpy(dtype=float)


# ---- B4 / B4b: DA plus per-horizon error correction --------------------------------
def _fit_correction(train: pd.DataFrame, xcols: list[str]) -> dict[int, np.ndarray]:
    t = train.dropna(subset=["actual", "da", *xcols])
    coefs = {}
    for k, g in t.groupby("horizon"):
        X = g[xcols].to_numpy(dtype=float)
        y = (g["actual"] - g["da"]).to_numpy(dtype=float)
        if len(g) >= 30:
            coefs[int(k)] = np.linalg.lstsq(X, y, rcond=None)[0]   # no intercept
    return coefs


def _predict_correction(coefs, p: pd.DataFrame, xcols: list[str]) -> np.ndarray:
    if not coefs:
        return np.full(len(p), np.nan)
    ks = np.array(sorted(coefs))
    B = np.vstack([coefs[k] for k in ks])
    # horizons never seen in training use the nearest fitted horizon
    pos = np.clip(np.searchsorted(ks, p["horizon"].to_numpy()), 0, len(ks) - 1)
    beta = B[pos]
    X = p[xcols].to_numpy(dtype=float)
    return p["da"].to_numpy(dtype=float) + np.sum(beta * X, axis=1)


B4_X = ["e_last"]
B4B_X = ["e_last", "e_sameh_prev"]


def b4_fit(train):
    return _fit_correction(train, B4_X)


def b4_predict(c, p):
    return _predict_correction(c, p, B4_X)


def b4b_fit(train):
    return _fit_correction(train, B4B_X)


def b4b_predict(c, p):
    return _predict_correction(c, p, B4B_X)


# ---- B2: Tao's Vanilla Benchmark ---------------------------------------------------
TREND_ORIGIN = pd.Timestamp("2020-01-01", tz="UTC")


def _vanilla_design(p: pd.DataFrame) -> np.ndarray:
    n = len(p)
    hour = p["hour"].to_numpy() - 1           # 0..23 (HE25 folds into 23)
    hour = np.clip(hour, 0, 23)
    month = p["month"].to_numpy() - 1
    dow = p["dow"].to_numpy()
    T = p["wx_temperature_2m"].to_numpy(dtype=float)
    trend = (pd.DatetimeIndex(p["target_utc"]) - TREND_ORIGIN) / pd.Timedelta(days=365.25)
    cols = [np.ones(n), np.asarray(trend, dtype=float)]

    def onehot(idx, size):
        m = np.zeros((n, size))
        m[np.arange(n), idx] = 1.0
        return m

    Hm, Mm, Dm = onehot(hour, 24), onehot(month, 12), onehot(dow, 7)
    mh = (Mm[:, :, None] * Hm[:, None, :]).reshape(n, -1)[:, 1:]
    dh = (Dm[:, :, None] * Hm[:, None, :]).reshape(n, -1)[:, 1:]
    polys = [T, T ** 2, T ** 3]
    th = np.hstack([pt[:, None] * Hm for pt in polys])
    tm = np.hstack([pt[:, None] * Mm for pt in polys])
    return np.column_stack([*cols, mh, dh, th, tm])


def b2_fit(train: pd.DataFrame):
    need = ["actual", "wx_temperature_2m", "hour", "month", "dow"]
    if not set(need) <= set(train.columns):
        return None
    # one row per target hour, using the freshest forecast that was available for it
    t = (train.dropna(subset=need).sort_values("horizon")
         .drop_duplicates("target_utc", keep="first"))
    if len(t) < 24 * 60:
        return None
    X = _vanilla_design(t)
    beta = np.linalg.lstsq(X, t["actual"].to_numpy(dtype=float), rcond=None)[0]
    return beta


def b2_predict(beta, p: pd.DataFrame):
    if beta is None or "wx_temperature_2m" not in p:
        return np.full(len(p), np.nan)
    out = _vanilla_design(p.fillna({"wx_temperature_2m": np.nan})) @ beta
    return np.where(p["wx_temperature_2m"].notna(), out, np.nan)


BASELINES = {
    "B0": (_nofit, b0_predict),
    "B1": (_nofit, b1_predict),
    "B2": (b2_fit, b2_predict),
    "B3": (_nofit, b3_predict),
    "B4": (b4_fit, b4_predict),
    "B4b": (b4b_fit, b4b_predict),
}
