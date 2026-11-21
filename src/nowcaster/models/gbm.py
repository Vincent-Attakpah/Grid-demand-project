"""Gradient-boosted models M2 and M3 (plan v2 section 8.2).

M2 (direct): LightGBM predicts demand itself from every as-of feature, including the DA
    forecast when it is posted (NaN before; LightGBM routes missing values natively).
M3 (residual on B4b): LightGBM predicts the error left after B4b ("Trick 2": DA + latest
    error + same-hour-yesterday error); the forecast is B4b + predicted correction. It
    exists only once the day's DA is posted. B4b is refit inside the same training window,
    so M3 can only add to what B4b already captures.

Both follow the (fit, predict) contract of baselines.simple, so rolling_origin refits them
on labels published before each origin only. Hyperparameters are fixed in advance (no
tuning on the evaluation months); the number of trees is chosen by early stopping on the
last 28 days of each training window, then the model is refit on the whole window.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..baselines.simple import b4b_fit, b4b_predict

CALENDAR = ["hour", "dow", "month", "holiday", "bridge", "days_to_xmas", "days_to_newyear",
            "xmas_window", "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
RECENT = ["y_last", "ramp3", "lag24", "lag168", "e_last", "e_sameh_prev", "da_err_today",
          "da_err_7d"]
WEATHER = ["wx_temperature_2m", "wx_dew_point_2m", "wx_wind_speed_10m", "wx_cloud_cover",
           "wx_shortwave_radiation", "wx_relative_humidity_2m", "hdh", "cdh", "wx_offset",
           "temp_mean_3h", "temp_mean_6h", "temp_mean_24h", "temp_mean_48h"]
BASE = ["horizon"] + CALENDAR + RECENT + WEATHER

PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              seed=7, deterministic=True, force_row_wise=True, num_threads=4, verbose=-1)
MAX_ROUNDS = 2000
VALID_DAYS = 28
MIN_ROWS = 24 * 90


def _cols(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    return [c for c in wanted if c in df.columns]


def _X(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].astype(float).to_numpy()


def _train(df: pd.DataFrame, cols: list[str], y: np.ndarray, params=PARAMS):
    """Early-stop on the last VALID_DAYS of targets, then refit on everything."""
    t = pd.DatetimeIndex(df["target_utc"])
    cut = t.max() - pd.Timedelta(days=VALID_DAYS)
    tr, va = np.asarray(t <= cut), np.asarray(t > cut)
    X = _X(df, cols)
    n_rounds = 300
    if tr.sum() >= MIN_ROWS and va.sum() >= 24 * 7:
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), MAX_ROUNDS,
                      valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        n_rounds = max(50, m.best_iteration or MAX_ROUNDS)
    return lgb.train(params, lgb.Dataset(X, y), n_rounds)


# ---- M2: direct -------------------------------------------------------------------
M2_COLS = BASE + ["da"]


def m2_fit(train: pd.DataFrame):
    t = train.dropna(subset=["actual"])
    if len(t) < MIN_ROWS:
        return None
    cols = _cols(t, M2_COLS)
    return cols, _train(t, cols, t["actual"].to_numpy(float))


def m2_predict(fitted, p: pd.DataFrame) -> np.ndarray:
    if fitted is None:
        return np.full(len(p), np.nan)
    cols, model = fitted
    return model.predict(_X(p, cols))


# ---- M3: residual on B4b ----------------------------------------------------------
M3_COLS = BASE + ["da"]


def m3_fit(train: pd.DataFrame, wanted: list[str] = M3_COLS):
    coefs = b4b_fit(train)
    t = train[train["da_posted"].astype(bool)].dropna(subset=["actual", "da"]).copy()
    if not coefs or len(t) < MIN_ROWS:
        return None
    t["_base"] = b4b_predict(coefs, t)
    t = t.dropna(subset=["_base"])
    cols = _cols(t, wanted)
    return coefs, cols, _train(t, cols, (t["actual"] - t["_base"]).to_numpy(float))


def m3_predict(fitted, p: pd.DataFrame) -> np.ndarray:
    if fitted is None:
        return np.full(len(p), np.nan)
    coefs, cols, model = fitted
    base = b4b_predict(coefs, p)
    out = base + model.predict(_X(p, cols))
    return np.where(np.isfinite(base), out, np.nan)


MODELS = {"M2": (m2_fit, m2_predict), "M3": (m3_fit, m3_predict)}

# Ablations: M3 with inputs removed, everything else identical (same rows, settings, seed).
M3_NO_WEATHER = [c for c in M3_COLS if c not in WEATHER]
M3_DEMAND_ONLY = [c for c in M3_NO_WEATHER if c not in CALENDAR]
ABLATIONS = {
    "M3_noWx": (lambda t: m3_fit(t, M3_NO_WEATHER), m3_predict),
    "M3_demandOnly": (lambda t: m3_fit(t, M3_DEMAND_ONLY), m3_predict),
}
