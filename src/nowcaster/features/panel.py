"""As-of issue-time panel (plan v2 section 7.1). One code path for backtest and live.

Each row is (issue time tau, target hour h) with h in the rest of tau's operating day.
Every feature is computed only from values published before tau:

* demand for hour-ending H is visible from H + `demand_lag_h` (default 3 h, bas.yaml);
  the latest visible hour is L = floor(tau - demand_lag_h);
* the DA forecast for operating day D is visible from its posting time P(D)
  (default 10:00 America/New_York on D); before that the whole day's DA is hidden.

The hourly inputs are cleaned first (`clean_hourly`): the target is EIA-adjusted demand,
with implausible hours (<= 0, or DA/demand outside [0.5, 2]) and EIA-imputed hours set
to NaN, because imputed hours copy the DA forecast (reports/audit/audit.md).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import lag_settings
from ..timeutils import to_utc

H = pd.Timedelta(hours=1)
ISSUE_MINUTE = 17          # live runs fire at :17
MAX_HORIZON = 25           # 25-hour fall-back days
FFILL_LIMIT = 6            # hours a missing published value may be carried forward

FEATURE_COLS = ["y_last", "e_last", "e_sameh_prev", "lag24", "lag168", "ramp3",
                "da_err_today", "da_err_7d"]


@dataclass(frozen=True)
class Lags:
    demand_lag_h: int = 3
    da_time: str = "10:00"
    da_tz: str = "America/New_York"

    @classmethod
    def from_config(cls) -> "Lags":
        c = lag_settings()
        return cls(int(c["demand_available_hours_after_hour_end"]),
                   str(c["da_forecast_available_local_time"]), str(c["da_forecast_available_tz"]))


def clean_hourly(eia: pd.DataFrame) -> pd.DataFrame:
    """One BA's eia_hourly rows -> complete hourly frame indexed by UTC hour-ending,
    columns actual, da, err (= actual - da)."""
    df = eia.copy()
    df["period_utc"] = to_utc(df["period_utc"])
    df = df.drop_duplicates("period_utc").set_index("period_utc").sort_index()
    idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(idx)
    act = df["demand_adjusted"].astype(float)
    da = df["da_forecast"].astype(float).where(lambda s: s > 0)
    ratio = da / act
    imputed = (df["demand_imputed"].fillna(False).astype(bool) if "demand_imputed" in df
               else pd.Series(False, index=idx))
    raw_ok = (act > 0) & ~imputed
    # Label: full filter, including the DA ratio rule (scorecard convention).
    label = act.where(raw_ok & ~(da.notna() & ~ratio.between(0.5, 2.0)))
    # Feature input: must not use DA (hidden before posting), so check each hour against
    # the trailing 7-day median of earlier hours instead. Causal by construction.
    med = act.where(raw_ok).rolling(168, min_periods=24).median().shift(1)
    feat = act.where(raw_ok & ~(med.notna() & ~(act / med).between(0.5, 2.0)))
    out = pd.DataFrame({"actual": label, "da": da, "feat": feat}, index=idx)
    out["err"] = out["feat"] - out["da"]
    return out


def op_day(he_utc: pd.DatetimeIndex, tz: str) -> np.ndarray:
    """Local operating date of hour-ending stamps, as datetime64[D]."""
    return ((he_utc.tz_convert(tz) - H).tz_localize(None).normalize()).values.astype("datetime64[D]")


def posting_time(days: np.ndarray, lags: Lags) -> pd.DatetimeIndex:
    """UTC instant at which the DA forecast for each operating day becomes visible."""
    hh, mm = (int(x) for x in lags.da_time.split(":"))
    local = pd.DatetimeIndex(days.astype("datetime64[ns]")) + pd.Timedelta(hours=hh, minutes=mm)
    return local.tz_localize(lags.da_tz).tz_convert("UTC")


def day_start_he(days: np.ndarray, tz: str) -> pd.DatetimeIndex:
    """UTC hour-ending stamp of local midnight at the start of each day (= HE24 of the day before)."""
    return pd.DatetimeIndex(days.astype("datetime64[ns]")).tz_localize(tz).tz_convert("UTC")


def issue_times(start, end) -> pd.DatetimeIndex:
    s, e = to_utc(start).floor("h"), to_utc(end).floor("h")
    return pd.date_range(s, e, freq="h", tz="UTC") + pd.Timedelta(minutes=ISSUE_MINUTE)


def _lookup(series: pd.Series, times) -> np.ndarray:
    return series.reindex(pd.DatetimeIndex(times)).to_numpy(dtype=float)


def build_panel(hourly: pd.DataFrame, tz: str, issues: pd.DatetimeIndex, lags: Lags | None = None,
                ba: str = "") -> pd.DataFrame:
    """Build the (issue, target) panel for one BA from `clean_hourly` output.

    `hourly` may contain values after the issue times (backtest); none of them reach the
    features. `actual` (the label) is the only column that looks at target hours.
    """
    lags = lags or Lags.from_config()
    label, act, da, err = hourly["actual"], hourly["feat"], hourly["da"], hourly["err"]
    act_ff = act.ffill(limit=FFILL_LIMIT)

    # Within-day cumulative mean DA error, and trailing 7-day mean.
    od_all = op_day(hourly.index, tz)
    err_cum = err.groupby(od_all).transform(lambda s: s.expanding().mean())
    err_7d = err.rolling(168, min_periods=100).mean()
    err_ff = err.ffill(limit=FFILL_LIMIT)

    # --- issue-level quantities ---
    tau = pd.DatetimeIndex(issues)
    tau_day = (tau.tz_convert(tz).tz_localize(None).normalize()).values.astype("datetime64[D]")
    L = (tau - pd.Timedelta(hours=lags.demand_lag_h)).floor("h")
    L_day = op_day(L, tz)
    posted_L = tau >= posting_time(L_day, lags)
    # Latest hour whose DA error is visible: L if its day's DA is posted, else end of the day before.
    Lda = pd.DatetimeIndex(np.where(posted_L, L, day_start_he(L_day, tz)))
    posted_today = tau >= posting_time(tau_day, lags)

    issue = pd.DataFrame({
        "issue_utc": tau, "op_day": tau_day, "last_obs_utc": L,
        "y_last": _lookup(act_ff, L),
        "ramp3": _lookup(act_ff, L) - _lookup(act_ff, L - 3 * H),
        "e_last": _lookup(err_ff, Lda),
        "da_err_7d": _lookup(err_7d, Lda),
        "da_err_today": np.where(posted_L & (L_day == tau_day), _lookup(err_cum, L), np.nan),
        "da_posted": posted_today,
    })

    # --- expand to targets ---
    k = np.arange(1, MAX_HORIZON + 1)
    rep = np.repeat(np.arange(len(issue)), len(k))
    panel = issue.iloc[rep].reset_index(drop=True)
    panel["horizon"] = np.tile(k, len(issue))
    target = pd.DatetimeIndex(panel["issue_utc"]).floor("h") + pd.to_timedelta(panel["horizon"].to_numpy(), unit="h")
    tgt_day = op_day(target, tz)
    keep = tgt_day == panel["op_day"].to_numpy()
    panel = panel[keep].reset_index(drop=True)
    target = target[keep]
    panel.insert(1, "target_utc", target)
    Lp = pd.DatetimeIndex(panel["last_obs_utc"])

    # Same-hour-yesterday quantities: h-24 if visible, else h-48.
    back = np.where(target - 24 * H <= Lp, 24, 48)
    t_back = target - pd.to_timedelta(back, unit="h")
    panel["lag24"] = _lookup(act, t_back)
    panel["e_sameh_prev"] = _lookup(err, t_back)
    panel["lag168"] = _lookup(act, target - 168 * H)

    # DA for the target hour only once that day's forecast is posted.
    panel["da"] = np.where(panel["da_posted"].to_numpy(), _lookup(da, target), np.nan)
    panel["actual"] = _lookup(label, target)
    panel.insert(0, "ba", ba)
    return panel


def build_panel_for_ba(eia: pd.DataFrame, ba: str, tz: str, start, end, lags: Lags | None = None) -> pd.DataFrame:
    hourly = clean_hourly(eia[eia["ba"] == ba] if "ba" in eia else eia)
    return build_panel(hourly, tz, issue_times(start, end), lags, ba=ba)
