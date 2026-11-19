"""Leakage tests (plan v2 section 7.1). Each perturbs data the issue time must not see and
checks that features are unchanged, plus a control showing the test is not vacuous."""

import numpy as np
import pandas as pd
import pytest

from nowcaster.baselines.simple import BASELINES
from nowcaster.eval.rolling import rolling_origin
from nowcaster.features.panel import FEATURE_COLS, Lags, build_panel, clean_hourly, issue_times
from nowcaster.features.weather_join import attach_weather

TZ = "America/New_York"
LAGS = Lags(demand_lag_h=3, da_time="10:00", da_tz="America/New_York")
RNG = np.random.default_rng(0)


def synth_eia(start="2025-01-01", end="2025-03-05"):
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    t = np.arange(len(idx))
    demand = 20000 + 3000 * np.sin(2 * np.pi * t / 24) + RNG.normal(0, 300, len(idx))
    da = demand + RNG.normal(0, 400, len(idx))
    return pd.DataFrame({"ba": "X", "period_utc": idx, "demand_adjusted": demand.round(),
                         "da_forecast": da.round(), "demand_imputed": False})


def panel_at(eia, tau):
    return build_panel(clean_hourly(eia), TZ, pd.DatetimeIndex([tau]), LAGS, ba="X")


def same(a, b, cols):
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


TAU = pd.Timestamp("2025-02-10 17:17", tz="UTC")          # 12:17 ET, DA posted


def test_demand_lag():
    eia = synth_eia()
    base = panel_at(eia, TAU)
    L = TAU.floor("h") - pd.Timedelta(hours=3)
    assert (pd.DatetimeIndex(base["last_obs_utc"]) <= TAU - pd.Timedelta(hours=3)).all()
    future = eia["period_utc"] > L
    pert = eia.copy()
    pert.loc[future, "demand_adjusted"] *= 1.3
    same(base, panel_at(pert, TAU), FEATURE_COLS + ["da"])
    # control: perturbing the last visible hour must change y_last
    ctrl = eia.copy()
    ctrl.loc[eia["period_utc"] == L, "demand_adjusted"] *= 1.3
    assert not np.allclose(panel_at(ctrl, TAU)["y_last"], base["y_last"])


def test_da_posting():
    eia = synth_eia()
    before = pd.Timestamp("2025-02-10 14:17", tz="UTC")    # 09:17 ET, before 10:00 posting
    base = panel_at(eia, before)
    assert base["da"].isna().all() and not base["da_posted"].any()
    day = (pd.DatetimeIndex(eia["period_utc"]).tz_convert(TZ) - pd.Timedelta(hours=1)).date
    today = pd.Timestamp("2025-02-10").date()
    pert = eia.copy()
    # x3 pushes DA/demand outside [0.5, 2]: hidden DA must not even affect data cleaning
    pert.loc[day == today, "da_forecast"] *= 3.0
    same(base, panel_at(pert, before), FEATURE_COLS + ["da"])
    after = panel_at(eia, pd.Timestamp("2025-02-10 15:17", tz="UTC"))  # 10:17 ET
    assert after["da"].notna().all() and after["da_posted"].all()


def synth_weather(start="2025-01-01", end="2025-03-05"):
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    rows = []
    for src in ("prev_day1", "prev_day2", "hist_forecast", "era5"):
        rows.append(pd.DataFrame({"source": src, "period_utc": idx,
                                  "temperature_2m": 10 + 5 * np.sin(np.arange(len(idx)) / 7)
                                  + RNG.normal(0, 1, len(idx))}))
    return pd.concat(rows, ignore_index=True)


def wx_at(wba, tau, eia):
    return attach_weather(panel_at(eia, tau), wba, availability_h=6, variables=["temperature_2m"])


def test_weather_future_source():
    eia, wba = synth_eia(), synth_weather()
    tau = pd.Timestamp("2025-02-10 06:17", tz="UTC")       # 01:17 ET -> horizons up to 23
    base = wx_at(wba, tau, eia)
    assert base["horizon"].max() >= 19
    assert (base.loc[base["horizon"] <= 18, "wx_offset"] == 1).all()
    assert (base.loc[base["horizon"] > 18, "wx_offset"] == 2).all()
    t = pd.DatetimeIndex(wba["period_utc"])
    pert = wba.copy()
    pert.loc[pert["source"].isin(["hist_forecast", "era5"]) & (t > tau), "temperature_2m"] += 15
    # day-1 values are not yet issued for horizons > 18 h: perturbing them must not matter there
    far = base.loc[base["horizon"] > 18, "target_utc"]
    pert.loc[(pert["source"] == "prev_day1") & t.isin(far), "temperature_2m"] += 15
    same(base, wx_at(pert, tau, eia), ["wx_temperature_2m"])
    ctrl = wba.copy()
    near = base.loc[base["horizon"] <= 18, "target_utc"]
    ctrl.loc[(ctrl["source"] == "prev_day1") & t.isin(near), "temperature_2m"] += 15
    assert not np.allclose(wx_at(ctrl, tau, eia)["wx_temperature_2m"], base["wx_temperature_2m"])


def test_weather_past_delay():
    eia, wba = synth_eia(), synth_weather()
    base = wx_at(wba, TAU, eia)
    anchor = (TAU - pd.Timedelta(hours=6)).floor("h")
    t = pd.DatetimeIndex(wba["period_utc"])
    cols = [c for c in base if c.startswith("temp_mean_")]
    assert cols
    pert = wba.copy()
    pert.loc[(pert["source"] == "hist_forecast") & (t > anchor), "temperature_2m"] += 15
    same(base, wx_at(pert, TAU, eia), cols)
    ctrl = wba.copy()
    ctrl.loc[(ctrl["source"] == "hist_forecast") & (t == anchor), "temperature_2m"] += 15
    assert not np.allclose(wx_at(ctrl, TAU, eia)["temp_mean_3h"], base["temp_mean_3h"])


def test_no_global_fit():
    eia = synth_eia()
    panel = build_panel(clean_hourly(eia), TZ, issue_times("2025-01-08", "2025-03-01"), LAGS, ba="X")
    models = {k: BASELINES[k] for k in ("B4", "B4b")}
    origin = pd.Timestamp("2025-02-01", tz="UTC")
    base = rolling_origin(panel, models, origin, "2025-03-01")
    # perturb every label not yet published at the origin
    pert = panel.copy()
    late = (pd.DatetimeIndex(pert["target_utc"]) + pd.Timedelta(hours=3)) >= origin
    pert.loc[late, "actual"] = pert.loc[late, "actual"] * 1.5
    out = rolling_origin(pert, models, origin, "2025-03-01")
    np.testing.assert_allclose(base["B4"], out["B4"])
    np.testing.assert_allclose(base["B4b"], out["B4b"])


def test_panel_rest_of_day_and_dst():
    eia = synth_eia("2025-10-25", "2025-11-05")
    tau = pd.Timestamp("2025-11-02 04:17", tz="UTC")       # 00:17 EDT on fall-back day
    p = panel_at(eia, tau)
    assert len(p) == 25 and p["horizon"].min() == 1
    assert (pd.DatetimeIndex(p["target_utc"]) > tau).all()
