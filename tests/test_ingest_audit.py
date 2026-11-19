import numpy as np
import pandas as pd
import pytest

from nowcaster.ingest.audit import (audit_ba, best_lag, best_single_break, daily_bias, diff_spikes,
                                    flag_all, lag_correlations, level_spikes, verdict)

TZ = "America/Chicago"


def synth(days=400, seed=0, start="2024-01-01 06:00"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=24 * days, freq="h", tz="UTC")
    h = idx.tz_convert(TZ).hour.values
    doy = idx.dayofyear.values
    demand = 20000 + 3000 * np.sin(2 * np.pi * (h - 6) / 24) + 2000 * np.cos(2 * np.pi * doy / 365) \
        + rng.normal(0, 150, len(idx))
    da = demand + rng.normal(0, 300, len(idx))
    return pd.DataFrame({"ba": "X", "period_utc": idx, "demand": demand, "demand_adjusted": demand,
                         "da_forecast": da, "demand_imputed": False})


def test_level_spike_detected_and_isolated():
    df = synth(30)
    s = df["demand"].copy()
    s.iloc[300] = 9_971_121
    flags = level_spikes(s)
    assert flags.iloc[300]
    assert flags.sum() == 1


def test_diff_spike_requires_reversal():
    df = synth(30)
    s = df["demand"].copy()
    s.iloc[200] += 15000          # isolated spike: up then down
    s.iloc[400:] += 15000         # a level step: up, no reversal
    f = diff_spikes(s)
    assert f.iloc[200]
    assert not f.iloc[400]


@pytest.mark.parametrize("shift", [-1, 1, 2])
def test_hour_shift_detected(shift):
    df = synth(60).set_index("period_utc")
    da_shifted = df["da_forecast"].shift(-shift)  # DA(t) = forecast for t+shift
    corrs = lag_correlations(da_shifted, df["demand"])
    assert best_lag(corrs) == shift


def test_aligned_forecast_lag_zero():
    df = synth(60).set_index("period_utc")
    assert best_lag(lag_correlations(df["da_forecast"], df["demand"])) == 0


def test_best_lag_margin_ties_go_to_zero():
    assert best_lag({-1: 0.80, 0: 0.90, 1: 0.895}) == 0
    assert best_lag({-1: 0.80, 0: 0.90, 1: 0.95}) == 1


def test_daily_bias_sign_and_operating_day():
    df = synth(10)
    df["da_forecast"] = df["demand"] * 1.05
    b = daily_bias(df, TZ)
    assert np.allclose(b.values, 0.05)
    # first UTC row 06:00 = local 00:00 = HE24 of the previous local day
    assert str(b.index[0]) == "2023-12-31"


def test_structural_break_found():
    df = synth(365 * 3, start="2021-01-01 06:00").set_index("period_utc")
    d = df["demand"].copy()
    d[d.index >= "2022-07-01"] *= 0.9
    m = d.resample("MS").mean()
    at, shift = best_single_break(m)
    assert at.strftime("%Y-%m") in {"2022-06", "2022-07", "2022-08"}
    assert shift == pytest.approx(-0.10, abs=0.02)


def test_no_break_on_steady_trend():
    df = synth(365 * 3, start="2021-01-01 06:00").set_index("period_utc")
    t = np.arange(len(df)) / (24 * 365)
    m = (df["demand"] * (1 + 0.02 * t)).resample("MS").mean()
    _, shift = best_single_break(m)
    assert abs(shift) < 0.04


def test_flags_and_clean_series():
    df = synth(30)
    df.loc[10, "demand"] = -5
    df.loc[20, "demand"] = np.nan
    df.loc[30, "da_forecast"] = df.loc[30, "demand"] * 3
    g = flag_all(df)
    assert g.loc[10, "f_nonpos"] and g.loc[20, "f_missing"] and g.loc[30, "f_ratio"]
    assert g.loc[10, "f_implausible"] and not g.loc[10, "fa_implausible"]  # adjusted is fine
    assert g["demand_clean"].notna().sum() >= len(g) - 2


def test_verdict_rules():
    df = synth(400)
    by, info = audit_ba(df, TZ)
    assert verdict(by, info)[0] == "KEEP"
    bad = synth(400)
    bad["da_forecast"] = bad["demand"] * 1.2
    by, info = audit_ba(bad, TZ)
    assert verdict(by, info)[0] == "DROP"
    shifted = synth(400)
    shifted["da_forecast"] = shifted["da_forecast"].shift(-1)
    by, info = audit_ba(shifted, TZ)
    assert verdict(by, info)[0].startswith("DROP or SHORTEN")
