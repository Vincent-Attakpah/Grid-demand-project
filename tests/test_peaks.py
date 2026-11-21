"""Peak ensembles use only earlier days' errors."""

import numpy as np
import pandas as pd

from nowcaster.models.peaks import actual_peak, issue_arrays, monthly_thresholds, peak_ensembles, summarize

TZ = "America/New_York"


def synth_pred(days=90, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2025-03-01", periods=days, freq="D"):
        for hr in (12, 15):                                  # two issue hours a day
            tau = pd.Timestamp(d.date()).tz_localize(TZ) + pd.Timedelta(hours=hr, minutes=17)
            K = 24 - hr
            for k in range(1, K + 1):
                truth = 20000 + 3000 * np.sin(np.pi * (hr + k) / 24) + rng.normal(0, 300)
                rows.append(dict(issue_utc=tau.tz_convert("UTC"), op_day=d.date(), horizon=k,
                                 da_posted=True, actual=truth, M3=truth + rng.normal(0, 200),
                                 B4b=truth + rng.normal(0, 400), da=truth + rng.normal(0, 600)))
    return pd.DataFrame(rows)


def ens_for(pred, model="M3"):
    meta, arrs = issue_arrays(pred, ["M3", "B4b", "da"], TZ)
    return meta, arrs, peak_ensembles(meta, arrs, model)


def test_no_future_errors():
    pred = synth_pred()
    meta, arrs, base = ens_for(pred)
    i = int(np.flatnonzero([e is not None for e in base])[10])
    day = meta.at[i, "op_day"].date()
    pert = pred.copy()
    pert.loc[pert["op_day"] >= day, "actual"] *= 1.4         # that day and every later day
    _, _, out = ens_for(pert)
    np.testing.assert_allclose(base[i], out[i])
    # control: changing an earlier day's actuals changes the ensemble
    ctrl = pred.copy()
    ctrl.loc[ctrl["op_day"] == (pd.Timestamp(day) - pd.Timedelta(days=1)).date(), "actual"] *= 1.4
    _, _, moved = ens_for(ctrl)
    assert not np.allclose(base[i], moved[i])


def test_summary_and_calibration():
    pred = synth_pred()
    meta, arrs, e = ens_for(pred)
    pk = actual_peak(arrs)
    assert np.isfinite(pk).all()
    thr = monthly_thresholds(pd.Series(pk, index=meta["op_day"]).groupby(level=0).max())
    s = summarize(meta, arrs, {"M5": e}, thr).dropna(subset=["p_M5"])
    inside = (s["peak_actual"] >= s["lo_M5"]) & (s["peak_actual"] <= s["hi_M5"])
    assert 0.75 <= inside.mean() <= 1.0
    assert s["p_M5"].between(0, 1).all()
