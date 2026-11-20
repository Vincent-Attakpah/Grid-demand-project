"""M2/M3 obey the same no-peeking rules as the baselines."""

import numpy as np
import pandas as pd

from nowcaster.eval.rolling import rolling_origin
from nowcaster.features.calendar import add_calendar
from nowcaster.features.panel import build_panel, clean_hourly, issue_times
from nowcaster.models.gbm import MODELS

from test_leakage import LAGS, TZ, synth_eia


def _panel():
    p = build_panel(clean_hourly(synth_eia()), TZ, issue_times("2025-01-08", "2025-03-01"), LAGS, ba="X")
    return add_calendar(p, TZ)


def test_models_no_global_fit():
    panel = _panel()
    origin = pd.Timestamp("2025-02-01", tz="UTC")
    base = rolling_origin(panel, MODELS, origin, "2025-03-01")
    assert base["M2"].notna().all()
    assert base.loc[base["da_posted"], "M3"].notna().all()
    assert base.loc[~base["da_posted"], "M3"].isna().all()
    pert = panel.copy()
    late = (pd.DatetimeIndex(pert["target_utc"]) + pd.Timedelta(hours=3)) >= origin
    pert.loc[late, "actual"] *= 1.5
    out = rolling_origin(pert, MODELS, origin, "2025-03-01")
    np.testing.assert_allclose(base["M2"], out["M2"])
    np.testing.assert_allclose(base["M3"], out["M3"], equal_nan=True)
    # control: perturbing labels that WERE published before the origin changes the fit
    early = ~late
    ctrl = panel.copy()
    ctrl.loc[early, "actual"] *= 1.5
    moved = rolling_origin(ctrl, MODELS, origin, "2025-03-01")
    assert not np.allclose(base["M2"], moved["M2"])


def test_models_accurate_on_synthetic():
    out = rolling_origin(_panel(), MODELS, "2025-02-01", "2025-03-01")
    err = (out["M2"] - out["actual"]).abs().mean()
    assert err < 1500        # signal amplitude 3000 MW, noise 300
