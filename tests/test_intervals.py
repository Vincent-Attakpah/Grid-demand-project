"""Forecast ranges: no peeking, ordered quantiles, calibration works."""

import numpy as np
import pandas as pd

from nowcaster.eval.rolling import rolling_origin
from nowcaster.models.intervals import INTERVAL_MODELS, LEVELS, conformal_shift

from test_models import _panel

Q = list(LEVELS)


def test_conformal_shift():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    # intervals far too narrow: the shift must widen them to ~90% coverage
    s = conformal_shift(y, np.full_like(y, -0.1), np.full_like(y, 0.1), miss=0.10)
    cover = np.mean((y >= -0.1 - s) & (y <= 0.1 + s))
    assert 0.89 <= cover <= 0.91
    # intervals far too wide: negative shift narrows them
    assert conformal_shift(y, np.full_like(y, -5), np.full_like(y, 5), miss=0.10) < 0


def test_intervals_ordered_no_peeking_and_cover():
    panel = _panel()
    panel = panel[panel["da_posted"]]
    origin = pd.Timestamp("2025-02-01", tz="UTC")
    base = rolling_origin(panel, INTERVAL_MODELS, origin, "2025-03-01")
    for m in ("M4", "Q0"):
        q = base[[f"{m}_{k}" for k in Q]].to_numpy()
        assert np.isfinite(q).all()
        assert (np.diff(q, axis=1) >= -1e-9).all()
        cover = np.mean((base["actual"] >= q[:, 0]) & (base["actual"] <= q[:, -1]))
        assert 0.80 <= cover <= 0.98, (m, cover)
    pert = panel.copy()
    late = (pd.DatetimeIndex(pert["target_utc"]) + pd.Timedelta(hours=3)) >= origin
    pert.loc[late, "actual"] *= 1.5
    out = rolling_origin(pert, INTERVAL_MODELS, origin, "2025-03-01")
    cols = [f"{m}_{k}" for m in ("M4", "Q0") for k in Q]
    np.testing.assert_allclose(base[cols].to_numpy(), out[cols].to_numpy())
