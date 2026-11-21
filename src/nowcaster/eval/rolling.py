"""Rolling-origin evaluation (plan v2 section 9.1): refit monthly on all data to date.

For origin month m, training rows are those whose label was published before m starts
(target_utc + label_lag_h < m_start); predictions are made for every row issued in m.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Model = tuple[Callable, Callable]


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def rolling_origin(panel: pd.DataFrame, models: dict[str, Model], start, end,
                   label_lag_h: int = 3, train_start=None) -> pd.DataFrame:
    """Return the rows issued in [start, end) with one prediction column per model."""
    issue = pd.DatetimeIndex(panel["issue_utc"])
    target = pd.DatetimeIndex(panel["target_utc"])
    lag = pd.Timedelta(hours=label_lag_h)
    t0 = _utc(train_start) if train_start is not None else None
    origins = pd.date_range(_utc(start), _utc(end), freq="MS")
    out = []
    for m0 in origins:
        m1 = m0 + pd.offsets.MonthBegin(1)
        test_mask = (issue >= m0) & (issue < min(m1, _utc(end)))
        if not test_mask.any():
            continue
        train_mask = (target + lag) < m0
        if t0 is not None:
            train_mask &= target >= t0
        train, test = panel[train_mask], panel[test_mask].copy()
        for name, (fit, predict) in models.items():
            pred = predict(fit(train), test)
            if isinstance(pred, pd.DataFrame):      # multi-output (e.g. quantiles): name_<col>
                for c in pred.columns:
                    test[f"{name}_{c}"] = pred[c].to_numpy()
            else:
                test[name] = pred
        test["origin"] = m0
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else panel.iloc[:0]
