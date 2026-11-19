import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nowcaster.eval.scorecard import aggregate, daily_mape


def _day(date, n_hours, demand=100.0, fc=110.0):
    return pd.DataFrame({"ba": "X", "date": date, "demand": [demand] * n_hours,
                         "da_forecast": [fc] * n_hours})


def test_min_20_valid_hours():
    d = pd.concat([_day(dt.date(2025, 1, 1), 20), _day(dt.date(2025, 1, 2), 19)])
    out = daily_mape(d).set_index("date")
    assert out.loc[dt.date(2025, 1, 1), "scored"]
    assert not out.loc[dt.date(2025, 1, 2), "scored"]
    assert np.isnan(out.loc[dt.date(2025, 1, 2), "mape"])


def test_ratio_filter_and_nonpositive():
    d = _day(dt.date(2025, 1, 1), 24)
    d.iloc[0, d.columns.get_loc("da_forecast")] = 9_971_121  # unit error -> excluded
    d.iloc[1, d.columns.get_loc("demand")] = 0               # non-positive -> excluded
    d.iloc[2, d.columns.get_loc("da_forecast")] = 200        # ratio exactly 2 -> kept
    out = daily_mape(d).iloc[0]
    assert out.hours_valid == 22
    assert out.mape == pytest.approx((21 * 0.1 + 1.0) / 22)


def test_hour_weighted_aggregate():
    d = pd.concat([_day(dt.date(2025, 1, 1), 24, fc=110), _day(dt.date(2025, 1, 2), 20, fc=120)])
    daily = daily_mape(d)
    agg = aggregate(daily, ["ba"]).iloc[0]
    assert agg.hours == 44
    assert agg.mape == pytest.approx((24 * 0.1 + 20 * 0.2) / 44)
