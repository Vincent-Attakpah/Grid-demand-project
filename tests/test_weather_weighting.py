import numpy as np
import pandas as pd
import pytest

from nowcaster.config import BA, City, load_bas
from nowcaster.weather.weighting import weight_ba

VARS = ["temperature_2m"]


def make_city_df(ba, cities, periods, values):
    rows = []
    for c, v in zip(cities, values):
        for p in periods:
            rows.append({"ba": ba, "city": c, "period_utc": p, "source": "prev_day1", "temperature_2m": v})
    return pd.DataFrame(rows)


def test_active_from_and_until():
    ba = BA(code="FX", name="x", tz="America/New_York", cities=(
        City("Old", "FL", 0, 0, 100, active_until="2022-07-01"),
        City("Main", "FL", 0, 0, 100),
        City("New", "FL", 0, 0, 200, active_from="2022-07-01")))
    periods = [pd.Timestamp("2022-06-30 23:00", tz="UTC"), pd.Timestamp("2022-07-01 00:00", tz="UTC")]
    df = make_city_df("FX", ["Old", "Main", "New"], periods, [10.0, 20.0, 30.0])
    out = weight_ba(df, {"FX": ba}, VARS).set_index("period_utc")
    assert out.loc[periods[0], "temperature_2m"] == pytest.approx(15.0)       # Old+Main
    assert out.loc[periods[1], "temperature_2m"] == pytest.approx((20 * 100 + 30 * 200) / 300)
    assert out.loc[periods[0], "n_cities"] == 2 and out.loc[periods[1], "n_cities"] == 2


def test_missing_value_renormalises():
    ba = BA(code="T", name="x", tz="UTC", cities=(City("A", "x", 0, 0, 100), City("B", "x", 0, 0, 300)))
    p = [pd.Timestamp("2024-01-01 01:00", tz="UTC")]
    df = make_city_df("T", ["A", "B"], p, [10.0, np.nan])
    out = weight_ba(df, {"T": ba}, VARS)
    assert out["temperature_2m"].iloc[0] == pytest.approx(10.0)


def test_configured_pensacola_moves_from_soco_to_fpl():
    bas = load_bas()
    soco = {c.name: c for c in bas["SOCO"].cities}
    fpl = {c.name: c for c in bas["FPL"].cities}
    assert soco["Pensacola"].active_until == fpl["Pensacola"].active_from
