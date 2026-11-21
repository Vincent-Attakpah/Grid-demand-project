"""Live forecaster pieces: issue time, input shapes, library, history files."""

import numpy as np
import pandas as pd

from nowcaster.config import load_bas
from nowcaster.live.forecast import issue_time, library
from nowcaster.live.inputs import eia_frame, weather_frame
from nowcaster.live.run import append_csv, read_dir


def test_issue_time_is_latest_17():
    assert issue_time("2026-11-20 14:50") == pd.Timestamp("2026-11-20 14:17", tz="UTC")
    assert issue_time("2026-11-20 14:17") == pd.Timestamp("2026-11-20 14:17", tz="UTC")
    assert issue_time("2026-11-20 14:16") == pd.Timestamp("2026-11-20 13:17", tz="UTC")   # late run keeps its hour


def test_eia_frame_pivots_types():
    rows = [{"ba": "SOCO", "type": "D", "period_utc": "2026-11-20T10:00:00Z", "value": 100.0},
            {"ba": "SOCO", "type": "DF", "period_utc": "2026-11-20T10:00:00Z", "value": 110.0},
            {"ba": "SOCO", "type": "DF", "period_utc": "2026-11-20T11:00:00Z", "value": 120.0}]
    f = eia_frame(rows).sort_values("period_utc")
    assert list(f["demand_adjusted"].fillna(-1)) == [100.0, -1]
    assert list(f["da_forecast"]) == [110.0, 120.0]
    assert not f["demand_imputed"].any()


def test_weather_frame_sources():
    ba = load_bas()["SOCO"]
    times = ["2026-11-20T00:00", "2026-11-20T01:00"]
    loc = {"hourly": {"time": times, "temperature_2m": [10.0, 11.0],
                      **{f"{v}_previous_day{d}": [1.0 * d, 2.0 * d]
                         for v in ["temperature_2m", "dew_point_2m", "wind_speed_10m", "cloud_cover",
                                   "shortwave_radiation", "relative_humidity_2m"] for d in (1, 2)}}}
    w = weather_frame("SOCO", ba, [loc] * len(ba.cities), [
        "temperature_2m", "dew_point_2m", "wind_speed_10m", "cloud_cover", "shortwave_radiation",
        "relative_humidity_2m"])
    assert set(w["source"]) == {"prev_day1", "prev_day2", "hist_forecast"}
    t1 = w[w["source"] == "prev_day1"].sort_values("period_utc")["temperature_2m"].to_numpy()
    np.testing.assert_allclose(t1, [1.0, 2.0])
    hist = w[w["source"] == "hist_forecast"].sort_values("period_utc")["temperature_2m"].to_numpy()
    np.testing.assert_allclose(hist, [10.0, 11.0])


def _rows(issue, model_val):
    t = pd.Timestamp(issue, tz="UTC")
    return pd.DataFrame({"issue_utc": t, "target_utc": [t.floor("h") + pd.Timedelta(hours=k) for k in (1, 2)],
                         "op_day": t.date(), "horizon": [1, 2], "da_posted": True, "da": 1.0,
                         "B4b": 2.0, "M3": model_val})


def test_library_prefers_live_and_live_actuals():
    seed = _rows("2026-11-10 15:17", 5.0).assign(actual=[50.0, 51.0])
    hist = _rows("2026-11-10 15:17", 7.0)
    actual = pd.Series([99.0], index=pd.DatetimeIndex([pd.Timestamp("2026-11-10 16:00", tz="UTC")]))
    lib = library(seed, hist, actual, pd.Timestamp("2026-11-01", tz="UTC"))
    assert list(lib["M3"]) == [7.0, 7.0]                 # live forecast replaces the seed copy
    assert list(lib["actual"]) == [99.0, 51.0]           # live actual where published, else seed


def test_history_round_trip(tmp_path):
    r = _rows("2026-11-20 15:17", 3.0)
    r["ba"] = "SOCO"
    append_csv(tmp_path / "history" / "2026-11-20.csv", r)
    append_csv(tmp_path / "history" / "2026-11-20.csv", r)
    back = read_dir(tmp_path / "history", pd.Timestamp("2026-11-01", tz="UTC"))
    assert len(back) == 4 and back["issue_utc"].dt.tz is not None
    assert read_dir(tmp_path / "history", pd.Timestamp("2026-11-21", tz="UTC")).empty


def test_actuals_archive_keeps_old_hours(tmp_path):
    from nowcaster.live.run import merge_actuals, read_actuals, write_actuals
    t = pd.date_range("2026-11-01 01:00", periods=3, freq="h", tz="UTC")
    now = pd.Timestamp("2026-11-20", tz="UTC")
    write_actuals(tmp_path / "a.csv", {"SOCO": pd.Series([1.0, 2.0, 3.0], index=t)}, now)
    old = read_actuals(tmp_path / "a.csv")
    new = pd.Series([20.0, np.nan], index=t[1:])                 # newer pull: one revised, one missing
    m = merge_actuals(old, "SOCO", new)
    assert list(m) == [1.0, 20.0, 3.0]
