import datetime as dt

import pandas as pd
import pytest

from nowcaster.config import BA, City
from nowcaster.weather import backfill as bf
from nowcaster.weather.asof import safe_offset, source_for
from nowcaster.weather.client import Throttle

VARS = ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "wind_speed_10m",
        "cloud_cover", "shortwave_radiation"]
BAS = {"XX": BA(code="XX", name="Test", tz="America/Chicago", cities=(
    City("A", "AL", 33.0, -86.0, 100), City("B", "AL", 34.0, -87.0, 300)))}


def fake_fetch_factory(calls):
    def fetch(url, params, throttle, weight):
        calls.append((url, params))
        times = pd.date_range(params["start_date"], params["end_date"] + " 23:00", freq="h")
        names = params["hourly"].split(",")
        loc = {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                          **{n: [1.0] * len(times) for n in names}}}
        return [loc for _ in params["latitude"].split(",")]
    return fetch


def test_month_chunks_marks_partial_month():
    chunks = list(bf.month_chunks(dt.date(2024, 1, 1), dt.date(2024, 3, 10)))
    assert chunks[0] == (dt.date(2024, 1, 1), dt.date(2024, 1, 31), True)
    assert chunks[-1] == (dt.date(2024, 3, 1), dt.date(2024, 3, 10), False)


def test_jobs_cover_sources_and_skip_resume(tmp_path):
    jobs = [j for j in bf.build_jobs(BAS, dt.date(2024, 3, 11)) if j.start >= dt.date(2024, 1, 1)]
    assert {j.source for j in jobs} == {"previous_runs_full", "historical_forecast", "era5"}
    calls = []
    th = Throttle(safety=1.0)
    s1 = bf.run_jobs(jobs, BAS, tmp_path, th, fetch=fake_fetch_factory(calls), max_jobs=2)
    assert s1["fetched"] == 2 and len(calls) == 2
    s2 = bf.run_jobs(jobs, BAS, tmp_path, th, fetch=fake_fetch_factory(calls))
    assert s2["skipped"] == 2 and s2["fetched"] == len(jobs) - 2
    s3 = bf.run_jobs(jobs, BAS, tmp_path, th, fetch=fake_fetch_factory(calls))
    assert s3["fetched"] == 0 and s3["skipped"] == len(jobs)


def test_partial_month_refetched_later(tmp_path):
    calls = []
    th = Throttle(safety=1.0)
    early = [j for j in bf.build_jobs(BAS, dt.date(2024, 3, 11)) if j.source == "era5" and j.start.month == 3 and j.start.year == 2024]
    late = [j for j in bf.build_jobs(BAS, dt.date(2024, 3, 20)) if j.source == "era5" and j.start.month == 3 and j.start.year == 2024]
    bf.run_jobs(early, BAS, tmp_path, th, fetch=fake_fetch_factory(calls))
    bf.run_jobs(late, BAS, tmp_path, th, fetch=fake_fetch_factory(calls))
    assert len(calls) == 2  # different end date -> new cache file


def test_previous_runs_params_and_parse(tmp_path):
    job = bf.Job("previous_runs_full", "XX", dt.date(2024, 2, 1), dt.date(2024, 2, 2), True)
    p = bf.job_params(job, BAS["XX"], VARS, "gfs_seamless")
    assert "temperature_2m_previous_day1" in p["hourly"] and "shortwave_radiation_previous_day2" in p["hourly"]
    assert p["models"] == "gfs_seamless" and p["timezone"] == "GMT"
    resp = fake_fetch_factory([])(None, p, None, 1)
    df = bf.parse_payload({"meta": {"source": job.source, "ba": "XX", "cities": ["A", "B"]},
                           "response": resp}, VARS)
    assert set(df["source"]) == {"prev_day1", "prev_day2"}
    assert len(df) == 2 * 2 * 48


def test_timestamp_convention_is_utc_hour_ending():
    payload = {"meta": {"source": "era5", "ba": "XX", "cities": ["A"]},
               "response": [{"hourly": {"time": ["2024-07-01T00:00", "2024-07-01T01:00"],
                                        "temperature_2m": [20.0, 19.5],
                                        "shortwave_radiation": [0.0, 0.0]}}]}
    df = bf.parse_payload(payload, VARS)
    assert str(df["period_utc"].dt.tz) == "UTC"
    # Open-Meteo stamp T (GMT) is used as hour-ending T: radiation covers (T-1h, T].
    assert df["period_utc"].iloc[0] == pd.Timestamp("2024-07-01 00:00", tz="UTC")
    assert df["dew_point_2m"].isna().all()  # missing variables become NaN columns


def test_era5_uses_era5_model():
    job = bf.Job("era5", "XX", dt.date(2024, 2, 1), dt.date(2024, 2, 29), True)
    assert bf.job_params(job, BAS["XX"], VARS, "gfs_seamless")["models"] == "era5"


def test_location_count_mismatch_raises():
    with pytest.raises(ValueError):
        bf.parse_payload({"meta": {"source": "era5", "ba": "XX", "start": "x", "cities": ["A", "B"]},
                          "response": [{"hourly": {"time": []}}]}, VARS)


def test_safe_offset_avoids_leak():
    assert safe_offset(1, 6) == 1
    assert safe_offset(18, 6) == 1
    assert safe_offset(19, 6) == 2   # day 1 would come from a run not yet available
    assert safe_offset(23, 6) == 2
    assert safe_offset(42, 6) == 2
    assert safe_offset(43, 6) == 3
    assert source_for(0, 6) == "hist_forecast"
    assert source_for(-3, 6) == "hist_forecast"
    assert source_for(5, 6) == "prev_day1"
