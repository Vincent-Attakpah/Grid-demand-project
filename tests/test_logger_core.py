"""Logger tests with mocked HTTP (no network)."""

import datetime as dt
import json

import pytest

from nowcaster.logger import eia, openmeteo, report, run as runmod

UTC = dt.timezone.utc


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code, self.text = payload, status, json.dumps(payload)[:200]

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    """Serves EIA from a mutable `series` dict {(ba,type,period): value}; Open-Meteo from
    a callable; meta returns a fixed init time."""

    def __init__(self, series, om_value=5.0):
        self.series, self.om_value, self.calls = series, om_value, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if url.startswith(eia.BASE_URL):
            p = dict(params)
            start = p["start"]
            data = [{"period": per, "respondent": ba, "type": ty, "value": v, "value-units": "MWh"}
                    for (ba, ty, per), v in sorted(self.series.items()) if per >= start]
            off, n = int(p["offset"]), int(p["length"])
            return FakeResp({"response": {"total": str(len(data)), "data": data[off:off + n]}})
        if "meta.json" in url:
            return FakeResp({"last_run_initialisation_time": 1736920800})
        if url == openmeteo.FORECAST_URL:
            lats = params["latitude"].split(",")
            vars_ = params["hourly"].split(",")
            times = ["2025-01-15T16:00", "2025-01-15T17:00"]
            loc = {"hourly": {"time": times, **{v: [self.om_value, self.om_value] for v in vars_}}}
            return FakeResp([loc] * len(lats) if len(lats) > 1 else loc)
        return FakeResp({}, 404)


def per(t):
    return t.strftime("%Y-%m-%dT%H")


NOW = dt.datetime(2025, 1, 15, 15, 20, tzinfo=UTC)


def test_parse_period_utc_hour_ending():
    assert eia.parse_period("2025-01-15T16") == dt.datetime(2025, 1, 15, 16, tzinfo=UTC)
    assert eia.parse_period("2025-01-15T10-06") == dt.datetime(2025, 1, 15, 16, tzinfo=UTC)


def test_eia_pagination(monkeypatch):
    monkeypatch.setattr(eia, "PAGE", 3)
    series = {("SOCO", "D", per(NOW - dt.timedelta(hours=h))): 100.0 + h for h in range(1, 8)}
    rows = eia.fetch(NOW, api_key="k", bas=("SOCO",), types=("D",), session=FakeSession(series))
    assert len(rows) == 7
    assert rows[0]["period_utc"].endswith("Z") and rows[0]["is_future"] is False


def test_eia_forward_values_flagged():
    future = per(NOW + dt.timedelta(hours=20))
    rows = eia.fetch(NOW, api_key="k", bas=("SOCO",), types=("DF",),
                     session=FakeSession({("SOCO", "DF", future): 9.0}))
    assert rows[0]["is_future"] is True


def _run(tmp_path, series, now, om_value=5.0):
    clock = iter([now] * 20)
    return runmod.run(tmp_path, now_fn=lambda: next(clock), session=FakeSession(series, om_value),
                      eia_kwargs={"api_key": "k", "bas": ("SOCO",), "types": ("D",)}, om_bas=["TVA"])


def _lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_dedup_skips_identical_and_keeps_revisions(tmp_path):
    series = {("SOCO", "D", per(NOW - dt.timedelta(hours=h))): 100.0 for h in range(1, 4)}
    r1 = _run(tmp_path, series, NOW)
    assert r1["sources"]["eia"]["rows_written"] == 3
    r2 = _run(tmp_path, series, NOW + dt.timedelta(hours=1))
    assert r2["sources"]["eia"]["rows_written"] == 0
    assert r2["sources"]["openmeteo"]["rows_written"] == 0
    series[("SOCO", "D", per(NOW - dt.timedelta(hours=1)))] = 101.5  # revision
    r3 = _run(tmp_path, series, NOW + dt.timedelta(hours=2), om_value=6.0)  # new weather run
    assert r3["sources"]["eia"]["rows_written"] == 1
    assert r3["sources"]["openmeteo"]["rows_written"] > 0
    rows = _lines(tmp_path / "raw" / "eia" / "2025-01-15.jsonl")
    assert [r["value"] for r in rows].count(101.5) == 1
    assert all({"retrieved_at_utc", "sched_hour_utc", "run_id", "src"} <= set(r) for r in rows)


def test_overlap_fills_missed_run(tmp_path):
    series = {("SOCO", "D", per(NOW - dt.timedelta(hours=1))): 1.0}
    _run(tmp_path, series, NOW)
    # Runs at +1h and +2h are dropped; hours arrive meanwhile; the +3h run picks all up.
    for h in range(0, 3):
        series[("SOCO", "D", per(NOW + dt.timedelta(hours=h)))] = 2.0 + h
    r = _run(tmp_path, series, NOW + dt.timedelta(hours=3))
    assert r["sources"]["eia"]["rows_written"] == 3


def test_one_source_failing_does_not_stop_other(tmp_path):
    clock = iter([NOW] * 20)
    rec = runmod.run(tmp_path, now_fn=lambda: next(clock), session=FakeSession({}),
                     eia_kwargs={"api_key": None}, om_bas=["TVA"])
    if rec["sources"]["eia"]["status"] == "ok":  # EIA_API_KEY set in env: force failure
        pytest.skip("EIA_API_KEY present in environment")
    assert rec["sources"]["openmeteo"]["status"] == "ok" and rec["ok"]


def test_index_prunes_old_keys(tmp_path):
    series = {("SOCO", "D", per(NOW - dt.timedelta(hours=1))): 1.0}
    _run(tmp_path, series, NOW)
    _run(tmp_path, {}, NOW + dt.timedelta(days=10))
    idx = json.loads((tmp_path / "state" / "index.json").read_text())
    assert idx["eia"] == {}


def test_scheduled_hour_and_delay():
    assert runmod.scheduled_hour(dt.datetime(2025, 1, 15, 14, 40, tzinfo=UTC), 17).hour == 14
    assert runmod.scheduled_hour(dt.datetime(2025, 1, 15, 15, 5, tzinfo=UTC), 17).hour == 14


# ---------- report math on a synthetic log ----------

def _write_log(root, rows, run_times):
    (root / "raw" / "eia").mkdir(parents=True)
    with open(root / "raw" / "eia" / "2025-01-15.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(root / "runs.jsonl", "w") as f:
        for t in run_times:
            f.write(json.dumps({"sources": {"eia": {"status": "ok", "retrieved_at_utc": t}}}) + "\n")


def test_report_math(tmp_path):
    runs = ["2025-01-15T12:20:00Z", "2025-01-15T13:20:00Z", "2025-01-15T14:20:00Z", "2025-01-15T15:20:00Z"]
    rows = [
        # pre-existing hour: excluded from lag stats
        {"ba": "TVA", "type": "D", "period_utc": "2025-01-15T11:00:00Z", "value": 100.0, "retrieved_at_utc": runs[0]},
        # HE13 first seen 14:20 -> upper lag 1.333h, lower (prev run 13:20) 0.333h
        {"ba": "TVA", "type": "D", "period_utc": "2025-01-15T13:00:00Z", "value": 200.0, "retrieved_at_utc": runs[2]},
        # HE14 first seen 15:20 -> 1.333h; then revised +2% at a later run
        {"ba": "TVA", "type": "D", "period_utc": "2025-01-15T14:00:00Z", "value": 100.0, "retrieved_at_utc": runs[3]},
        {"ba": "TVA", "type": "D", "period_utc": "2025-01-15T14:00:00Z", "value": 102.0, "retrieved_at_utc": "2025-01-15T16:20:00Z"},
        # DF for Jan 16 (Central) seen on Jan 15 local -> tomorrow exposed
        {"ba": "TVA", "type": "DF", "period_utc": "2025-01-16T12:00:00Z", "value": 5.0, "retrieved_at_utc": runs[1]},
    ]
    _write_log(tmp_path, rows, runs)
    e = report.load_eia(tmp_path)
    rt = report.eia_run_times(tmp_path)
    lag = report.demand_lag(e, rt).loc["TVA"]
    assert lag["n_hours"] == 2
    assert lag["lag_upper_median_h"] == pytest.approx(1.33, abs=0.01)
    assert lag["lag_lower_median_h"] == pytest.approx(0.33, abs=0.01)
    rev = report.revisions(e).loc[("TVA", "D")]
    assert rev["share_revised"] == pytest.approx(1 / 3, abs=1e-3)
    assert rev["share_gt_1pct"] == pytest.approx(1 / 3, abs=1e-3)
    tom = report.tomorrow_exposure(e, {"TVA": "America/Chicago"})
    assert int(tom.loc[0, "future_days_seen"]) == 1
    fa = report.da_first_appearance(e, rt, {"TVA": "America/Chicago"})
    assert fa.loc[0, "first_seen_local"] == "2025-01-15 07:20" and bool(fa.loc[0, "seen_before_day_start"])
    assert "Demand lag" in report.build(tmp_path)


@pytest.mark.network
def test_live_openmeteo_smoke(tmp_path):
    import os
    if os.environ.get("NOWCASTER_NETWORK_TESTS") != "1":
        pytest.skip("set NOWCASTER_NETWORK_TESTS=1")
    rows, errs = openmeteo.fetch(dt.datetime.now(UTC), bas=["TVA"])
    assert rows and not errs
