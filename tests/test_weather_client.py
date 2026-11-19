import pytest

from nowcaster.weather.client import Throttle, call_weight, get_json, write_once


def test_call_weight_matches_documented_examples():
    assert call_weight(15, 14) == pytest.approx(1.5)      # "2 weeks, 15 variables = 1.5"
    assert call_weight(5, 14) == 1.0
    assert call_weight(5, 28) == pytest.approx(2.0)
    assert call_weight(12, 31, n_locations=8) == pytest.approx(8 * 1.2 * 31 / 14)


class FakeClock:
    def __init__(self):
        self.t = 1_000_000.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_throttle_minute_window():
    c = FakeClock()
    th = Throttle(safety=1.0, clock=c, sleep=c.sleep)
    for _ in range(6):
        th.acquire(100)  # 600 in the first minute
    assert c.slept == []
    th.acquire(100)      # must wait for the first event to leave the minute window
    assert sum(c.slept) == pytest.approx(60.5)
    assert th.used("hour") == 700


def test_throttle_hour_and_day_windows():
    c = FakeClock()
    th = Throttle(safety=0.5, clock=c, sleep=c.sleep)  # caps: 300/min, 2500/h, 5000/day
    for _ in range(25):
        th.acquire(100)
        c.t += 61  # stay clear of the minute window
    th.acquire(100)  # 2600 > 2500 within the hour -> wait
    assert c.slept and c.slept[-1] > 0
    assert th.used("hour") <= 2500


def test_throttle_rejects_oversized_request():
    with pytest.raises(ValueError):
        Throttle(safety=0.5).wait_time(400)


def test_throttle_ledger_persists(tmp_path):
    c = FakeClock()
    ledger = tmp_path / "ledger.jsonl"
    Throttle(clock=c, sleep=c.sleep, ledger=ledger).acquire(42)
    th2 = Throttle(clock=c, sleep=c.sleep, ledger=ledger)
    assert th2.used("day") == 42
    c.t += 86401
    th3 = Throttle(clock=c, sleep=c.sleep, ledger=ledger)
    assert th3.used("day") == 0


class Resp:
    def __init__(self, code, body):
        self.status_code, self._body, self.text = code, body, str(body)

    def json(self):
        return self._body


class Sess:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


def test_get_json_retries_429_and_normalises_list():
    c = FakeClock()
    th = Throttle(clock=c, sleep=c.sleep)
    s = Sess([Resp(429, {"reason": "limit"}), Resp(200, {"hourly": {}})])
    out = get_json("u", {}, th, 1.0, session=s, sleep=c.sleep)
    assert out == [{"hourly": {}}]
    assert s.calls == 2 and max(c.slept) >= 60


def test_get_json_does_not_retry_400():
    from nowcaster.weather.client import OpenMeteoError
    c = FakeClock()
    th = Throttle(clock=c, sleep=c.sleep)
    s = Sess([Resp(400, {"reason": "bad"})])
    with pytest.raises(OpenMeteoError):
        get_json("u", {}, th, 1.0, session=s, sleep=c.sleep)
    assert s.calls == 1


def test_write_once(tmp_path):
    p = tmp_path / "a" / "b.json"
    assert write_once(p, {"x": 1}) is True
    assert write_once(p, {"x": 2}) is False
    assert '"x":1' in p.read_text()
