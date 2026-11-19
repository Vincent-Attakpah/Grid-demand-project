"""Open-Meteo HTTP client: call weighting, rate-limit throttle, retries, write-once cache.

Free-tier terms (open-meteo.com/en/terms, checked 2026-11-18): fewer than 10,000 calls/day,
5,000/hour, 600/minute, non-commercial, CC BY 4.0. Limits are per IP.

Call weighting (open-meteo.com/en/pricing): a request counts as several calls when it asks
for more than 10 variables or more than 2 weeks per location; fractional counts are used
("2 weeks with 15 variables = 1.5 calls"). We implement
    weight = n_locations * n_models * max(1, n_vars/10) * max(1, n_days/14)
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests

ENDPOINTS = {
    "previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "forecast": "https://api.open-meteo.com/v1/forecast",
}

# Free-tier limits and the fraction of each we allow ourselves (headroom for the logger,
# other users on the same IP, and weighting rounding on the server side).
LIMITS = {"minute": (60, 600), "hour": (3600, 5000), "day": (86400, 10000)}
DEFAULT_SAFETY = 0.75


def call_weight(n_vars: int, n_days: float, n_locations: int = 1, n_models: int = 1) -> float:
    return n_locations * n_models * max(1.0, n_vars / 10) * max(1.0, n_days / 14)


@dataclass
class Throttle:
    """Sliding-window budget over minute/hour/day. Usage can be persisted to a ledger file
    so a restarted backfill still counts what it spent in the last 24 hours."""

    safety: float = DEFAULT_SAFETY
    ledger: Path | None = None
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    events: deque = field(default_factory=deque)  # (timestamp, weight)

    def __post_init__(self):
        if self.ledger and self.ledger.exists():
            now = self.clock()
            for line in self.ledger.read_text(encoding="utf-8").splitlines():
                try:
                    t, w = json.loads(line)
                except Exception:
                    continue
                if now - t < LIMITS["day"][0]:
                    self.events.append((t, w))

    def used(self, window: str, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        span = LIMITS[window][0]
        return sum(w for t, w in self.events if now - t < span)

    def wait_time(self, weight: float, now: float | None = None) -> float:
        """Seconds until `weight` fits in every window (0 if it fits now)."""
        now = self.clock() if now is None else now
        while self.events and now - self.events[0][0] >= LIMITS["day"][0]:
            self.events.popleft()
        wait = 0.0
        for window, (span, limit) in LIMITS.items():
            cap = limit * self.safety
            if weight > cap:
                raise ValueError(f"request weight {weight:.1f} exceeds the {window} budget {cap:.0f}")
            used = self.used(window, now)
            if used + weight <= cap:
                continue
            # Drop oldest events in this window until it fits; wait until the last dropped expires.
            excess = used + weight - cap
            for t, w in self.events:
                if now - t >= span:
                    continue
                excess -= w
                if excess <= 0:
                    wait = max(wait, t + span - now + 0.5)
                    break
        return wait

    def acquire(self, weight: float) -> float:
        """Block until the request fits, then record it. Returns seconds slept."""
        slept = 0.0
        while (w := self.wait_time(weight)) > 0:
            self.sleep(w)
            slept += w
        now = self.clock()
        self.events.append((now, weight))
        if self.ledger:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger, "a", encoding="utf-8") as f:
                f.write(json.dumps([now, weight]) + "\n")
        return slept


class OpenMeteoError(RuntimeError):
    pass


def get_json(url: str, params: dict, throttle: Throttle, weight: float, *, retries: int = 5,
             session: requests.Session | None = None, sleep: Callable[[float], None] = time.sleep,
             timeout: float = 120) -> list[dict]:
    """GET with throttling and retries. Always returns a list (one dict per location)."""
    s = session or requests
    delay = 10.0
    last = None
    for attempt in range(retries):
        throttle.acquire(weight)
        try:
            r = s.get(url, params=params, timeout=timeout)
        except requests.RequestException as e:  # network error: back off and retry
            last = str(e)
            sleep(delay)
            delay *= 2
            continue
        if r.status_code == 200:
            j = r.json()
            return j if isinstance(j, list) else [j]
        last = f"HTTP {r.status_code}: {r.text[:300]}"
        if r.status_code == 429:
            sleep(max(delay, 60.0))
        elif r.status_code >= 500:
            sleep(delay)
        else:
            raise OpenMeteoError(last)  # 4xx other than 429: bad request, don't retry
        delay *= 2
    raise OpenMeteoError(f"giving up after {retries} attempts: {last}")


def write_once(path: Path, payload: dict) -> bool:
    """Write a raw cache file only if it does not exist (raw data is immutable)."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)
    return True
