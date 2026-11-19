"""Point-in-time rules for archived weather (plan v2 section 7.1).

Open-Meteo documents `<var>_previous_dayN` as the value predicted N*24 hours before the
valid time, i.e. from a run initialised at least 24N hours before the target hour. A run is
only usable some hours after initialisation (`weather_run_availability_hours` in bas.yaml).

So for issue time tau and target hour h = tau + k, the day-N value is provably available at
tau only if 24N >= k + availability. With 6 h availability, day 1 is safe up to k = 18 h and
day 2 up to k = 42 h. Rest-of-day horizons reach ~23 h (issue just after midnight), so
"day 1 for every future hour" would leak for k > 18. Use `safe_offset` instead.
"""

from __future__ import annotations

import math

from ..config import lag_settings


def safe_offset(horizon_h: float, availability_h: float | None = None, max_day: int = 7) -> int:
    """Smallest previous-runs day offset N whose forecast existed at issue time."""
    if availability_h is None:
        availability_h = float(lag_settings()["weather_run_availability_hours"])
    if horizon_h < 0:
        raise ValueError("horizon must be >= 0 (past hours use historical_forecast)")
    n = max(1, math.ceil((horizon_h + availability_h) / 24))
    if n > max_day:
        raise ValueError(f"no archived offset covers horizon {horizon_h} h")
    return n


def source_for(horizon_h: float, availability_h: float | None = None) -> str:
    """Weather source label in weather_*_hourly for a target `horizon_h` hours ahead.

    horizon <= 0 (hours at or before issue time) -> 'hist_forecast', which the caller must
    also shift by the availability delay (test_weather_past_delay). Future hours -> prev_dayN.
    """
    if horizon_h <= 0:
        return "hist_forecast"
    return f"prev_day{safe_offset(horizon_h, availability_h)}"
