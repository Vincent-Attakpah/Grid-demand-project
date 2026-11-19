# Design decisions (changes to plan v2)

## 2026-11-19 — B4b is the bar to beat

**Decision.** The minimum success criterion becomes: in the 2026 backtest, M3 beats **B4b**
(DA + latest observed error + same-hour-yesterday error) at 1–6 hours, with significance in
at least 2 of 3 core BAs; live skill against B4b at 1–6 hours is positive. B4 and B3 are
still reported alongside.

**Why.** On 2025 rolling origins (reports/baselines/baselines_2025.md), B4b beats B4 by
about 30% in SOCO and 18% in FPL at 1–6 h: the DA forecasts carry hour-of-day bias that
repeats from day to day. A model that only beats B4 could be relearning that bias, which
B4b captures for free.

## 2026-11-19 — Weather offsets by horizon

Previous Runs day-1 values are used only up to 18 h ahead; day-2 from 19 h (weather/asof.py).
Plan v2's "day 1 for all future hours" would use forecasts not yet issued for long horizons.
Full-feature training starts 2024-01-21 (gfs_seamless Previous Runs archive start).

## 2026-11-19 — Feature cleaning never uses the hidden DA forecast

Labels use the scorecard filter (DA/demand in [0.5, 2], no imputed hours). Feature inputs use
a causal trailing-median check instead, so today's DA cannot influence features before it
is posted (tests/test_leakage.py::test_da_posting).
