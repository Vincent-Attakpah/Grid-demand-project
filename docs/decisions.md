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

## 2026-11-20 — What the comparison with Trick 2 does and does not show

The raw day-ahead forecast (B3) is a day old at issue time, so beating it is context only.
B4b updates it with the same as-of demand information the models get. Remaining caveats for
any write-up:

1. Our weather inputs are fresher than the weather the utility used for its day-ahead
   forecast. This is legitimate (public at issue time) but part of the gain is newer weather.
2. Utilities' internal intraday forecasts are not public; the claim is "better than the best
   public forecast updated with public data", not "better than what the utility knows".

Ablation (reports/models/ablation_weather.md), 2026, 1-6 h, MAE reduction:
weather on top of identical inputs adds 3.1% DUK, 4.2% FPL, 2.9% SOCO (all significant),
1.7% TVA (p = 0.09). Learning from the same demand data alone beats B4b by 9.2% in SOCO
and 4.8% in FPL. Calendar features add nothing at 1-6 h (slightly negative in SOCO 2025).

## 2026-11-21 — High-peak warnings: yearly line and stretched ensembles

1. The high-day line for year Y is now the 90th percentile of daily peaks per calendar month
   in Y-2 and Y-1 (2025: 2023-2024; 2026: 2024-2025). The fixed 2023-2024 line had gone
   stale: in 2026 SOCO crossed it on 98 of 319 days. With the yearly line: SOCO 81, DUK 68,
   TVA 70, FPL 58 of about 320 days (18-25%). 2026 is still well above the 10% the line
   implies, so demand growth outpaces even a two-year window.
2. Peak ensembles were too narrow (90% ranges caught 85-88%). At each month start a stretch
   factor is chosen from the previous 90 days of issues so their 90% ranges would have caught
   90%, and every ensemble in the month is widened around its median by it (same rule for
   M5 and R). Average factors are 1.04-1.33. 2026 coverage is now 89-92%.
3. Unchanged: M5 beats climatology by 67-95% (Brier) and is level with R; the M3 point
   forecast's better peak size does not turn into better yes/no warnings.

## 2026-11-21 — FPL revises its day-ahead forecast during the day

The logger shows FPL's DF for day D first posted about 06:30 ET on D and revised about
14:00-15:30 ET (36 of 124 logged hours changed, by up to about 300 MW; bulk vs logged up to 492 MW). SOCO, TVA, DUK and ERCO
never changed. The bulk files keep only the final value, so for FPL issues between 10:00 and
~15:00 ET the backtest used a DA that did not exist yet. This flatters every DA-based
forecast for FPL (DA, B4b, M3, M4); the M3-vs-B4b comparison is less affected because both
see the same DA. The live system uses the DA as posted at each issue time, so its FPL scores
are the honest ones. A clean FPL backtest needs as-of DF history, which only the logger has
(from 2026-11-18).

## 2026-11-21 — Live system

* Hourly, on GitHub Actions: `forecast.yml` runs after every logger run (schedule or laptop
  backup). It pulls 65 days of EIA demand/DA and each BA's weather (Previous Runs API: the
  same day-1/day-2 values as the backtest), builds the issue with the backtest's own
  functions, and writes forecasts, peak warnings, scores and a static dashboard to the
  orphan branch `live`.
* Monthly, on the laptop: `scripts/monthly_refresh.ps1` updates data, refits M3/M4 on all
  published labels and uploads the bundle to the release `live-models`.
* Checked on 2026-11-18/19 issues: live weather features equal the backtest's exactly; demand
  features differ only where the bulk file ended before the issue (backtest carried old
  values forward) and FPL's DA (revision above).
* The API's demand (D) equals bulk adjusted demand (identical for SOCO/TVA/FPL, DUK within
  3 MW). The bulk "imputed" flag is not in the API, so live hours are not filtered for it.
