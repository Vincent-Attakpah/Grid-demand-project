# References check (cloned 2026-11-18, shallow, read-only; no code run)

Clones: scratchpad/refs/sc (cardinalgrid/ba-forecast-scorecard, commit 1ddc4b4 "Daily update: 2026-11-18") and scratchpad/refs/jn (Jeneidi/balancing-authority, commit 1363a81, 2026-10-15). Both reachable. Shallow clone means no history, so I cannot say what earlier daily commits reported.

## 1. Scorecard results (current as of 2026-11-18, regenerated daily)

results/ contains: summary.json, ranking_overall.csv, ranking_recent.csv (last 30 days), ba_monthly.csv, ba_yearly.csv, events_by_ba.csv, excluded_series_mismatch.csv. (ba_daily.csv is written locally by the code but not committed.)
Date range: 2015-07-01 to 2026-11-18 (results/summary.json first_date/last_date; README.md line 15). generated_at_utc 2026-11-18T22:39:33. Version 0.1.1.
DOIs (README.md lines 79-80): concept 10.5281/zenodo.22697386; v0.1.1 10.5281/zenodo.22697387. The DOI you cited is the v0.1.1 version DOI: confirmed.

Claim checks (README.md lines 19-25; results/summary.json):
- Peak-hour under-forecast share: README.md line 21 says "60% of eligible BA-days"; summary.json share_of_ba_days_under_forecast_at_peak_eligible = 0.59505. Your "59%" is the truncated value; say "about 59.5% (README rounds to 60%)".
- 1 BA-day in 100 under-forecast by more than: README.md line 21 says 27.9%; summary.json p99_under_forecast_at_peak_eligible = -0.27872. Your "27.7%" does NOT match the current files (27.9%). It may match an earlier daily run; unverifiable without history.
- TVA DA MAPE: ranking_overall.csv rank 3, mape 0.022232; README.md line 22 "TVA (2.2%)". Confirmed.
- Data-fault warning: README.md line 24 "Single-day extremes may still be data problems; the percentile above is the robust figure"; Known limitations lines 69-70 warn of unit errors, zeros, stale values. Confirmed. Worst day: PACE 2016-04-02, -50% on a 10,048 MW peak.
- Other headline numbers: demand-weighted MAPE 3.56%, median BA 4.22%, 43 eligible of 72 BAs, 213,505 BA-days, 5,117,432 hours, 115,589 hours flagged implausible.

## 2. Scorecard method and published MAPE (src/ba_scorecard/metrics.py, data.py, scorecard.py)

Published MAPE, results/ranking_overall.csv, full period 2015-07-01..2026-11-18:

| BA | MAPE | days | hours | rank | mean demand MW | bias_pct |
|---|---|---|---|---|---|---|
| TVA | 2.2232% | 4122 | 98909 | 3 | 17478 | +0.67% |
| DUK | 2.8002% | 4152 | 99643 | 10 | 12108 | +0.51% |
| SOCO | 2.8068% | 3985 | 95622 | 11 | 26946 | -1.96% |
| FPL | 4.1717% | 4106 | 98458 | 19 | 15475 | -2.14% |

Last 30 days (ranking_recent.csv): SOCO 1.804%, DUK 1.931%, TVA 2.042%, FPL 2.706%.
Per-year values are in results/ba_yearly.csv (2025: TVA 2.313%, SOCO 1.216%, DUK 2.573%, FPL 3.183%; 2026 partial: TVA 2.390%, SOCO 1.283%, DUK 3.648%, FPL 3.096%). Full-period figures move slightly each day as the window grows.

Definition:
- Source: EIA-930 six-month BALANCE files, https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_<YEAR>_<Jan_Jun|Jul_Dec>.csv, from 2015 Jul_Dec (data.py FIRST_PERIOD "2015H2") to the current half-year.
- Demand column: "Demand (MW)" (raw reported). "Demand (MW) (Adjusted)" is read into the tidy table but NOT used by metrics.py. "Demand (MW) (Imputed)" is used only for an imputed flag.
- Forecast: "Demand Forecast (MW)". Alignment: none beyond the same row (same BA, same hour). Day = EIA "Data Date" (local date, data.py line 126), grouped by (ba, date).
- error = forecast - demand; hourly abs pct error = |forecast - demand| / demand.
- Valid hour: demand > 0 and forecast > 0. Implausible hour (excluded, counted): forecast/demand > 2 or < 0.5 (IMPLAUSIBLE_RATIO = 2.0).
- Daily MAPE = mean of hourly abs pct error over valid hours; a BA-day is scored only with at least 20 valid hours (MIN_HOURS_PER_DAY = 20).
- BA MAPE (aggregate) = sum(daily MAPE x valid hours) / sum(valid hours) over scored days.
- Ranking eligibility: at least 180 scored days, mean demand at least 500 MW, |median daily bias| at most 25%. All four BAs qualify.
- Imputed demand is not removed (0.0002% of hours). No extra DST or dedup handling beyond EIA's Data Date/Hour Number; thousands separators stripped; non-numeric coerced to NaN.

Reproduction recipe (to within 0.1 pt):
1. Load all EIA930_BALANCE_* files 2015_Jul_Dec through the 2026 files; keep BA in SOCO, TVA, DUK, FPL.
2. demand = raw "Demand (MW)"; forecast = "Demand Forecast (MW)"; same-row alignment; strip commas, coerce to numeric.
3. Keep hours with demand > 0, forecast > 0 and 0.5 <= forecast/demand <= 2.
4. Per (BA, Data Date): MAPE_day = mean(|f-d|/d) if at least 20 valid hours, else drop.
5. BA MAPE = valid-hour-weighted mean of MAPE_day.
Expected for the full period to 2026-11-18: TVA 2.22, DUK 2.80, SOCO 2.81, FPL 4.17. For a different window or Adjusted demand, differences are expected (not tested here). I did not run the code or download the EIA files, so the recipe is verified only by reading the source.

## 3. Jeneidi/balancing-authority

- BAs: FPL, DUK, TVA only (src/forecast.py lines 13-17).
- Method: one LightGBM regressor per BA (500 trees, lr 0.04, early stopping 30 on validation); features hour, day of week, month, weekend, day sin/cos, demand lag 24 h and 168 h, Open-Meteo previous-day-1 temperature forecast (one lat/lon per BA). Baselines: the BA's own DA forecast and same-hour-last-week. 90% interval from validation residual 5th/95th percentiles; 1,000-draw bootstrap of daily MAPE.
- Horizon: day-ahead hourly. Split: train 2024-01..2025-06, validation 2025-07..2025-12, test 2026-01-01..2026-06-30 (README.md lines 25-28, 38).
- Results, test 2026 H1 (README.md lines 15-19): MAPE = mean(|a-p|/max(|a|,1)) over hourly rows (forecast.py lines 44-48).

| BA | Model | Operator | Last week | Win rate | 90% coverage |
|---|---|---|---|---|---|
| FPL | 4.290 | 3.225 | 11.216 | 43.09% | 90.33% |
| DUK | 4.622 | 4.947 | 13.502 | 28.73% | 85.43% |
| TVA | 3.840 | 2.395 | 13.033 | 20.99% | 87.48% |

  Their operator MAPE covers only 2026 H1, so it is not comparable with the scorecard's all-years figures.
- TVA 9,971,121 MW: YES. README.md line 21 ("one corrupted raw value of `9971121 MW`", replaced by `16915 MW` from the Adjusted field); also tests/test_core.py line 47 and site/index.html. The README does not give the date.
- Cleaning rules worth reusing (forecast.py load_eia, lines 58-74): demand = Adjusted demand with fallback to raw where Adjusted is blank (adjusted.fillna(raw)); coerce numerics; drop rows missing timestamp, demand or forecast; drop duplicates on (BA, UTC timestamp); sort by BA and time. There is no range or outlier filter beyond using Adjusted. Lags use row shift(24)/shift(168), so missing hours would silently misalign lags.
- Cross-check: the scorecard's 0.5-2 ratio rule would drop that 9.97M MW hour (TVA hours_implausible = 13 in ranking_overall.csv).
