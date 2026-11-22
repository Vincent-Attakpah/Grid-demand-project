# Live Grid Demand Nowcaster

Hourly re-forecasts of electricity demand for the rest of the operating day, for U.S.
balancing authorities (SOCO, TVA, DUK, FPL) whose only public forecast is the EIA-930
day-ahead number. Includes calibrated probabilities that today's remaining peak will
exceed the day-ahead forecast. Design: revised plan v2 (14 Nov 2026).

Status: under construction.

## Layout

```
config/bas.yaml        BA codes, time zones, weather cities, lag rules
src/nowcaster/
  config.py            loads bas.yaml, data dir, API keys
  timeutils.py         UTC hour-ending + operating-day helpers
  storage.py           append-only raw JSONL, partitioned Parquet, DuckDB
  logger/              hourly EIA + Open-Meteo logger (GitHub Actions, :17)
  ingest/              EIA-930 bulk history + data-quality audit
  weather/             Open-Meteo backfill + population weighting
tests/
```

## Setup

```
python -m venv ~/.venvs/nowcaster
~/.venvs/nowcaster/Scripts/python -m pip install -e ".[dev]"
```

Environment variables:
- `EIA_API_KEY` — free key from https://www.eia.gov/opendata/ (also a repository secret for Actions).
- `NOWCASTER_DATA` — local data root; defaults to `~/nowcaster-data` (keep it out of OneDrive).

## Live system

* **Every hour** (GitHub Actions, right after the logger): `.github/workflows/forecast.yml`
  runs `python -m nowcaster.live.run`, which forecasts the rest of today for each region,
  computes the high-peak warning, scores earlier forecasts against published demand, and
  writes everything to the `live` branch: `site/index.html` (dashboard),
  `site/scores.json`, `history/`, `peaks/`, `actuals.csv`, `status.json`.
* **Every month** (laptop, scheduled for the 2nd at 9 AM by `scripts/install_refresh_task.ps1`):
  `scripts/monthly_refresh.ps1` downloads new data, refits the
  models and uploads them to the GitHub release `live-models`.
* To view the dashboard while the repo is private: `powershell -File scripts/open_dashboard.ps1`.
* Before 10 AM Eastern there is no forecast yet: the day-ahead forecast counts as available
  from then (placeholder until the logger measures it).

## Data and attribution

Demand data: U.S. Energy Information Administration, Form EIA-930.
Weather data: [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), non-commercial use.
