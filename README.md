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

## Data and attribution

Demand data: U.S. Energy Information Administration, Form EIA-930.
Weather data: [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), non-commercial use.
