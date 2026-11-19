PY ?= python

.PHONY: install test log audit backfill-weather

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

log:            ## one logger run (what the hourly GitHub Action executes)
	$(PY) -m nowcaster.logger.run

audit:          ## EIA-930 history download + data-quality audit
	$(PY) -m nowcaster.ingest.audit

backfill-weather:
	$(PY) -m nowcaster.weather.backfill
