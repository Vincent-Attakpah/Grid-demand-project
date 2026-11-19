"""Open-Meteo weather backfill (plan v2 sections 5.1, 7.1).

    python -m nowcaster.weather.backfill              # fetch everything missing, then clean
    python -m nowcaster.weather.backfill --dry-run    # list jobs and total weighted calls
    python -m nowcaster.weather.backfill --clean-only # rebuild clean tables from the raw cache

Jobs are (source, BA, calendar month); each requests all of the BA's cities at once and is
cached write-once to <data_dir>/raw/openmeteo_<source>/<BA>/<YYYY-MM>[_<end>].json. A month
that is complete is fetched once; the current, partial month is cached under its end date,
so a later run fetches a fuller copy and cleaning keeps the newest. Re-running resumes.

Sources (all for the fixed model in bas.yaml, except ERA5):
  previous_runs   *_previous_day1 and *_previous_day2, six variables, 2024-01 onward
  previous_runs   temperature_2m only, 2022-01 to 2023-12 (long-history ablation)
  historical_forecast   six variables, 2022-01 onward (past/current hours only!)
  archive (ERA5)  six variables, 2022-01 onward (diagnostics / upper bound only)

Time convention: requests use timezone=GMT. Open-Meteo stamps instantaneous variables
(temperature, dew point, humidity, wind, cloud cover) at the instant T, and
shortwave_radiation as the mean over the preceding hour. Both map to project hour-ending
period_utc = T: radiation exactly covers (T-1h, T]; instantaneous values are taken at the
end of the hour.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import BA, data_dir, load_bas, weather_settings
from ..storage import write_partitioned
from .client import ENDPOINTS, Throttle, call_weight, get_json, write_once

log = logging.getLogger("nowcaster.weather.backfill")

FULL_START = dt.date(2024, 1, 1)
LONG_START = dt.date(2022, 1, 1)
ERA5_LAG_DAYS = 6  # ERA5 (and ERA5T) is published with a few days' delay


@dataclass(frozen=True)
class Job:
    source: str            # previous_runs_full | previous_runs_temp | historical_forecast | era5
    ba: str
    start: dt.date
    end: dt.date           # inclusive
    complete: bool         # True if the month is fully in the past (cache never refreshed)

    @property
    def endpoint(self) -> str:
        return {"previous_runs_full": "previous_runs", "previous_runs_temp": "previous_runs",
                "historical_forecast": "historical_forecast", "era5": "archive"}[self.source]

    def variables(self, base_vars: list[str]) -> list[str]:
        if self.source == "previous_runs_full":
            return [f"{v}_previous_day{d}" for v in base_vars for d in (1, 2)]
        if self.source == "previous_runs_temp":
            return ["temperature_2m_previous_day1", "temperature_2m_previous_day2"]
        return list(base_vars)

    def cache_path(self, root: Path) -> Path:
        name = self.start.strftime("%Y-%m") + ("" if self.complete else f"_{self.end:%Y%m%d}")
        return root / "raw" / f"openmeteo_{self.source}" / self.ba / f"{name}.json"


def month_chunks(start: dt.date, end: dt.date):
    """Yield (month_start, month_end_clipped, complete) for calendar months in [start, end]."""
    m = dt.date(start.year, start.month, 1)
    while m <= end:
        last = dt.date(m.year, m.month, calendar.monthrange(m.year, m.month)[1])
        yield max(m, start), min(last, end), last <= end
        m = last + dt.timedelta(days=1)


def build_jobs(bas: dict[str, BA], today: dt.date) -> list[Job]:
    yesterday = today - dt.timedelta(days=1)
    plan = [
        ("previous_runs_full", FULL_START, yesterday),
        ("previous_runs_temp", LONG_START, FULL_START - dt.timedelta(days=1)),
        ("historical_forecast", LONG_START, yesterday),
        ("era5", LONG_START, today - dt.timedelta(days=ERA5_LAG_DAYS)),
    ]
    jobs = []
    for source, s, e in plan:  # priority order: the backtest-critical source first
        for ba in bas:
            for cs, ce, complete in month_chunks(s, e):
                jobs.append(Job(source, ba, cs, ce, complete))
    return jobs


def job_params(job: Job, ba: BA, base_vars: list[str], model: str) -> dict:
    p = {
        "latitude": ",".join(f"{c.lat:.4f}" for c in ba.cities),
        "longitude": ",".join(f"{c.lon:.4f}" for c in ba.cities),
        "hourly": ",".join(job.variables(base_vars)),
        "start_date": job.start.isoformat(),
        "end_date": job.end.isoformat(),
        "timezone": "GMT",
    }
    p["models"] = "era5" if job.source == "era5" else model
    return p


def job_weight(job: Job, ba: BA, base_vars: list[str]) -> float:
    n_days = (job.end - job.start).days + 1
    return call_weight(len(job.variables(base_vars)), n_days, n_locations=len(ba.cities))


def run_jobs(jobs: list[Job], bas: dict[str, BA], root: Path, throttle: Throttle,
             fetch=get_json, max_jobs: int | None = None) -> dict:
    ws = weather_settings()
    base_vars, model = ws["variables"], ws["model"]
    stats = {"fetched": 0, "skipped": 0, "failed": 0, "weight": 0.0}
    for job in jobs:
        path = job.cache_path(root)
        if path.exists():
            stats["skipped"] += 1
            continue
        if max_jobs is not None and stats["fetched"] >= max_jobs:
            break
        ba = bas[job.ba]
        params = job_params(job, ba, base_vars, model)
        w = job_weight(job, ba, base_vars)
        try:
            resp = fetch(ENDPOINTS[job.endpoint], params, throttle, w)
        except Exception as e:  # keep going; a failed chunk is retried next run
            log.warning("failed %s %s %s: %s", job.source, job.ba, job.start, e)
            stats["failed"] += 1
            continue
        payload = {
            "meta": {"source": job.source, "ba": job.ba, "start": job.start.isoformat(),
                     "end": job.end.isoformat(), "complete": job.complete,
                     "cities": [c.name for c in ba.cities], "params": params, "weight": w,
                     "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat()},
            "response": resp,
        }
        write_once(path, payload)
        stats["fetched"] += 1
        stats["weight"] += w
        log.info("fetched %s %s %s..%s (w=%.1f)", job.source, job.ba, job.start, job.end, w)
    return stats


# ---------------------------------------------------------------- cleaning

SOURCE_LABELS = {"previous_runs_full": None, "previous_runs_temp": None,
                 "historical_forecast": "hist_forecast", "era5": "era5"}


def parse_payload(payload: dict, base_vars: list[str]) -> pd.DataFrame:
    """Raw cache payload -> long-by-source rows: ba, city, period_utc, source, <vars>."""
    meta, resp = payload["meta"], payload["response"]
    if len(resp) != len(meta["cities"]):
        raise ValueError(f"{meta['source']} {meta['ba']} {meta['start']}: "
                         f"{len(resp)} locations returned for {len(meta['cities'])} cities")
    frames = []
    for city, loc in zip(meta["cities"], resp):
        h = loc["hourly"]
        t = pd.to_datetime(h["time"]).tz_localize("UTC")  # timezone=GMT; stamp = hour-ending
        if meta["source"].startswith("previous_runs"):
            for d, label in ((1, "prev_day1"), (2, "prev_day2")):
                cols = {v: h.get(f"{v}_previous_day{d}") for v in base_vars}
                frames.append(_frame(meta["ba"], city, t, label, cols, base_vars))
        else:
            cols = {v: h.get(v) for v in base_vars}
            frames.append(_frame(meta["ba"], city, t, SOURCE_LABELS[meta["source"]], cols, base_vars))
    return pd.concat(frames, ignore_index=True)


def _frame(ba, city, t, source, cols, base_vars):
    df = pd.DataFrame({"ba": ba, "city": city, "period_utc": t, "source": source})
    for v in base_vars:
        vals = cols.get(v)
        df[v] = pd.to_numeric(pd.Series(vals if vals is not None else [None] * len(t)), errors="coerce").to_numpy()
    return df


def load_raw(root: Path, base_vars: list[str]) -> pd.DataFrame:
    files = sorted((root / "raw").glob("openmeteo_*/*/*.json"))
    files = [f for f in files if f.parent.parent.name.removeprefix("openmeteo_") in SOURCE_LABELS]
    frames = []
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        df = parse_payload(payload, base_vars)
        df["_retrieved"] = payload["meta"]["retrieved_at_utc"]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ba", "city", "period_utc", "source", *base_vars])
    df = pd.concat(frames, ignore_index=True)
    # Partial-month refreshes overlap earlier copies: keep the newest retrieval per key.
    df = (df.sort_values("_retrieved")
            .drop_duplicates(["ba", "city", "period_utc", "source"], keep="last")
            .drop(columns="_retrieved"))
    # previous_runs_temp yields all-null non-temperature columns; drop fully-empty rows.
    df = df.dropna(subset=base_vars, how="all")
    return df.sort_values(["ba", "source", "city", "period_utc"]).reset_index(drop=True)


def clean(root: Path | None = None) -> pd.DataFrame:
    from .weighting import weight_ba

    root = root or data_dir()
    base_vars = weather_settings()["variables"]
    city = load_raw(root, base_vars)
    if city.empty:
        log.warning("no raw weather files found under %s", root / "raw")
        return city
    write_partitioned(city, "weather_city_hourly", root=root)
    ba = weight_ba(city, load_bas(), base_vars)
    write_partitioned(ba, "weather_ba_hourly", root=root)
    log.info("clean: %d city rows, %d BA rows", len(city), len(ba))
    return ba


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--safety", type=float, default=0.75, help="fraction of free-tier limits to use")
    ap.add_argument("--source", action="append", help="restrict to these sources")
    ap.add_argument("--ba", action="append", help="restrict to these BAs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)

    root = data_dir()
    if args.clean_only:
        clean(root)
        return 0
    bas = load_bas()
    jobs = build_jobs(bas, dt.datetime.now(dt.timezone.utc).date())
    if args.source:
        jobs = [j for j in jobs if j.source in args.source]
    if args.ba:
        jobs = [j for j in jobs if j.ba in args.ba]
    base_vars = weather_settings()["variables"]
    todo = [j for j in jobs if not j.cache_path(root).exists()]
    total = sum(job_weight(j, bas[j.ba], base_vars) for j in todo)
    log.info("%d jobs, %d cached, %d to fetch, %.0f weighted calls", len(jobs),
             len(jobs) - len(todo), len(todo), total)
    if args.dry_run:
        by = {}
        for j in todo:
            by[j.source] = by.get(j.source, 0) + job_weight(j, bas[j.ba], base_vars)
        for k, v in by.items():
            print(f"  {k:22s} {v:8.0f}")
        return 0
    throttle = Throttle(safety=args.safety, ledger=root / "raw" / "openmeteo_ledger.jsonl")
    stats = run_jobs(jobs, bas, root, throttle, max_jobs=args.max_jobs)
    log.info("done: %s", stats)
    if not args.no_clean:
        clean(root)
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
