"""One logger run: `python -m nowcaster.logger.run [--data-root DIR]`.

Overlapping-window design: every run re-pulls the full window (EIA: last 72 h plus
forward-dated values; Open-Meteo: past 1 day + next 2 days) and appends only rows whose
value differs from the latest value already logged for that key. So
  * an identical re-pull writes nothing,
  * a revised EIA value or a changed weather forecast is a new row (with its own
    retrieved_at_utc), and
  * a late or dropped run is filled in by the next one.

Layout under the data root (the `data-log` branch in production):
  raw/eia/<UTC retrieval day>.jsonl
  raw/openmeteo/<UTC retrieval day>.jsonl
  raw/openmeteo_meta/<UTC retrieval day>.jsonl
  state/index.json      latest value signature per key (pruned to the live window)
  runs.jsonl            one record per run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import uuid
from pathlib import Path

import requests

from ..config import weather_settings
from . import eia, openmeteo
from .common import append_jsonl, floor_hour, iso, parse_iso, utcnow, write_json_atomic

# Keys older than this (relative to now) are dropped from the index: they fall outside
# every future pull window, so they can never be re-fetched.
RETENTION_HOURS = {"eia": 24 * 5, "openmeteo": 48}


class Index:
    """Latest-value index: {source: {key: [signature, key_time_iso]}}."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict[str, list[str]]] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)

    def filter_new(self, source: str, rows, keyfn) -> tuple[list[dict], dict]:
        """Rows whose value differs from the latest logged one, plus the index updates to
        apply (via commit) once those rows are safely written."""
        idx = self.data.get(source, {})
        new, updates = [], {}
        for r in rows:
            k, sig, ktime = keyfn(r)
            prev = updates.get(k) or idx.get(k)
            if prev is None or prev[0] != sig:
                updates[k] = [sig, ktime]
                new.append(r)
        return new, updates

    def commit(self, source: str, updates: dict) -> None:
        self.data.setdefault(source, {}).update(updates)

    def prune(self, now: dt.datetime) -> None:
        for source, idx in self.data.items():
            cutoff = now - dt.timedelta(hours=RETENTION_HOURS.get(source, 120))
            for k in [k for k, (_, t) in idx.items() if parse_iso(t) < cutoff]:
                del idx[k]

    def save(self) -> None:
        write_json_atomic(self.path, self.data)


def scheduled_hour(start: dt.datetime, offset_min: int) -> dt.datetime:
    """Hour of the cron slot this run belongs to (cron fires at :offset_min).
    A run delayed by more than ~(60 - offset) min is attributed to the next slot."""
    return floor_hour(start - dt.timedelta(minutes=offset_min))


def _stamp(rows, run_id: str, retrieved_at: dt.datetime, sched: dt.datetime, source: str):
    for r in rows:
        r["src"] = source
        r["retrieved_at_utc"] = iso(retrieved_at)
        r["sched_hour_utc"] = iso(sched)
        r["run_id"] = run_id
    return rows


def run(data_root: Path, *, now_fn=utcnow, session=None, eia_kwargs=None, om_bas=None,
        offset_min: int = 17, heartbeat_url: str | None = None) -> dict:
    session = session or requests.Session()
    start = now_fn()
    sched = scheduled_hour(start, offset_min)
    run_id = os.environ.get("GITHUB_RUN_ID", uuid.uuid4().hex[:12])
    index = Index(data_root / "state" / "index.json")
    variables = weather_settings()["variables"]
    record: dict = {"run_id": run_id, "started_at_utc": iso(start), "sched_hour_utc": iso(sched),
                    "delay_min": round((start - sched - dt.timedelta(minutes=offset_min)).total_seconds() / 60, 1),
                    "sources": {}}

    def do(source: str, fetcher, keyfn):
        t = now_fn()
        try:
            rows, extra = fetcher(t)
            new, updates = index.filter_new(source, rows, keyfn)
            _stamp(new, run_id, t, sched, source)
            n = append_jsonl(data_root / "raw" / source / f"{t:%Y-%m-%d}.jsonl", new)
            index.commit(source, updates)
            record["sources"][source] = {"status": "ok", "retrieved_at_utc": iso(t),
                                         "rows_fetched": len(rows), "rows_written": n, **extra}
        except Exception as e:
            record["sources"][source] = {"status": "error", "retrieved_at_utc": iso(t),
                                         "error": f"{type(e).__name__}: {e}"[:500]}

    do("eia", lambda t: (eia.fetch(t, session=session, **(eia_kwargs or {})), {}), eia.key_and_sig)

    def om_fetch(t):
        rows, errs = openmeteo.fetch(t, bas=om_bas, session=session)
        return rows, ({"ba_errors": errs} if errs else {})
    do("openmeteo", om_fetch, lambda r: openmeteo.key_and_sig(r, variables))

    # Model-run metadata: small, written every run as a timeline (non-critical).
    try:
        t = now_fn()
        meta = openmeteo.fetch_meta(weather_settings()["model"], session=session)
        append_jsonl(data_root / "raw" / "openmeteo_meta" / f"{t:%Y-%m-%d}.jsonl",
                     [{"retrieved_at_utc": iso(t), "run_id": run_id, "meta": meta}])
    except Exception as e:
        record["meta_error"] = f"{type(e).__name__}: {e}"[:300]

    index.prune(start)
    index.save()
    ok = any(s["status"] == "ok" for s in record["sources"].values())
    record["ok"] = ok
    if ok and heartbeat_url:
        try:
            session.get(heartbeat_url, timeout=10)
            record["heartbeat"] = "sent"
        except Exception as e:
            record["heartbeat"] = f"failed: {e}"[:200]
    record["ended_at_utc"] = iso(now_fn())
    append_jsonl(data_root / "runs.jsonl", [record])
    return record


def default_root() -> Path:
    if os.environ.get("NOWCASTER_LOG_ROOT"):
        return Path(os.environ["NOWCASTER_LOG_ROOT"])
    from ..config import data_dir
    return data_dir() / "log"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--offset-min", type=int, default=17, help="cron minute of the schedule")
    args = ap.parse_args(argv)
    root = args.data_root or default_root()
    rec = run(root, offset_min=args.offset_min, heartbeat_url=os.environ.get("HEARTBEAT_URL") or None)
    print(json.dumps(rec, indent=1))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
