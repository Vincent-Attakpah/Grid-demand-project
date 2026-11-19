"""Shared helpers for the logger. Stdlib + requests only, so the hourly Action stays fast
(no pandas/duckdb import on the hot path)."""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Iterable

import requests

UTC = dt.timezone.utc


def utcnow() -> dt.datetime:
    return dt.datetime.now(UTC)


def iso(t: dt.datetime) -> str:
    """ISO-8601 UTC with a Z suffix, second precision."""
    return t.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def floor_hour(t: dt.datetime) -> dt.datetime:
    return t.replace(minute=0, second=0, microsecond=0)


def get_json(session, url: str, params, retries: int = 3, timeout: float = 30.0, backoff: float = 2.0):
    """GET with retries on timeouts, connection errors, 429 and 5xx."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                last_exc = requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
            else:
                r.raise_for_status()
                return r.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def append_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
            n += 1
    return n


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, separators=(",", ":"), sort_keys=True)
    os.replace(tmp, path)
