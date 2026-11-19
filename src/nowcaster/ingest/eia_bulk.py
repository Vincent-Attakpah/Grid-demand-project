"""Download EIA-930 six-month BALANCE files (Grid Monitor bulk downloads).

URL pattern (confirmed 2026-11-18, HTTP 200):
  https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_<YYYY>_<Jan_Jun|Jul_Dec>.csv

Files are final/revised values, not what was visible in real time (plan v2 §2.1).
Usage: python -m nowcaster.ingest.eia_bulk [--start 2021] [--end 2026]
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import requests

from ..config import data_dir

BASE = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles"
HALVES = ("Jan_Jun", "Jul_Dec")


def file_names(start_year: int = 2021, end_year: int = 2026, today: dt.date | None = None) -> list[str]:
    today = today or dt.date.today()
    names = []
    for y in range(start_year, end_year + 1):
        for i, h in enumerate(HALVES):
            first_month = 1 if i == 0 else 7
            if dt.date(y, first_month, 1) > today:
                continue
            names.append(f"EIA930_BALANCE_{y}_{h}.csv")
    return names


def raw_dir() -> Path:
    d = data_dir() / "raw" / "eia930_bulk"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_current_half(name: str, today: dt.date | None = None) -> bool:
    today = today or dt.date.today()
    y, h = name.removeprefix("EIA930_BALANCE_").removesuffix(".csv").split("_", 1)
    return int(y) == today.year and (h == "Jan_Jun") == (today.month <= 6)


def download(name: str, dest: Path, refresh_current: bool = False, retries: int = 3) -> Path:
    path = dest / name
    if path.exists() and path.stat().st_size > 0 and not (refresh_current and is_current_half(name)):
        return path
    tmp = path.with_suffix(".part")
    for attempt in range(retries):
        try:
            with requests.get(f"{BASE}/{name}", stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(path)
            return path
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2021)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--refresh-current", action="store_true",
                    help="re-download the in-progress half-year file")
    a = ap.parse_args(argv)
    dest = raw_dir()
    for n in file_names(a.start, a.end):
        p = download(n, dest, refresh_current=a.refresh_current)
        print(f"{n}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
