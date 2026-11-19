"""Storage helpers.

Layout under data_dir():
  raw/<source>/<YYYY-MM-DD>.jsonl            append-only raw API records (never overwritten)
  clean/<table>/ba=<BA>/month=<YYYY-MM>/part.parquet
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from .config import REPO_ROOT, data_dir


def code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def append_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Append records to a JSON-lines file. Returns the number of lines written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str, separators=(",", ":")) + "\n")
            n += 1
    return n


def raw_path(source: str, day: str, root: Path | None = None) -> Path:
    return (root or data_dir()) / "raw" / source / f"{day}.jsonl"


def write_partitioned(df: pd.DataFrame, table: str, time_col: str = "period_utc",
                      root: Path | None = None) -> list[Path]:
    """Write a clean table partitioned by ba and month, replacing those partitions.

    Requires columns `ba` and `time_col` (tz-aware UTC). Adds `code_version`.
    """
    root = (root or data_dir()) / "clean" / table
    df = df.copy()
    df["code_version"] = code_version()
    month = pd.DatetimeIndex(df[time_col]).tz_convert("UTC").strftime("%Y-%m")
    written = []
    for (ba, m), part in df.groupby([df["ba"], month]):
        d = root / f"ba={ba}" / f"month={m}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "part.parquet"
        part.to_parquet(p, index=False)
        written.append(p)
    return written


def query(sql: str, **params) -> pd.DataFrame:
    """Run DuckDB SQL. Use {clean} in SQL as the clean-root placeholder, e.g.
    query("select * from read_parquet('{clean}/eia_hourly/**/*.parquet', hive_partitioning=true)")
    """
    clean = (data_dir() / "clean").as_posix()
    con = duckdb.connect()
    try:
        return con.execute(sql.replace("{clean}", clean), params or None).df()
    finally:
        con.close()
