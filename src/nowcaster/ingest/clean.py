"""Clean EIA-930 six-month BALANCE files into the `eia_hourly` table.

Verified against the files (2021 H1 .. 2026 H2, downloaded 2026-11-18):
- 44 columns through 2024 H1, 65 from 2024 H2 (fuel-type columns added); the 16
  balance columns used here have identical names in both layouts.
- "UTC Time at End of Hour" is hour-ending UTC, format "MM/DD/YYYY H:MM:SS AM".
- "(Imputed)" columns hold the imputed MW value when EIA imputed, blank otherwise,
  so the imputed flag is `notna()`.
- "(Adjusted)" columns are EIA's adjusted series.
- Local time at end of hour shows SOCO and TVA in Central, DUK and FPL in Eastern,
  matching config/bas.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..storage import write_partitioned
from .eia_bulk import raw_dir

TARGET_BAS = ("SOCO", "TVA", "DUK", "FPL", "ERCO")

COLS = {
    "Balancing Authority": "ba",
    "UTC Time at End of Hour": "period_utc",
    "Local Time at End of Hour": "period_local",
    "Demand Forecast (MW)": "da_forecast",
    "Demand (MW)": "demand",
    "Net Generation (MW)": "net_generation",
    "Total Interchange (MW)": "total_interchange",
    "Demand (MW) (Imputed)": "demand_imputed_value",
    "Net Generation (MW) (Imputed)": "net_generation_imputed_value",
    "Total Interchange (MW) (Imputed)": "total_interchange_imputed_value",
    "Demand (MW) (Adjusted)": "demand_adjusted",
    "Net Generation (MW) (Adjusted)": "net_generation_adjusted",
    "Total Interchange (MW) (Adjusted)": "total_interchange_adjusted",
}
NUMERIC = [v for k, v in COLS.items() if "(MW)" in k]
TS_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(",", "", regex=False).str.strip(),
                         errors="coerce")


def read_balance_file(path: Path, bas=TARGET_BAS) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(COLS), dtype=str)
    df = df.rename(columns=COLS)
    df = df[df["ba"].isin(bas)].copy()
    return tidy(df, source_file=Path(path).name)


def tidy(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    df = df.copy()
    df["period_utc"] = pd.to_datetime(df["period_utc"], format=TS_FORMAT).dt.tz_localize("UTC")
    for c in NUMERIC:
        df[c] = _to_num(df[c])
    for base in ("demand", "net_generation", "total_interchange"):
        df[f"{base}_imputed"] = df[f"{base}_imputed_value"].notna()
        df = df.drop(columns=f"{base}_imputed_value")
    df["source_file"] = source_file
    return df


def build(bas=TARGET_BAS, root: Path | None = None) -> pd.DataFrame:
    files = sorted(raw_dir().glob("EIA930_BALANCE_*.csv"))
    if not files:
        raise FileNotFoundError("No bulk files; run python -m nowcaster.ingest.eia_bulk first")
    parts = [read_balance_file(f, bas) for f in files]
    df = pd.concat(parts, ignore_index=True)
    # Later files win if a period appears twice (they hold the latest revisions).
    df = df.sort_values(["ba", "period_utc", "source_file"]).drop_duplicates(
        ["ba", "period_utc"], keep="last").reset_index(drop=True)
    write_partitioned(df.drop(columns=["period_local"]), "eia_hourly", root=root)
    return df


def main(argv=None):
    argparse.ArgumentParser().parse_args(argv)
    df = build()
    print(df.groupby("ba")["period_utc"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
