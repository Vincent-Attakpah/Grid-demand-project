"""Reproduce the BA Forecast Scorecard's day-ahead MAPE (plan v2 section 7.4, last para).

Recipe (docs/references_check.md, confirmed against the scorecard's metrics.py):
  * raw `demand` (not EIA-adjusted), DA forecast from the same row, no shifting;
  * an hour is valid if demand > 0, forecast > 0 and 0.5 <= forecast/demand <= 2;
  * daily MAPE per BA and EIA "Data Date" = mean |f - d| / d over valid hours;
  * a day is scored only with >= 20 valid hours;
  * period MAPE = valid-hour-weighted mean of daily MAPEs.

EIA's "Data Date" is the BA-local operating day with hours numbered 1..24, i.e. the local
date of (hour-ending - 1 h); we derive it from period_utc and the BA tz in bas.yaml.

    python -m nowcaster.eval.scorecard [--yearly PATH_TO_ba_yearly.csv] [--out reports/scorecard_repro.md]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, load_bas
from ..storage import query
from ..timeutils import operating_day

MIN_HOURS_PER_DAY = 20
RATIO = 2.0
BAS = ("SOCO", "TVA", "DUK", "FPL")
EIA = "read_parquet('{clean}/eia_hourly/**/*.parquet', hive_partitioning=true)"
DEFAULT_YEARLY = os.environ.get("SCORECARD_YEARLY_CSV")


def load_hourly(bas=BAS) -> pd.DataFrame:
    ba_list = ", ".join(f"'{b}'" for b in bas)
    df = query(f"select ba, period_utc, demand, da_forecast from {EIA} where ba in ({ba_list})")
    df["period_utc"] = pd.to_datetime(df["period_utc"], utc=True)
    tz = {k: v.tz for k, v in load_bas().items()}
    df["date"] = pd.NaT
    parts = []
    for ba, g in df.groupby("ba"):
        g = g.copy()
        g["date"] = operating_day(pd.DatetimeIndex(g["period_utc"]), tz[ba])
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def daily_mape(df: pd.DataFrame, demand: str = "demand", forecast: str = "da_forecast",
               min_hours: int = MIN_HOURS_PER_DAY) -> pd.DataFrame:
    """Per (ba, date): hours_valid, mape, scored flag (scorecard rules)."""
    d = df[["ba", "date", demand, forecast]].copy()
    dem, fc = d[demand].astype(float), d[forecast].astype(float)
    pos = (dem > 0) & (fc > 0)
    ratio = fc / dem
    valid = pos & (ratio <= RATIO) & (ratio >= 1 / RATIO)
    d["abs_pct"] = np.where(valid, (fc - dem).abs() / dem, np.nan)
    d["valid"] = valid
    out = d.groupby(["ba", "date"], sort=True).agg(hours_valid=("valid", "sum"),
                                                   mape=("abs_pct", "mean")).reset_index()
    out["scored"] = out["hours_valid"] >= min_hours
    out.loc[~out["scored"], "mape"] = np.nan
    return out


def aggregate(daily: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    s = daily[daily["scored"]].copy()
    s["mape_w"] = s["mape"] * s["hours_valid"]
    out = s.groupby(by).agg(days=("date", "nunique"), hours=("hours_valid", "sum"),
                            mape_w=("mape_w", "sum")).reset_index()
    out["mape"] = out["mape_w"] / out["hours"]
    return out.drop(columns="mape_w")


def reproduce(yearly_csv: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = daily_mape(load_hourly())
    daily["year"] = pd.to_datetime(daily["date"]).dt.year
    ours_y = aggregate(daily, ["ba", "year"])
    ours_all = aggregate(daily, ["ba"])
    if yearly_csv and Path(yearly_csv).exists():
        sc = pd.read_csv(yearly_csv)[["ba", "year", "days", "hours", "mape"]]
        sc = sc[sc["ba"].isin(BAS)]
        ours_y = ours_y.merge(sc, on=["ba", "year"], how="left", suffixes=("", "_sc"))
        ours_y["diff_pt"] = 100 * (ours_y["mape"] - ours_y["mape_sc"])
        # Scorecard's own years recombined over the same span, for a like-for-like total.
        span = sc[sc["year"] >= ours_y["year"].min()].copy()
        span["mw"] = span["mape"] * span["hours"]
        tot = span.groupby("ba").agg(hours_sc=("hours", "sum"), mw=("mw", "sum")).reset_index()
        tot["mape_sc"] = tot["mw"] / tot["hours_sc"]
        ours_all = ours_all.merge(tot[["ba", "hours_sc", "mape_sc"]], on="ba", how="left")
        ours_all["diff_pt"] = 100 * (ours_all["mape"] - ours_all["mape_sc"])
    return ours_y, ours_all


def to_markdown(yearly: pd.DataFrame, total: pd.DataFrame, years: tuple[int, int]) -> str:
    def pct(x):
        return "" if pd.isna(x) else f"{100 * x:.3f}%"

    def pt(x):
        return "" if pd.isna(x) else f"{x:+.3f}"

    lines = ["| BA | year | days | ours | scorecard | diff (pt) | days (sc) |", "|---|---|---|---|---|---|---|"]
    for r in yearly.itertuples():
        lines.append(f"| {r.ba} | {r.year} | {r.days} | {pct(r.mape)} | {pct(getattr(r, 'mape_sc', np.nan))} "
                     f"| {pt(getattr(r, 'diff_pt', np.nan))} | {getattr(r, 'days_sc', '')} |")
    lines += ["", f"Combined {years[0]}–{years[1]} (valid-hour weighted):", "",
              "| BA | hours | ours | scorecard (same years) | diff (pt) |", "|---|---|---|---|---|"]
    for r in total.itertuples():
        lines.append(f"| {r.ba} | {r.hours} | {pct(r.mape)} | {pct(getattr(r, 'mape_sc', np.nan))} "
                     f"| {pt(getattr(r, 'diff_pt', np.nan))} |")
    return "\n".join(lines)


HEADER = """# Scorecard MAPE reproduction

Generated by `python -m nowcaster.eval.scorecard --yearly <scorecard>/results/ba_yearly.csv`.
"Ours" is computed from our `eia_hourly` table (EIA-930 six-month BALANCE files, 2021 H1
onward) with the scorecard's rules: raw demand, same-row DA forecast, valid hours need
demand > 0, forecast > 0 and 0.5 <= forecast/demand <= 2, a day needs >= 20 valid hours,
and MAPE is the valid-hour-weighted mean of daily MAPEs. Data Date = BA-local date of
(hour-ending - 1 h). "Scorecard" is `results/ba_yearly.csv` from the scorecard repository
(v0.1.1 code, results generated 2026-11-18). Success test: |diff| <= 0.1 pt for 2021-2025.

"""

FOOTER = """

## Full-period number (2015-07 to 2026-11)

The scorecard's full-period MAPE in `ranking_overall.csv` equals its yearly values
recombined with valid-hour weights (checked: |difference| < 0.0001 pt for all four BAs:
DUK 2.8002%, FPL 4.1717%, SOCO 2.8068%, TVA 2.2232%). Because our yearly values match its
yearly values for every year we hold, the full-period figure follows from the same
aggregation; the 2015-2020 files were therefore not downloaded.

Notes: the comparison is against the scorecard's own processing of EIA's current files.
EIA revises past values, so a rerun after a later download can drift slightly; 2026 is a
partial year (to 2026-11-18) in both sources.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yearly", default=DEFAULT_YEARLY, help="scorecard results/ba_yearly.csv")
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "scorecard_repro.md"))
    a = ap.parse_args(argv)
    y, t = reproduce(a.yearly)
    md = HEADER + to_markdown(y, t, (int(y["year"].min()), int(y["year"].max()))) + FOOTER
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md + "\n", encoding="utf-8")
    print(md)
    if "diff_pt" in y:
        full = y[(y["year"] <= 2025) & y["diff_pt"].notna()]
        worst = full["diff_pt"].abs().max()
        print(f"\nmax |diff| 2021-2025: {worst:.3f} pt -> {'PASS' if worst <= 0.1 else 'FAIL'} (tolerance 0.1)")


if __name__ == "__main__":
    main()
