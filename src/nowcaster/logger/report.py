"""Logger report: `python -m nowcaster.logger.report [--data-root DIR] [--out FILE]`.

Answers plan v2 section 2.3 from the logged data:
  1. demand lag: hour end -> first retrieval of a non-null D value, per BA;
  2. clock time at which each operating day's DF (day-ahead forecast) first appears;
  3. revisions: first-reported vs latest value;
  4. whether DF values for *tomorrow* (local operating day) are ever exposed.

Caveat: runs fire hourly (and GitHub Actions can start them late), so a first-appearance
time is an UPPER bound; the previous successful run gives a LOWER bound. Both are
reported. Times use retrieved_at_utc (actual request time), never the schedule.
Only items that appeared after the logger's first successful EIA run are used for
lag/first-appearance, so pre-existing history does not masquerade as long lags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_bas
from .run import default_root


def load_jsonl(paths) -> pd.DataFrame:
    recs = []
    for p in sorted(paths):
        with open(p, encoding="utf-8") as f:
            recs += [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(recs)


def load_eia(root: Path) -> pd.DataFrame:
    df = load_jsonl((root / "raw" / "eia").glob("*.jsonl"))
    if df.empty:
        return df
    df["period_utc"] = pd.to_datetime(df["period_utc"], utc=True)
    df["retrieved_at_utc"] = pd.to_datetime(df["retrieved_at_utc"], utc=True)
    return df


def eia_run_times(root: Path) -> pd.Series:
    p = root / "runs.jsonl"
    if not p.exists():
        return pd.Series(dtype="datetime64[ns, UTC]")
    runs = load_jsonl([p])
    t = [r["eia"]["retrieved_at_utc"] for r in runs["sources"] if r.get("eia", {}).get("status") == "ok"]
    return pd.Series(pd.to_datetime(t, utc=True)).sort_values().reset_index(drop=True)


def _prev_run(times: pd.Series, t: pd.Timestamp):
    i = times.searchsorted(t, side="left") - 1
    return times.iloc[i] if i >= 0 else pd.NaT


def demand_lag(eia: pd.DataFrame, run_times: pd.Series) -> pd.DataFrame:
    d = eia[(eia["type"] == "D") & eia["value"].notna()]
    if d.empty:
        return pd.DataFrame()
    first = d.groupby(["ba", "period_utc"], as_index=False)["retrieved_at_utc"].min()
    logger_start = run_times.min() if len(run_times) else eia["retrieved_at_utc"].min()
    first = first[first["period_utc"] > logger_start]
    first["lag_upper_h"] = (first["retrieved_at_utc"] - first["period_utc"]).dt.total_seconds() / 3600
    prev = first["retrieved_at_utc"].map(lambda t: _prev_run(run_times, t))
    first["lag_lower_h"] = ((prev - first["period_utc"]).dt.total_seconds() / 3600).clip(lower=0)
    return first.groupby("ba").agg(
        n_hours=("lag_upper_h", "size"),
        lag_upper_median_h=("lag_upper_h", "median"),
        lag_upper_p95_h=("lag_upper_h", lambda s: s.quantile(0.95)),
        lag_lower_median_h=("lag_lower_h", "median"),
        lag_lower_p95_h=("lag_lower_h", lambda s: s.quantile(0.95)),
    ).round(2)


def da_first_appearance(eia: pd.DataFrame, run_times: pd.Series, tz_by_ba: dict) -> pd.DataFrame:
    """Per BA and local operating day: when the first DF value for that day appeared."""
    df = eia[(eia["type"] == "DF") & eia["value"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    logger_start = run_times.min() if len(run_times) else eia["retrieved_at_utc"].min()
    out = []
    for ba, g in df.groupby("ba"):
        tz = tz_by_ba.get(ba, "UTC")
        g = g.assign(op_day=(g["period_utc"].dt.tz_convert(tz) - pd.Timedelta(hours=1)).dt.date)
        first = g.groupby("op_day")["retrieved_at_utc"].min().reset_index()
        first = first[first["retrieved_at_utc"] > logger_start + pd.Timedelta(minutes=1)]
        for _, r in first.iterrows():
            loc = r["retrieved_at_utc"].tz_convert(tz)
            prev = _prev_run(run_times, r["retrieved_at_utc"])
            out.append({"ba": ba, "op_day": r["op_day"], "first_seen_local": loc.strftime("%Y-%m-%d %H:%M"),
                        "prev_run_local": prev.tz_convert(tz).strftime("%Y-%m-%d %H:%M") if pd.notna(prev) else None,
                        "seen_before_day_start": loc.date() < r["op_day"]})
    return pd.DataFrame(out)


def tomorrow_exposure(eia: pd.DataFrame, tz_by_ba: dict) -> pd.DataFrame:
    """Count DF rows whose operating day is later than the retrieval's local day."""
    df = eia[(eia["type"] == "DF") & eia["value"].notna()]
    rows = []
    for ba, g in df.groupby("ba"):
        tz = tz_by_ba.get(ba, "UTC")
        op_day = (g["period_utc"].dt.tz_convert(tz) - pd.Timedelta(hours=1)).dt.date
        ret_day = g["retrieved_at_utc"].dt.tz_convert(tz).dt.date
        ahead = g[op_day > ret_day]
        rows.append({"ba": ba, "df_rows": len(g), "rows_for_future_days": len(ahead),
                     "future_days_seen": int(pd.Series(op_day[op_day > ret_day]).nunique())})
    return pd.DataFrame(rows)


def revisions(eia: pd.DataFrame) -> pd.DataFrame:
    d = eia[eia["value"].notna()].sort_values("retrieved_at_utc")
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(["ba", "type", "period_utc"])["value"]
    first, last, n = g.first(), g.last(), g.size()
    t = pd.DataFrame({"first": first, "last": last, "n_versions": n}).reset_index()
    t["abs_diff"] = (t["last"] - t["first"]).abs()
    t["abs_pct"] = np.where(t["first"].abs() > 0, 100 * t["abs_diff"] / t["first"].abs(), np.nan)
    return t.groupby(["ba", "type"]).agg(
        n_hours=("abs_diff", "size"),
        share_revised=("n_versions", lambda s: (s > 1).mean()),
        median_abs_diff=("abs_diff", "median"),
        p95_abs_diff=("abs_diff", lambda s: s.quantile(0.95)),
        share_gt_1pct=("abs_pct", lambda s: (s > 1).mean()),
    ).round(4)


def _tbl(df: pd.DataFrame, index: bool = True) -> str:
    if df.empty:
        return '(no data)'
    return "```\n" + df.to_string(index=index) + "\n```"


def build(root: Path) -> str:
    eia = load_eia(root)
    if eia.empty:
        return f"No EIA log rows under {root}."
    run_times = eia_run_times(root)
    tz_by_ba = {k: v.tz for k, v in load_bas(include_logging_only=True).items()}
    parts = [
        "# Logger report",
        f"Data root: `{root}`  ",
        f"EIA rows: {len(eia)}; successful EIA runs: {len(run_times)}; "
        f"span {eia['retrieved_at_utc'].min()} to {eia['retrieved_at_utc'].max()}",
        "",
        "First-appearance times are upper bounds (hourly runs, Actions start delays); "
        "the previous successful run is the lower bound.",
        "",
        "## 1. Demand lag (hours after hour end)",
        _tbl(demand_lag(eia, run_times)),
        "",
        "## 2. Day-ahead forecast first appearance (BA local time)",
    ]
    fa = da_first_appearance(eia, run_times, tz_by_ba)
    parts.append(_tbl(fa, index=False) if not fa.empty else "None observed after logger start yet.")
    parts += ["", "## 3. Revisions (first-reported vs latest)", _tbl(revisions(eia)), "",
              "## 4. DF values for future operating days", _tbl(tomorrow_exposure(eia, tz_by_ba), index=False)]
    return "\n".join(str(p) for p in parts) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Logger report (plan v2 section 2.3)")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    text = build(a.data_root or default_root())
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
