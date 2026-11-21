"""One live run: `python -m nowcaster.live.run --root <live branch checkout> --bundle bundle.pkl.gz`.

Steps: work out the issue time (the latest :17), pull 65 days of EIA demand + day-ahead
forecasts and each BA's weather forecast, forecast every remaining hour of today, compute
the peak warning, append both to the history, re-score past forecasts against published
demand, and rewrite the dashboard.

Layout under --root (the `live` branch in production):
  history/<UTC issue date>.csv   hourly forecasts, one row per (BA, issue, target hour)
  peaks/<UTC issue date>.csv     one row per (BA, issue): peak forecast and warning
  site/index.html                dashboard
  site/scores.json               live scores
  actuals.csv                    published actual demand (label), kept ACTUALS_DAYS days, so
                                 past forecasts can still be scored after leaving the EIA pull
  status.json                    last run: issue time, errors, model bundle date
A run whose issue time is already in the history writes nothing new (re-runs are safe).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

from ..config import load_bas
from ..features.panel import Lags, clean_hourly
from ..logger.run import redact
from . import inputs, score, site
from .forecast import HIST_COLS, LIB_WINDOW_DAYS, forecast_issue, issue_time, library, peak_now
from .train import load

ACTUALS_DAYS = 200
PEAK_COLS = ["ba", "issue_utc", "op_day", "K", "local_hour", "threshold", "stretch", "n_lib", "peak_M5",
             "lo_M5", "hi_M5", "p_M5", "peak_point_M3", "peak_point_B4b", "peak_point_da"]


def read_dir(d: Path, since: pd.Timestamp) -> pd.DataFrame:
    files = sorted(f for f in d.glob("*.csv") if f.stem >= since.strftime("%Y-%m-%d"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ("issue_utc", "target_utc"):
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True)
    return df


def append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for c in ("issue_utc", "target_utc"):
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df.to_csv(path, mode="a", header=not path.exists(), index=False, float_format="%.1f", lineterminator="\n")


def read_actuals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ba", "period_utc", "actual"])
    a = pd.read_csv(path)
    a["period_utc"] = pd.to_datetime(a["period_utc"], utc=True)
    return a


def merge_actuals(old: pd.DataFrame, code: str, new: pd.Series) -> pd.Series:
    """This BA's actuals: newly pulled values win, archived ones fill the rest."""
    o = old[old["ba"] == code].set_index("period_utc")["actual"].astype(float)
    o.index = pd.DatetimeIndex(pd.to_datetime(o.index, utc=True))
    out = new.dropna().astype(float).combine_first(o).sort_index()
    out.index = pd.DatetimeIndex(out.index)
    return out


def write_actuals(path: Path, series: dict[str, pd.Series], now: pd.Timestamp) -> None:
    rows = [pd.DataFrame({"ba": c, "period_utc": s.index, "actual": s.to_numpy()}) for c, s in series.items()]
    a = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ba", "period_utc", "actual"])
    a["period_utc"] = pd.to_datetime(a["period_utc"], utc=True)
    a = a[a["period_utc"] >= now - pd.Timedelta(days=ACTUALS_DAYS)].dropna()
    a["period_utc"] = a["period_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    a.to_csv(path, index=False, float_format="%.1f", lineterminator="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--now", default=None, help="override the clock (testing)")
    a = ap.parse_args(argv)

    now = pd.Timestamp(a.now, tz="UTC") if a.now else pd.Timestamp.now(tz="UTC")
    tau = issue_time(now)
    root = a.root
    bundle = load(a.bundle)
    bas = {c: b for c, b in load_bas().items() if c in bundle["bas"]}
    lags = Lags.from_config()
    session = requests.Session()
    errors: dict[str, str] = {}

    since = tau.normalize() - pd.Timedelta(days=LIB_WINDOW_DAYS)
    hist = read_dir(root / "history", since)
    peaks = read_dir(root / "peaks", since)
    done = set() if hist.empty else set(hist.loc[hist["issue_utc"] == tau, "ba"])

    archive = read_actuals(root / "actuals.csv")
    kept: dict[str, pd.Series] = {}
    eia = inputs.fetch_eia(now.to_pydatetime(), bas, session=session)
    new_rows, new_peaks, cards = [], [], []
    for code, ba in bas.items():
        models = bundle["bas"][code]
        e_ba = eia[eia["ba"] == code]
        hourly = clean_hourly(e_ba) if not e_ba.empty else None
        pulled = hourly["actual"] if hourly is not None else pd.Series(dtype=float)
        actual = kept[code] = merge_actuals(archive, code, pulled)
        h_ba = hist[hist["ba"] == code] if not hist.empty else pd.DataFrame()
        p_ba = peaks[peaks["ba"] == code] if not peaks.empty else pd.DataFrame()
        rows, pk = None, None
        try:
            if code in done:
                rows = h_ba[h_ba["issue_utc"] == tau]
                m = p_ba["issue_utc"] == tau if not p_ba.empty else []
                pk = p_ba[m].iloc[0].to_dict() if len(m) and m.any() else None
            else:
                wba = inputs.fetch_weather(code, ba, session=session)
                rows = forecast_issue(models, e_ba, wba, code, ba.tz, tau, lags)
                lib = library(models["seed"], h_ba, actual, since)
                pk = peak_now(rows, lib, models["daily_peaks"], ba.tz)
                new_rows.append(rows[HIST_COLS])
                h_ba = pd.concat([h_ba, rows[HIST_COLS]], ignore_index=True)
                if pk:
                    new_peaks.append(pd.DataFrame([pk])[PEAK_COLS])
                    p_ba = pd.concat([p_ba, new_peaks[-1]], ignore_index=True)
        except Exception as ex:                       # keep the other BAs going
            errors[code] = redact(f"{type(ex).__name__}: {ex}")
        sc = score.hourly_scores(h_ba, actual, now) if len(h_ba) else {}
        ps = score.peak_scores(p_ba, h_ba, actual, now) if len(p_ba) and len(h_ba) else {}
        hours = site.day_hours(tau.tz_convert(ba.tz).strftime("%Y-%m-%d"), ba.tz)
        util = hourly["da"] if hourly is not None else pd.Series(dtype=float)
        cards.append({"code": code, "name": ba.name, "tz": ba.tz, "issue_utc": tau, "hours": hours,
                      "actual": actual, "util": util, "forecast": rows, "peak": pk,
                      "scores": sc, "peak_scores": ps, "error": errors.get(code)})

    stamp = tau.strftime("%Y-%m-%d")
    if new_rows:
        append_csv(root / "history" / f"{stamp}.csv", pd.concat(new_rows, ignore_index=True))
    if new_peaks:
        append_csv(root / "peaks" / f"{stamp}.csv", pd.concat(new_peaks, ignore_index=True))
    root.mkdir(parents=True, exist_ok=True)
    write_actuals(root / "actuals.csv", kept, now)
    status = {"issue_utc": tau.isoformat(), "ran_at_utc": pd.Timestamp.now(tz="UTC").floor("s").isoformat(),
              "trained_at": bundle.get("trained_at", ""), "errors": errors,
              "new_forecasts": int(sum(len(r) for r in new_rows)), "already_issued": sorted(done)}
    (root / "site").mkdir(parents=True, exist_ok=True)
    (root / "site" / "index.html").write_text(site.render(cards, status), encoding="utf-8")
    scores = {c["code"]: {"hourly": c["scores"], "peaks": c["peak_scores"]} for c in cards}
    (root / "site" / "scores.json").write_text(json.dumps(scores, indent=1, default=str), encoding="utf-8")
    (root / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    print(json.dumps(status))
    return 1 if errors and len(errors) == len(bas) else 0


if __name__ == "__main__":
    sys.exit(main())
