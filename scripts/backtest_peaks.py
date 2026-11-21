"""Peak and high-peak-warning backtest (M5), built on saved rolling-origin predictions.

    python scripts/backtest_models.py     # first (writes pred_<BA>.parquet)
    python scripts/backtest_peaks.py

Writes reports/models/peaks.{md,csv} and <data>/backtest/peaks_<BA>.parquet.
Thresholds and climatology for year Y come from Y-2 and Y-1 (2025: 2023-2024; 2026:
2024-2025), all after the 2022 Gulf Power move. Ensembles are stretched by the monthly
calibration factor. See nowcaster.models.peaks for the method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nowcaster.config import REPO_ROOT, data_dir, load_bas
from nowcaster.eval.significance import auto_lag, dm_test, holm
from nowcaster.features.build import load_eia
from nowcaster.features.panel import clean_hourly, op_day
from nowcaster.models.peaks import (actual_peak, inflation_factors, issue_arrays, peak_ensembles, stretch,
                                    summarize, threshold_for, yearly_thresholds)

HIST = ("2023-01-01", "2026-01-01")
EVAL_YEARS = (2025, 2026)
LEFT = [("13-15 h left (morning)", 13, 15), ("7-12 h left (midday)", 7, 12), ("1-6 h left (evening)", 1, 6)]


def history(eia, code, tz):
    """Daily peaks (for thresholds) and remaining-hours peaks by local issue hour (climatology)."""
    act = clean_hourly(eia[eia["ba"] == code])["actual"]
    act = act[(act.index >= HIST[0]) & (act.index < HIST[1])]
    day = pd.to_datetime(op_day(act.index, tz))
    he = (act.index.tz_convert(tz) - pd.Timedelta(hours=1)).hour + 1
    df = pd.DataFrame({"day": day, "he": he, "y": act.to_numpy()})
    daily = df.groupby("day")["y"].max()
    wide = df.pivot_table(index="day", columns="he", values="y", aggfunc="max")
    # remaining peak after local hour h = max over HE > h
    rem = {h: wide.loc[:, [c for c in wide.columns if c > h]].max(axis=1) for h in range(24)}
    return daily, rem


def climatology(rem, thr, year):
    """P(remaining peak > year's threshold) by (month, local issue hour), years Y-2 .. Y-1."""
    out = {}
    for h, s in rem.items():
        s = s.dropna()
        d = pd.DatetimeIndex(s.index)
        s, d = s[(d.year >= year - 2) & (d.year < year)], d[(d.year >= year - 2) & (d.year < year)]
        ev = s.to_numpy() > thr.loc[year].reindex(d.month).to_numpy()
        for mon in range(1, 13):
            sel = d.month == mon
            out[(year, mon, h)] = ev[sel].mean() if sel.any() else np.nan
    return out


def brier(p, y):
    return (p - y) ** 2


def main():
    eia = load_eia()
    frames = []
    for code, ba in load_bas().items():
        f = data_dir() / "backtest" / f"pred_{code}.parquet"
        if not f.exists():
            continue
        pred = pd.read_parquet(f)
        daily, rem = history(eia, code, ba.tz)
        thr = yearly_thresholds(daily, EVAL_YEARS)
        meta, arrs = issue_arrays(pred, ["M3", "B4b", "da"], ba.tz)
        pk = actual_peak(arrs)
        ens, fac = {}, {}
        for name, col in (("M5", "M3"), ("R", "B4b")):
            raw = peak_ensembles(meta, arrs, col)
            fac[name] = inflation_factors(meta, raw, pk)
            ens[name] = stretch(raw, fac[name])
        s = summarize(meta, arrs, ens, thr)
        for name in fac:
            s[f"stretch_{name}"] = fac[name]
        clim = {}
        for y_ in EVAL_YEARS:
            clim.update(climatology(rem, thr, y_))
        s["p_clim"] = [clim.get((d.year, d.month, h), np.nan) for d, h in zip(s["op_day"], s["local_hour"])]
        s["ba"] = code
        s.to_parquet(data_dir() / "backtest" / f"peaks_{code}.parquet", index=False)
        frames.append(s)
        print(code, "issues", len(s), "with ensembles", s["p_M5"].notna().sum(), flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["peak_actual", "p_M5", "p_R", "p_clim", "peak_point_da"])
    df["year"] = df["op_day"].dt.year
    y = df["event"].astype(float)
    for m in ("M5", "R", "clim"):
        df[f"bs_{m}"] = brier(df[f"p_{m}"], y)

    rows = []
    for (year, code), g0 in df.groupby(["year", "ba"]):
        for name, lo, hi in [("all", 1, 15)] + LEFT:
            g = g0[(g0["K"] >= lo) & (g0["K"] <= hi)]
            if g.empty:
                continue
            ev = g["event"].astype(bool)
            warn = g["p_M5"] >= 0.5
            r = {"year": year, "ba": code, "when": name, "issues": len(g), "days": g["op_day"].nunique(),
                 "event_days": g.loc[ev, "op_day"].nunique(), "event_rate": ev.mean(),
                 "threshold_mean": g["threshold"].mean()}
            for c in ("peak_point_da", "peak_point_B4b", "peak_point_M3", "peak_M5"):
                r[f"mae_{c}"] = (g[c] - g["peak_actual"]).abs().mean()
            r["stretch_M5"] = g["stretch_M5"].mean()
            r["cover90_M5"] = ((g["peak_actual"] >= g["lo_M5"]) & (g["peak_actual"] <= g["hi_M5"])).mean()
            for m in ("M5", "R", "clim"):
                r[f"brier_{m}"] = g[f"bs_{m}"].mean()
            r["bss_M5_vs_clim"] = 1 - r["brier_M5"] / r["brier_clim"]
            r["bss_M5_vs_R"] = 1 - r["brier_M5"] / r["brier_R"]
            dd = (g["bs_M5"] - g["bs_R"]).groupby(g["op_day"].to_numpy()).mean().sort_index()
            r["p_vs_R"] = dm_test(dd.to_numpy(), np.zeros(len(dd)), lag=auto_lag(len(dd)))["pvalue"]
            r["hits"] = int((warn & ev).sum())
            r["misses"] = int((~warn & ev).sum())
            r["false_alarms"] = int((warn & ~ev).sum())
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab["p_vs_R_holm"] = tab.groupby(["year", "when"])["p_vs_R"].transform(lambda s: holm(s.to_numpy()))
    out = REPO_ROOT / "reports" / "models"
    tab.to_csv(out / "peaks.csv", index=False)

    fp = lambda p: "n/a" if not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")
    L = ["# Peak forecasts and high-peak warnings (M5)", "",
         "Target: highest demand over the remaining hours of today, from each forecast issued after "
         "the day-ahead forecast is posted. High-peak event: that peak exceeds the 90th percentile of "
         "daily peaks for the calendar month in the two previous years (2025: 2023-2024; 2026: 2024-2025). "
         "Ensembles are stretched each month so the previous 90 days' 90% ranges would have caught 90%. M5 = M3 forecasts + 60 earlier days' whole-day "
         "error paths; R = the same method on Trick 2 (B4b); climatology = 2023-2024 event rate by month "
         "and issue hour over the same two years. Brier score: lower is better. p = DM on daily Brier differences, M5 vs R, Holm "
         "across BAs. Warnings: probability >= 50%.", ""]
    for y_ in sorted(tab["year"].unique()):
        L += [f"## {y_}", "",
              "| BA | when | days | high-peak days | peak MAE: utility fcst | Trick 2 | M3 | M5 | M5 90% range cover "
              "| stretch | Brier M5 | Brier R | Brier clim | skill vs clim | skill vs R | p | hits | misses | false alarms |",
              "|" + "---|" * 19]
        for _, r in tab[tab["year"] == y_].iterrows():
            L.append(f"| {r.ba} | {r.when} | {r.days} | {r.event_days} | {r.mae_peak_point_da:.0f} "
                     f"| {r.mae_peak_point_B4b:.0f} | {r.mae_peak_point_M3:.0f} | {r.mae_peak_M5:.0f} "
                     f"| {r.cover90_M5:.1%} | {r.stretch_M5:.2f} | {r.brier_M5:.4f} | {r.brier_R:.4f} | {r.brier_clim:.4f} "
                     f"| {r.bss_M5_vs_clim:+.1%} | {r.bss_M5_vs_R:+.1%} | {fp(r.p_vs_R_holm)} "
                     f"| {r.hits} | {r.misses} | {r.false_alarms} |")
        L.append("")
    (out / "peaks.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
