"""What does weather add? M3 vs the same model with inputs removed (plan v2 section 9.4).

    python scripts/ablation_weather.py            # backtest (about 30 min) + report
    python scripts/ablation_weather.py --report   # report only, from saved predictions

Steps compared on identical rows (issues after DA posting, 2025-01 to date):
  B4b            Trick 2: DA + latest error + same-hour-yesterday error
  M3_demandOnly  M3 with only the demand/DA information Trick 2 has (no weather, no calendar)
  M3_noWx        + calendar
  M3             + weather (the full model)
Each step is tested against the one before (Diebold-Mariano on daily MAE, Holm across BAs).
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from nowcaster.baselines.simple import BASELINES
from nowcaster.config import REPO_ROOT, data_dir, load_bas
from nowcaster.eval.rolling import rolling_origin
from nowcaster.eval.significance import dm_daily, holm
from nowcaster.features.build import FULL_WEATHER_START, load_panel, panel_path
from nowcaster.models.gbm import ABLATIONS, MODELS

STEPS = ["B4b", "M3_demandOnly", "M3_noWx", "M3"]
LABEL = {"M3_demandOnly": "learning from the same demand data (vs Trick 2)",
         "M3_noWx": "adding the calendar", "M3": "adding weather"}
BUCKETS = [("1-3", 1, 3), ("4-6", 4, 6), ("1-6", 1, 6), ("7-12", 7, 12), ("13-24", 13, 25)]
KEEP = ["ba", "issue_utc", "target_utc", "op_day", "horizon", "actual"]


def run(start, end):
    models = {"B4b": BASELINES["B4b"], "M3": MODELS["M3"], **ABLATIONS}
    out = data_dir() / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    for code in [c for c in load_bas() if panel_path(c).exists()]:
        t0 = time.time()
        p = load_panel(code)
        p = p[p["da_posted"]].reset_index(drop=True)
        res = rolling_origin(p, models, start, end, train_start=FULL_WEATHER_START)
        res[KEEP + STEPS].to_parquet(out / f"abl_{code}.parquet", index=False)
        print(f"{code}: {len(res)} rows, {time.time() - t0:.0f}s", flush=True)


def report():
    files = sorted((data_dir() / "backtest").glob("abl_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["actual", *STEPS])
    df["year"] = pd.DatetimeIndex(df["issue_utc"]).year
    rows = []
    for (year, ba), g0 in df.groupby(["year", "ba"]):
        for name, lo, hi in BUCKETS:
            g = g0[(g0["horizon"] >= lo) & (g0["horizon"] <= hi)]
            r = {"year": year, "ba": ba, "bucket": name, "days": g["op_day"].nunique()}
            for m in STEPS:
                r[f"MAE_{m}"] = (g[m] - g["actual"]).abs().mean()
            for prev, m in zip(STEPS, STEPS[1:]):
                r[f"gain_{m}"] = 1 - r[f"MAE_{m}"] / r[f"MAE_{prev}"]
                r[f"p_{m}"] = dm_daily(g, m, prev)["pvalue"]
            r["gain_total"] = 1 - r["MAE_M3"] / r["MAE_B4b"]
            rows.append(r)
    tab = pd.DataFrame(rows)
    for m in STEPS[1:]:
        tab[f"p_{m}"] = tab.groupby(["year", "bucket"])[f"p_{m}"].transform(lambda s: holm(s.to_numpy()))
    out = REPO_ROOT / "reports" / "models"
    tab.to_csv(out / "ablation_weather.csv", index=False)

    fp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    L = ["# What does weather add? Step-by-step comparison", "",
         "Same rows, settings and seed for every step; issues after the day-ahead forecast is "
         "posted. Each gain is the % reduction in MAE versus the previous step; p = Diebold-Mariano "
         "on daily MAE vs the previous step, Holm-adjusted across BAs.", "",
         "Steps: Trick 2 (B4b) -> " + " -> ".join(f"{LABEL[m]} ({m})" for m in STEPS[1:]), ""]
    for y in sorted(tab["year"].unique()):
        L += [f"## {y}", "",
              "| BA | hours ahead | Trick 2 MAE | demand-only MAE | +calendar MAE | +weather MAE "
              "| same-data gain | p | calendar gain | p | weather gain | p | total gain |",
              "|" + "---|" * 13]
        for _, r in tab[tab["year"] == y].iterrows():
            L.append(f"| {r.ba} | {r.bucket} | {r.MAE_B4b:.0f} | {r.MAE_M3_demandOnly:.0f} | {r.MAE_M3_noWx:.0f} "
                     f"| {r.MAE_M3:.0f} | {r.gain_M3_demandOnly:+.1%} | {fp(r.p_M3_demandOnly)} "
                     f"| {r.gain_M3_noWx:+.1%} | {fp(r.p_M3_noWx)} | {r.gain_M3:+.1%} | {fp(r.p_M3)} "
                     f"| {r.gain_total:+.1%} |")
        L.append("")
    (out / "ablation_weather.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-11-19")
    a = ap.parse_args(argv)
    if not a.report:
        run(a.start, a.end)
    report()


if __name__ == "__main__":
    main()
