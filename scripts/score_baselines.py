"""Score baselines B0, B1, B3, B4, B4b on 2025 with monthly rolling origins (plan v2 week 1).

    python scripts/score_baselines.py [--start 2025-01-01 --end 2026-01-01]

B2 needs the weather backfill; it is skipped (NaN) until weather_ba_hourly exists.
Writes reports/baselines/baselines_2025.md and .csv. All models are scored on identical
rows: issued after the DA posting time, with every compared prediction non-missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nowcaster.baselines.simple import BASELINES
from nowcaster.config import REPO_ROOT, load_bas
from nowcaster.eval.rolling import rolling_origin
from nowcaster.features.panel import Lags, build_panel_for_ba
from nowcaster.storage import query

BUCKETS = [(1, 3), (4, 6), (7, 12), (13, 25)]
MODELS = ["B0", "B1", "B3", "B4", "B4b"]


def bucket(h):
    for lo, hi in BUCKETS:
        if lo <= h <= hi:
            return f"{lo}-{min(hi, 24)}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--train-start", default="2023-01-01")
    a = ap.parse_args(argv)

    bas = load_bas()
    eia = query("select ba, period_utc, demand_adjusted, da_forecast, demand_imputed "
                "from read_parquet('{clean}/eia_hourly/**/*.parquet', hive_partitioning=true)")
    eia["period_utc"] = pd.to_datetime(eia["period_utc"], utc=True)
    lags = Lags.from_config()
    scored = []
    for code, ba in bas.items():
        panel = build_panel_for_ba(eia, code, ba.tz, a.train_start, a.end, lags)
        res = rolling_origin(panel, {m: BASELINES[m] for m in MODELS}, a.start, a.end)
        scored.append(res)
        print(code, len(res), "rows")
    df = pd.concat(scored, ignore_index=True)
    df = df[df["da_posted"]].dropna(subset=["actual", *MODELS])
    df["bucket"] = df["horizon"].map(bucket)

    rows = []
    for (ba, b), g in df.groupby(["ba", "bucket"]):
        r = {"ba": ba, "bucket": b, "n": len(g), "days": g["op_day"].nunique()}
        for m in MODELS:
            ae = (g[m] - g["actual"]).abs()
            r[f"MAE_{m}"] = ae.mean()
            r[f"MAPE_{m}"] = (ae / g["actual"]).mean() * 100
        for m in MODELS:
            r[f"skill_vs_B3_{m}"] = 1 - r[f"MAE_{m}"] / r["MAE_B3"]
            r[f"skill_vs_B4_{m}"] = 1 - r[f"MAE_{m}"] / r["MAE_B4"]
        rows.append(r)
    tab = pd.DataFrame(rows)
    order = {f"{lo}-{min(hi, 24)}": i for i, (lo, hi) in enumerate(BUCKETS)}
    tab = tab.sort_values(["ba", "bucket"], key=lambda s: s.map(order) if s.name == "bucket" else s)

    out = REPO_ROOT / "reports" / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "baselines_2025.csv", index=False)
    lines = [f"# Baselines, rolling origin {a.start} to {a.end}", "",
             f"Lags: demand +{lags.demand_lag_h} h, DA from {lags.da_time} {lags.da_tz} (placeholders). "
             "Issues at :17 after DA posting only; identical rows for every model. B4/B4b refit monthly "
             f"on labels published before each origin (training from {a.train_start}).", "",
             "## MAE (MW) and MAPE (%)", "",
             "| BA | horizon | days | " + " | ".join(f"{m} MAE" for m in MODELS) + " | B3 MAPE | B4 MAPE | B4 skill vs B3 | B4b skill vs B4 |",
             "|" + "---|" * (3 + len(MODELS) + 4)]
    for _, r in tab.iterrows():
        lines.append(f"| {r.ba} | {r.bucket} | {r.days} | " + " | ".join(f"{r[f'MAE_{m}']:.0f}" for m in MODELS)
                     + f" | {r.MAPE_B3:.2f} | {r.MAPE_B4:.2f} | {r.skill_vs_B3_B4:+.1%} | {r.skill_vs_B4_B4b:+.1%} |")
    (out / "baselines_2025.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
