"""Tables for forecast ranges (reads <data>/backtest/int_*.parquet).

    python scripts/report_intervals.py

Coverage: share of hours where actual demand fell inside the range (target 90% / 50%).
Width: average size of the 90% range (MW). Pinball: average quantile loss over the five
levels (lower is better); skill = 1 - pinball(M4)/pinball(Q0); p = Diebold-Mariano on the
daily mean pinball difference, Holm-adjusted across BAs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nowcaster.config import REPO_ROOT, data_dir
from nowcaster.eval.significance import auto_lag, dm_test, holm
from nowcaster.models.intervals import LEVELS

BUCKETS = [("1-3", 1, 3), ("4-6", 4, 6), ("1-6", 1, 6), ("7-12", 7, 12), ("13-24", 13, 25)]
METHODS = ["M4", "Q0"]


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def load() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in sorted((data_dir() / "backtest").glob("int_*.parquet"))],
                   ignore_index=True)
    cols = [f"{m}_{k}" for m in METHODS for k in LEVELS]
    df = df.dropna(subset=["actual", *cols])
    df["year"] = pd.DatetimeIndex(df["issue_utc"]).year
    y = df["actual"].to_numpy(float)
    for m in METHODS:
        df[f"pb_{m}"] = np.mean([pinball(y, df[f"{m}_{k}"].to_numpy(float), t) for k, t in LEVELS.items()], axis=0)
        df[f"in90_{m}"] = (df["actual"] >= df[f"{m}_q05"]) & (df["actual"] <= df[f"{m}_q95"])
        df[f"in50_{m}"] = (df["actual"] >= df[f"{m}_q25"]) & (df["actual"] <= df[f"{m}_q75"])
        df[f"w90_{m}"] = df[f"{m}_q95"] - df[f"{m}_q05"]
    return df


def table(df):
    rows = []
    for (year, ba), g0 in df.groupby(["year", "ba"]):
        for name, lo, hi in BUCKETS:
            g = g0[(g0["horizon"] >= lo) & (g0["horizon"] <= hi)]
            r = {"year": year, "ba": ba, "bucket": name, "days": g["op_day"].nunique()}
            for m in METHODS:
                r[f"cov90_{m}"] = g[f"in90_{m}"].mean()
                r[f"cov50_{m}"] = g[f"in50_{m}"].mean()
                r[f"width90_{m}"] = g[f"w90_{m}"].mean()
                r[f"pinball_{m}"] = g[f"pb_{m}"].mean()
            r["skill"] = 1 - r["pinball_M4"] / r["pinball_Q0"]
            daily = (g["pb_M4"] - g["pb_Q0"]).groupby(g["op_day"].to_numpy()).mean().sort_index()
            r["p"] = dm_test(daily.to_numpy(), np.zeros(len(daily)), lag=auto_lag(len(daily)))["pvalue"]
            # worst month: how far coverage can drift
            mon = g.groupby(pd.DatetimeIndex(g["issue_utc"]).month)["in90_M4"].mean()
            r["cov90_M4_min_month"] = mon.min()
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab["pholm"] = tab.groupby(["year", "bucket"])["p"].transform(lambda s: holm(s.to_numpy()))
    return tab


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    tab = table(load())
    out = REPO_ROOT / "reports" / "models"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "intervals.csv", index=False)
    L = ["# Forecast ranges: M4 (quantile model + calibration) vs Q0 (fixed error spread around B4b)", "",
         "Rolling origin, refit monthly, training from 2024-01-21; issues after the DA is posted. "
         "Coverage targets: 90% and 50%. Pinball = mean quantile loss over 5/25/50/75/95% "
         "(lower is better). Skill = 1 - pinball(M4)/pinball(Q0); p = DM on daily pinball "
         "differences, Holm across BAs.", ""]
    for y in sorted(tab["year"].unique()):
        t = tab[tab["year"] == y]
        L += [f"## {y}", "",
              "| BA | hours ahead | 90% cover M4 | 90% cover Q0 | worst month M4 | 50% cover M4 | 50% cover Q0 "
              "| 90% width M4 | 90% width Q0 | pinball M4 | pinball Q0 | M4 skill | p |",
              "|" + "---|" * 13]
        for _, r in t.iterrows():
            L.append(f"| {r.ba} | {r.bucket} | {r.cov90_M4:.1%} | {r.cov90_Q0:.1%} | {r.cov90_M4_min_month:.1%} "
                     f"| {r.cov50_M4:.1%} | {r.cov50_Q0:.1%} | {r.width90_M4:.0f} | {r.width90_Q0:.0f} "
                     f"| {r.pinball_M4:.1f} | {r.pinball_Q0:.1f} | {r.skill:+.1%} | {fmt_p(r.pholm)} |")
        L.append("")
    (out / "intervals.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
