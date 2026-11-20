"""Tables and significance tests for the backtest (reads <data>/backtest/pred_*.parquet).

    python scripts/report_models.py

2025 is the development year; 2026 (Jan to date) is the test year named in docs/decisions.md.
All models are scored on identical rows: issued after the DA posting time, with the actual
and every compared prediction present. DM tests compare each model with B4b on daily mean
absolute-error differences (Newey-West, HLN); Holm adjusts across BAs within each
(year, horizon bucket, model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nowcaster.config import REPO_ROOT, data_dir
from nowcaster.eval.significance import dm_daily, holm

MODELS = ["B0", "B1", "B2", "B3", "B4", "B4b", "M2", "M3"]
SHOW = ["B2", "B3", "B4", "B4b", "M2", "M3"]
CHALLENGERS = ["M2", "M3"]
BUCKETS = [("1-3", 1, 3), ("4-6", 4, 6), ("1-6", 1, 6), ("7-12", 7, 12), ("13-24", 13, 25)]


def load() -> pd.DataFrame:
    files = sorted((data_dir() / "backtest").glob("pred_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["da_posted"]].dropna(subset=["actual", *MODELS])
    df["year"] = pd.DatetimeIndex(df["issue_utc"]).year
    return df


def table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, ba), g0 in df.groupby(["year", "ba"]):
        for name, lo, hi in BUCKETS:
            g = g0[(g0["horizon"] >= lo) & (g0["horizon"] <= hi)]
            if g.empty:
                continue
            r = {"year": year, "ba": ba, "bucket": name, "days": g["op_day"].nunique(), "n": len(g)}
            for m in MODELS:
                ae = (g[m] - g["actual"]).abs()
                r[f"MAE_{m}"] = ae.mean()
                r[f"MAPE_{m}"] = (ae / g["actual"]).mean() * 100
            for m in CHALLENGERS:
                r[f"skill_{m}"] = 1 - r[f"MAE_{m}"] / r["MAE_B4b"]
                dm = dm_daily(g, m, "B4b")
                r[f"dm_{m}"], r[f"p_{m}"] = dm["stat"], dm["pvalue"]
            rows.append(r)
    tab = pd.DataFrame(rows)
    for m in CHALLENGERS:
        tab[f"pholm_{m}"] = tab.groupby(["year", "bucket"])[f"p_{m}"].transform(lambda s: holm(s.to_numpy()))
    return tab


def verdict(tab: pd.DataFrame, year: int, model: str = "M3", bucket: str = "1-6") -> tuple[int, int]:
    t = tab[(tab["year"] == year) & (tab["bucket"] == bucket)]
    wins = ((t[f"skill_{model}"] > 0) & (t[f"pholm_{model}"] < 0.05)).sum()
    return int(wins), len(t)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    df = load()
    tab = table(df)
    out = REPO_ROOT / "reports" / "models"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "backtest.csv", index=False)
    span = df.groupby("year")["issue_utc"].agg(["min", "max"])
    L = ["# Model backtest: weather models vs the corrected utility forecast", "",
         "Rolling origin, refit monthly on labels published before each month; training from "
         "2024-01-21. 2025 = development year, 2026 = test year. Identical rows for every model "
         "(issued after the day-ahead forecast is posted). Skill = 1 - MAE(model)/MAE(B4b). "
         "p = Diebold-Mariano vs B4b on daily mean absolute error, Holm-adjusted across BAs.", ""]
    for y, (a, b) in span.iterrows():
        L.append(f"- {y}: issues {a:%Y-%m-%d} to {b:%Y-%m-%d}")
    L.append("")
    for y in sorted(tab["year"].unique()):
        w, n = verdict(tab, y)
        L.append(f"**{y}, M3 vs B4b at 1-6 h:** better and significant in {w} of {n} BAs.  ")
    L.append("")
    for y in sorted(tab["year"].unique()):
        t = tab[tab["year"] == y]
        L += [f"## {y}", "", "| BA | hours ahead | days | " + " | ".join(f"{m} MAE" for m in SHOW)
              + " | B4b MAPE | M3 MAPE | M2 skill | M2 p | M3 skill | M3 p |",
              "|" + "---|" * (3 + len(SHOW) + 6)]
        for _, r in t.iterrows():
            L.append(f"| {r.ba} | {r.bucket} | {r.days} | " + " | ".join(f"{r[f'MAE_{m}']:.0f}" for m in SHOW)
                     + f" | {r.MAPE_B4b:.2f} | {r.MAPE_M3:.2f} | {r.skill_M2:+.1%} | {fmt_p(r.pholm_M2)}"
                     + f" | {r.skill_M3:+.1%} | {fmt_p(r.pholm_M3)} |")
        L.append("")
    (out / "backtest.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
