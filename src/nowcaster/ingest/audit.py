"""EIA-930 data-quality audit (plan v2 §7.4) and keep/drop rule (§4.1).

Usage: python -m nowcaster.ingest.audit [--skip-build]
Writes reports/audit/audit.md, reports/audit/metrics.csv and PNG plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, load_bas, load_config
from ..storage import query

REPORT_DIR = REPO_ROOT / "reports" / "audit"
AUDIT_BAS = ("SOCO", "TVA", "DUK", "FPL", "ERCO")

RATIO_LO, RATIO_HI = 0.5, 2.0
SPIKE_Z = 10.0
MAX_IMPLAUSIBLE = 0.02
MAX_ABS_BIAS = 0.10
BREAK_THRESHOLD = 0.05
MAD_K = 1.4826


# ---------------------------------------------------------------- metrics
def robust_scale(x: pd.Series) -> float:
    x = x.dropna()
    s = MAD_K * np.median(np.abs(x - np.median(x))) if len(x) else np.nan
    return float(s) if s and s > 0 else np.nan


def level_spikes(demand: pd.Series, window: int = 25, z: float = SPIKE_Z) -> pd.Series:
    """Hours far from a centred rolling median, in robust-z units of the residual."""
    med = demand.rolling(window, center=True, min_periods=window // 2).median()
    resid = demand - med
    scale = robust_scale(resid)
    return (resid.abs() / scale > z).fillna(False)


def diff_spikes(demand: pd.Series, z: float = SPIKE_Z) -> pd.Series:
    """Isolated jumps: a large first difference immediately reversed by the next one."""
    d = demand.diff()
    scale = robust_scale(d)
    big = (d - d.median()).abs() / scale > z
    nxt = d.shift(-1)
    reversed_ = np.sign(nxt) == -np.sign(d)
    similar = (nxt.abs() > 0.5 * d.abs())
    return (big & reversed_ & similar).fillna(False)


def daily_bias(df: pd.DataFrame, tz: str, demand_col: str = "demand") -> pd.Series:
    """Per operating day: (sum DA - sum demand) / sum demand over hours with both."""
    ok = df[["da_forecast", demand_col]].notna().all(axis=1) & (df[demand_col] > 0)
    d = df.loc[ok]
    day = (d["period_utc"].dt.tz_convert(tz) - pd.Timedelta(hours=1)).dt.date
    g = d.groupby(day)
    return (g["da_forecast"].sum() - g[demand_col].sum()) / g[demand_col].sum()


def lag_correlations(da: pd.Series, demand: pd.Series, lags=range(-2, 3)) -> dict[int, float]:
    """corr(diff DA(t), diff demand(t+k)). Best k != 0 means DA is stamped k hours early
    (k>0) or late (k<0) relative to demand. Inputs indexed by a complete hourly index."""
    a = da.diff()
    b = demand.diff()
    return {k: float(a.corr(b.shift(-k))) for k in lags}


LAG_MARGIN = 0.01


def best_lag(corrs: dict[int, float], margin: float = LAG_MARGIN) -> int:
    """Lag with the highest correlation; 0 unless another lag beats lag 0 by `margin`
    (smooth DA series make lag ±1 nearly tie with 0, which is not a stamping shift)."""
    valid = {k: v for k, v in corrs.items() if np.isfinite(v)}
    if not valid:
        return 0
    k = max(valid, key=valid.get)
    if k != 0 and 0 in valid and valid[k] - valid[0] < margin:
        return 0
    return k


def best_single_break(monthly: pd.Series, window: int = 12) -> tuple[pd.Timestamp | None, float]:
    """Largest local step: for each month m, mean of the `window` months from m divided by
    mean of the `window` months before m, minus 1. Using full 12-month windows cancels the
    seasonal cycle; a local window keeps a steady growth trend (a few %/yr) from posing as a
    break. Returns (first month after the step, step size)."""
    m = monthly.asfreq("MS")
    before = m.rolling(window, min_periods=window - 1).mean().shift(1)
    after = m[::-1].rolling(window, min_periods=window - 1).mean()[::-1]
    step = (after / before - 1).dropna()
    if step.empty:
        return None, 0.0
    i = step.abs().idxmax()
    return i, float(step[i])


def yoy_shift(monthly: pd.Series, at: str, months: int = 6) -> float:
    """Mean YoY ratio of the `months` after `at` minus the `months` before (both deseasonalised
    by comparing with the same months a year earlier)."""
    m = monthly.dropna()
    yoy = m / m.shift(12) - 1
    t = pd.Timestamp(at)
    before = yoy[(yoy.index < t) & (yoy.index >= t - pd.DateOffset(months=months))]
    after = yoy[(yoy.index >= t) & (yoy.index < t + pd.DateOffset(months=months))]
    return float(after.mean() - before.mean())


# ---------------------------------------------------------------- audit
def load_eia(bas=AUDIT_BAS) -> pd.DataFrame:
    df = query("select * from read_parquet('{clean}/eia_hourly/**/*.parquet', hive_partitioning=true)")
    df = df[df["ba"].isin(bas)].copy()
    df["period_utc"] = pd.to_datetime(df["period_utc"], utc=True)
    return df.sort_values(["ba", "period_utc"]).reset_index(drop=True)


def flag_hours(g: pd.DataFrame, col: str = "demand", prefix: str = "f_") -> pd.DataFrame:
    """Implausibility flags for demand column `col`, written as `<prefix>*` columns."""
    g = g.copy()
    x = g[col]
    g[prefix + "missing"] = x.isna()
    g[prefix + "nonpos"] = x <= 0
    ratio = g["da_forecast"] / x.where(x > 0)
    g[prefix + "ratio"] = (ratio < RATIO_LO) | (ratio > RATIO_HI)
    dem = x.where(x > 0)
    g[prefix + "spike_level"] = level_spikes(dem).values
    g[prefix + "spike_diff"] = diff_spikes(dem).values
    g[prefix + "spike"] = g[prefix + "spike_level"] | g[prefix + "spike_diff"]
    g[prefix + "implausible"] = g[[prefix + s for s in ("missing", "nonpos", "ratio", "spike")]].any(axis=1)
    return g


def flag_all(g: pd.DataFrame) -> pd.DataFrame:
    """Flags on reported demand (what the live system sees) and on EIA-adjusted demand
    (the training target). `demand_clean` = adjusted demand with implausible hours removed;
    it is the series used for bias, hour-shift and break checks."""
    g = flag_hours(g, "demand", "f_")
    g = flag_hours(g, "demand_adjusted", "fa_")
    g["f_da_missing"] = g["da_forecast"].isna()
    g["adj_differs"] = (g["demand_adjusted"] - g["demand"]).abs() > 0.5
    g["demand_clean"] = g["demand_adjusted"].where(~g["fa_implausible"]).astype("float64")
    g["da_clean"] = g["da_forecast"].where(~g["fa_ratio"]).astype("float64")
    return g


def audit_ba(g: pd.DataFrame, tz: str) -> tuple[pd.DataFrame, dict]:
    g = g.set_index("period_utc").asfreq("h").reset_index()
    g["ba"] = g["ba"].ffill().bfill()
    # audit window ends at the last hour with reported demand
    last = g.loc[g["demand"].notna(), "period_utc"].max()
    g = flag_all(g[g["period_utc"] <= last])
    g["year"] = g["period_utc"].dt.tz_convert(tz).dt.year
    gb = g.assign(da_forecast=g["da_clean"], demand=g["demand_clean"])
    bias = daily_bias(gb, tz)  # on cleaned adjusted demand
    gr = g.assign(demand=g["demand"].where(~g["f_implausible"]))
    bias_rep = daily_bias(gr, tz)  # on reported demand, implausible hours removed
    bias_year = pd.Series(pd.to_datetime(bias.index).year, index=bias.index)
    bias_rep_year = pd.Series(pd.to_datetime(bias_rep.index).year, index=bias_rep.index)

    rows = []
    for yr, gy in g.groupby("year"):
        s = gy.set_index("period_utc")
        corrs = lag_correlations(s["da_clean"], s["demand_clean"])
        ape = (s["da_clean"] - s["demand_clean"]).abs() / s["demand_clean"]
        rows.append({
            "year": yr, "hours": len(gy),
            "missing": gy["f_missing"].mean(), "nonpos": gy["f_nonpos"].mean(),
            "ratio_out": gy["f_ratio"].mean(), "da_missing": gy["f_da_missing"].mean(),
            "imputed": gy["demand_imputed"].fillna(False).astype(bool).mean(),
            "adj_differs": gy["adj_differs"].mean(),
            "spikes_level": int(gy["f_spike_level"].sum()), "spikes_diff": int(gy["f_spike_diff"].sum()),
            "implausible": gy["f_implausible"].mean(),
            "implausible_adj": gy["fa_implausible"].mean(),
            "median_daily_bias": float(bias[bias_year == yr].median()),
            "median_daily_bias_rep": float(bias_rep[bias_rep_year == yr].median()),
            "mape_hourly": float(ape.median()),
            "mean_ape": float(ape.mean()),
            "best_lag": best_lag(corrs), **{f"corr_lag{k:+d}": v for k, v in corrs.items()},
        })
    by_year = pd.DataFrame(rows)

    # monthly best lag (to locate shifted periods inside a year)
    month = g["period_utc"].dt.tz_convert(tz).dt.tz_localize(None).dt.to_period("M")
    mlag, mcorr = {}, {}
    for mo, gm in g.groupby(month):
        s = gm.set_index("period_utc")
        if s["da_clean"].notna().sum() > 24 * 10:
            c = lag_correlations(s["da_clean"], s["demand_clean"])
            mlag[mo.to_timestamp()] = best_lag(c)
            mcorr[mo.to_timestamp()] = c
    monthly_mean = g.groupby(month)["demand_clean"].mean()
    monthly_mean.index = monthly_mean.index.to_timestamp()
    brk_at, brk_shift = best_single_break(monthly_mean)

    imp = g["demand_imputed"].fillna(False).astype(bool)
    spikes = g.loc[g["f_spike"] | g["f_nonpos"],
                   ["period_utc", "demand", "demand_adjusted", "da_forecast", "fa_implausible"]]
    info = {
        "last_demand_hour": last, "bias_daily": bias, "monthly_mean": monthly_mean,
        "monthly_lag": pd.Series(mlag, dtype=float), "monthly_corr": mcorr,
        "break_at": brk_at, "break_shift": brk_shift,
        "yoy_shift_2022_07": yoy_shift(monthly_mean, "2022-07-01"),
        "spikes": spikes, "overall_implausible": g["f_implausible"].mean(),
        "overall_implausible_adj": g["fa_implausible"].mean(),
        "overall_median_bias": float(bias.median()),
        "overall_median_bias_rep": float(bias_rep.median()),
        "missing_runs": _runs(g.loc[g["f_missing"], "period_utc"]),
        "imputed_hours": int(imp.sum()),
        "imputed_eq_da": float((g.loc[imp, "demand_adjusted"] == g.loc[imp, "da_forecast"])
                               .fillna(False).astype(bool).mean())
        if imp.any() else float("nan"),
    }
    return by_year, info


def _runs(ts: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Contiguous runs of hourly timestamps: (start, end, hours)."""
    ts = ts.sort_values().reset_index(drop=True)
    if ts.empty:
        return []
    grp = (ts.diff() != pd.Timedelta(hours=1)).cumsum()
    return [(s.iloc[0], s.iloc[-1], len(s)) for _, s in ts.groupby(grp)]


def verdict(by_year: pd.DataFrame, info: dict) -> tuple[str, list[str]]:
    """Rule applied to EIA-adjusted demand (the training target). Reported-demand quality
    is shown separately because it drives live lag features, not the training data."""
    reasons = []
    bad_years = by_year[(by_year["implausible_adj"] > MAX_IMPLAUSIBLE) |
                        (by_year["median_daily_bias"].abs() > MAX_ABS_BIAS) |
                        (by_year["best_lag"] != 0)]
    if info["overall_implausible_adj"] > MAX_IMPLAUSIBLE:
        reasons.append(f"implausible hours {info['overall_implausible_adj']:.2%} > 2%")
    if abs(info["overall_median_bias"]) > MAX_ABS_BIAS:
        reasons.append(f"median daily bias {info['overall_median_bias']:+.2%} beyond 10%")
    for _, r in bad_years.iterrows():
        why = []
        if r["implausible_adj"] > MAX_IMPLAUSIBLE:
            why.append(f"implausible {r['implausible_adj']:.2%}")
        if abs(r["median_daily_bias"]) > MAX_ABS_BIAS:
            why.append(f"bias {r['median_daily_bias']:+.2%}")
        if r["best_lag"] != 0:
            why.append(f"DA best lag {int(r['best_lag']):+d}h")
        reasons.append(f"{int(r['year'])}: " + ", ".join(why))
    if info["overall_implausible_adj"] > MAX_IMPLAUSIBLE or abs(info["overall_median_bias"]) > MAX_ABS_BIAS:
        return "DROP", reasons
    if len(bad_years):
        # does any bad year fall in the modelling window (2024+)?
        if (bad_years["year"] >= 2024).any():
            return "DROP or SHORTEN (problem inside 2024+ modelling window)", reasons
        return "KEEP (shorten long-history variant to exclude flagged years)", reasons
    return "KEEP", reasons


# ---------------------------------------------------------------- report
def _plots(results: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.2 * len(results)), sharex=True)
    for ax, (ba, (_, info)) in zip(np.atleast_1d(axes), results.items()):
        mm = info["monthly_mean"]
        ax.plot(mm.index, mm.values / 1e3, lw=1.2)
        if info["break_at"] is not None and abs(info["break_shift"]) > BREAK_THRESHOLD:
            ax.axvline(info["break_at"], color="crimson", ls="--", lw=1)
        ax.axvline(pd.Timestamp("2022-07-01"), color="grey", ls=":", lw=0.8)
        ax.set_ylabel(f"{ba}\nGW")
    np.atleast_1d(axes)[0].set_title("Monthly mean demand (dotted: 2022-07; dashed: best break if >5%)")
    fig.tight_layout()
    fig.savefig(out / "monthly_mean_demand.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.0 * len(results)), sharex=True)
    for ax, (ba, (_, info)) in zip(np.atleast_1d(axes), results.items()):
        b = info["bias_daily"]
        idx = pd.to_datetime(b.index)
        ax.plot(idx, b.values * 100, lw=0.4, alpha=0.6)
        ax.plot(idx, pd.Series(b.values * 100, index=idx).rolling(30, min_periods=10).median(), lw=1.2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-30, 30)
        ax.set_ylabel(f"{ba}\n%")
    np.atleast_1d(axes)[0].set_title("Daily DA bias (DA - demand)/demand, with 30-day rolling median")
    fig.tight_layout()
    fig.savefig(out / "daily_bias.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(len(results), 1, figsize=(10, 1.6 * len(results)), sharex=True)
    for ax, (ba, (_, info)) in zip(np.atleast_1d(axes), results.items()):
        ml = info["monthly_lag"]
        ax.step(ml.index, ml.values, where="post")
        ax.set_ylim(-2.5, 2.5)
        ax.set_ylabel(ba)
    np.atleast_1d(axes)[0].set_title("Monthly best lag of DA vs demand (hours; 0 = aligned)")
    fig.tight_layout()
    fig.savefig(out / "monthly_best_lag.png", dpi=110)
    plt.close(fig)


def write_report(results: dict, out: Path = REPORT_DIR) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    _plots(results, out)
    all_years = pd.concat([by.assign(ba=ba) for ba, (by, _) in results.items()], ignore_index=True)
    all_years.to_csv(out / "metrics.csv", index=False)

    L = ["# EIA-930 data-quality audit", "",
         f"Generated by `python -m nowcaster.ingest.audit`. Source: EIA-930 six-month BALANCE files "
         f"(final, revised values). Window: 2021-01 to the last hour with reported demand. "
         f"Years are BA-local calendar years.", "",
         "Rule (plan v2 §4.1): drop a BA if >2% of hours are implausible or |median daily DA bias| >10%; "
         "drop or shorten the window if the DA forecast is shifted by an hour against demand. "
         "The rule is applied to **EIA-adjusted demand** (the training target). "
         "Implausible = demand missing, ≤0, DA/demand outside [0.5, 2], or an isolated spike "
         f"(robust z > {SPIKE_Z:g} on level vs a 25-h rolling median, or on a reversed first difference). "
         "Bias, hour-shift and break checks use adjusted demand with implausible hours removed. "
         f"A DA lag ≠ 0 is reported only if it beats lag 0 by > {LAG_MARGIN} in correlation of first differences.", "",
         "## Verdicts", "",
         "| BA | Verdict | Implausible (adjusted / reported) | Median daily bias (adjusted / reported) | Best break (deseasonalised) | 2022-07 YoY step | Reasons |",
         "|---|---|---|---|---|---|---|"]
    for ba, (by, info) in results.items():
        v, reasons = verdict(by, info)
        brk = (f"{info['break_at']:%Y-%m} {info['break_shift']:+.1%}" if info["break_at"] is not None else "n/a")
        L.append(f"| {ba} | {v} | {info['overall_implausible_adj']:.2%} / {info['overall_implausible']:.2%} | "
                 f"{info['overall_median_bias']:+.2%} / {info['overall_median_bias_rep']:+.2%} | {brk} | "
                 f"{info['yoy_shift_2022_07']:+.1%} | {'; '.join(reasons) or '—'} |")
    L += ["", "![monthly mean](monthly_mean_demand.png)", "", "![daily bias](daily_bias.png)", "",
          "![monthly best lag](monthly_best_lag.png)", ""]

    fmt = {"missing": "{:.2%}", "nonpos": "{:.2%}", "ratio_out": "{:.2%}", "da_missing": "{:.2%}",
           "imputed": "{:.2%}", "adj_differs": "{:.2%}", "implausible": "{:.2%}", "implausible_adj": "{:.2%}",
           "median_daily_bias": "{:+.2%}", "median_daily_bias_rep": "{:+.2%}", "mape_hourly": "{:.2%}",
           "mean_ape": "{:.2%}"}
    cols = ["year", "hours", "missing", "nonpos", "ratio_out", "da_missing", "imputed", "adj_differs",
            "spikes_level", "spikes_diff", "implausible", "implausible_adj", "median_daily_bias",
            "median_daily_bias_rep", "mape_hourly", "mean_ape", "best_lag", "corr_lag-1", "corr_lag+0", "corr_lag+1"]
    L += ["## Per-BA detail", "",
          "Columns `missing` … `implausible` are on **reported** demand; `implausible_adj` on adjusted demand. "
          "`median_daily_bias` is on cleaned adjusted demand, `_rep` on reported demand with implausible hours removed. "
          "`mape_hourly` = median hourly APE, `mean_ape` = mean hourly APE (both on cleaned adjusted demand; "
          "not the scorecard's definition).", ""]
    for ba, (by, info) in results.items():
        L += [f"## {ba}", "", f"Last hour with reported demand: {info['last_demand_hour']:%Y-%m-%d %H:%M} UTC.", "",
              "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in by.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                if c in fmt:
                    cells.append(fmt[c].format(v))
                elif c.startswith("corr"):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(int(v)))
            L.append("| " + " | ".join(cells) + " |")
        ml = info["monthly_lag"]
        off = ml[ml != 0]
        L += ["", f"Months where DA best lag ≠ 0: "
              + (", ".join(f"{i:%Y-%m} ({int(v):+d}h)" for i, v in off.items()) if len(off) else "none") + "."]
        if info["imputed_hours"]:
            L += ["", f"EIA-imputed hours: {info['imputed_hours']}; adjusted demand equals the DA forecast "
                  f"in {info['imputed_eq_da']:.0%} of them."]
        runs = [r for r in info["missing_runs"] if r[2] >= 6]
        if runs:
            L += ["", "Gaps in reported demand of 6+ hours: "
                  + ", ".join(f"{s:%Y-%m-%d %H:%M}–{e:%Y-%m-%d %H:%M} UTC ({n} h)" for s, e, n in runs) + "."]
        sp = info["spikes"]
        if len(sp):
            fixed = int((~sp["fa_implausible"]).sum())
            L += ["", f"Spikes / non-positive values in reported demand ({len(sp)}; {fixed} corrected in the "
                  f"adjusted series). First 15:", ""]
            L += [f"- {r.period_utc:%Y-%m-%d %H:%M} UTC: reported {r.demand:,.0f} MW, adjusted "
                  f"{r.demand_adjusted:,.0f} MW, DA {r.da_forecast:,.0f} MW"
                  for r in sp.head(15).itertuples()]
        L.append("")
    L += ["## Implications for modelling", "",
          "- **Imputed hours carry DA as demand.** Where EIA imputes demand, the adjusted value is the DA "
          "forecast, so the residual-on-DA target is exactly zero there. Exclude `demand_imputed` hours from "
          "training targets and scoring (M3 would otherwise learn spurious zero corrections).",
          "- **Adjusted demand still has a few bad hours** (see spike lists where adjusted = reported). "
          "Use `demand_clean` logic (adjusted with implausible hours removed) for targets.",
          "- **Reported demand has extreme spikes** (TVA ±9.9 GW-scale typos, FPL 60–70 GW). Live lag "
          "features need a plausibility filter before use.",
          "- **SOCO and FPL step in mid-2022** (Gulf Power load moving from SOCO to FPL): the long-history "
          "(2022+) temperature-only variant should start after 2022-08 for these two, or include a level dummy. "
          "The 2024+ modelling window is unaffected.",
          "- **DUK/FPL whole-day gaps** on several days, including DUK on both DST fall-back days in the "
          "window. Rows exist but demand is blank (adjusted is imputed).",
          ""]
    L += ["Notes:", "",
          "- `adj_differs` is the share of hours where EIA-adjusted demand differs from reported demand.",
          "- These are final values. First-reported (live) values differ; see plan v2 §7.2. Spikes that "
          "survive in *reported* demand mean the live pipeline needs its own implausibility filter on "
          "lag features (the API may serve reported, not adjusted, values).",
          "- The 2022-07 YoY step compares the 6 months after vs before July 2022, each against the same "
          "months a year earlier; weather moves it by several percent, so read it with the plot."]
    p = out / "audit.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def run(bas=AUDIT_BAS) -> dict:
    df = load_eia(bas)
    tzs = {**{b.code: b.tz for b in load_bas(include_logging_only=True).values()}}
    results = {}
    for ba in bas:
        results[ba] = audit_ba(df[df["ba"] == ba], tzs[ba])
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="rebuild eia_hourly from bulk files first")
    a = ap.parse_args(argv)
    if a.build:
        from .clean import build
        build()
    results = run()
    p = write_report(results)
    for ba, (by, info) in results.items():
        print(ba, verdict(by, info))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
