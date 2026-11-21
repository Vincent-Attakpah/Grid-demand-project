"""Static dashboard: one self-contained HTML page (inline SVG charts, no scripts)."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd

W, H_, PAD_L, PAD_R, PAD_T, PAD_B = 660, 250, 58, 14, 14, 30

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dc;
--actual:#1d1d1b;--ours:#2563c9;--band90:rgba(37,99,201,.14);--band50:rgba(37,99,201,.28);
--util:#a05a00;--warn-bg:#fde8e4;--warn-ink:#9b1c0c;--ok-bg:#e6f2ea;--ok-ink:#1e6b3a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--card:#1f1f1d;
--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;--actual:#ecebe6;--ours:#7aa7ff;
--band90:rgba(122,167,255,.16);--band50:rgba(122,167,255,.32);--util:#e0a050;
--warn-bg:#3d1c16;--warn-ink:#ffb4a5;--ok-bg:#16301f;--ok-ink:#9fdcb3}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:0}
.sub{color:var(--muted);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:16px;margin:0 0 18px}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.muted{color:var(--muted);font-size:13px}svg{width:100%;height:auto;display:block;margin:8px 0}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--muted)}
.sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}
.peak{border-radius:8px;padding:10px 12px;margin:10px 0;background:var(--ok-bg);color:var(--ok-ink)}
.peak.warn{background:var(--warn-bg);color:var(--warn-ink)}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:right}
td:first-child,th:first-child{text-align:left}th{color:var(--muted);font-weight:600}
.axis text{fill:var(--muted);font-size:11px}.grid{stroke:var(--line)}
"""


def mw(x) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:,.0f} MW"


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def local(t, tz, fmt="%I:%M %p %Z") -> str:
    return pd.Timestamp(t).tz_convert(tz).strftime(fmt).lstrip("0")


def day_hours(op_day: str, tz: str) -> pd.DatetimeIndex:
    s = pd.Timestamp(op_day).tz_localize(tz)
    e = (pd.Timestamp(op_day) + pd.Timedelta(days=1)).tz_localize(tz)
    return pd.date_range(s, e, freq="h", tz=tz)[1:].tz_convert("UTC")


def chart(hours, actual, util, fc: pd.DataFrame | None, tau, tz) -> str:
    n = len(hours)
    a = actual.reindex(hours).to_numpy(float)
    u = util.reindex(hours).to_numpy(float)
    vals = [a, u]
    if fc is not None and not fc.empty:
        f = fc.set_index("target_utc").reindex(hours)
        vals += [f[c].to_numpy(float) for c in ("M3", "M4_q05", "M4_q95")]
    allv = np.concatenate(vals)
    allv = allv[np.isfinite(allv)]
    if not len(allv):
        return '<p class="muted">No data yet for today.</p>'
    lo, hi = allv.min(), allv.max()
    pad = (hi - lo) * 0.08 or 100
    lo, hi = lo - pad, hi + pad
    X = lambda i: PAD_L + (W - PAD_L - PAD_R) * i / max(n - 1, 1)
    Y = lambda v: PAD_T + (H_ - PAD_T - PAD_B) * (1 - (v - lo) / (hi - lo))

    def path(v):
        segs, cur = [], []
        for i, y in enumerate(v):
            if np.isfinite(y):
                cur.append(f"{X(i):.1f},{Y(y):.1f}")
            elif cur:
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        return " ".join("M" + " L".join(s) for s in segs if s)

    def band(l, h):
        ok = np.isfinite(l) & np.isfinite(h)
        idx = np.flatnonzero(ok)
        if len(idx) < 2:
            return ""
        top = [f"{X(i):.1f},{Y(h[i]):.1f}" for i in idx]
        bot = [f"{X(i):.1f},{Y(l[i]):.1f}" for i in idx[::-1]]
        return "M" + " L".join(top + bot) + " Z"

    out = [f'<svg viewBox="0 0 {W} {H_}" role="img" aria-label="Demand today">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        out.append(f'<g class="axis"><text x="{PAD_L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v / 1000:,.1f}k</text></g>')
    for i in range(0, n, 3):
        lab = pd.Timestamp(hours[i]).tz_convert(tz).strftime("%I %p").lstrip("0")
        out.append(f'<g class="axis"><text x="{X(i):.1f}" y="{H_ - 10}" text-anchor="middle">{lab}</text></g>')
    if fc is not None and not fc.empty:
        out.append(f'<path d="{band(vals[3], vals[4])}" fill="var(--band90)"/>')
        out.append(f'<path d="{band(f["M4_q25"].to_numpy(float), f["M4_q75"].to_numpy(float))}" fill="var(--band50)"/>')
        out.append(f'<path d="{path(vals[2])}" fill="none" stroke="var(--ours)" stroke-width="2.2"/>')
    out.append(f'<path d="{path(u)}" fill="none" stroke="var(--util)" stroke-width="1.6" stroke-dasharray="5 4"/>')
    out.append(f'<path d="{path(a)}" fill="none" stroke="var(--actual)" stroke-width="2.2"/>')
    pos = np.searchsorted(hours, pd.Timestamp(tau))
    if 0 < pos < n:
        x = X(pos - 1 + (pd.Timestamp(tau) - hours[pos - 1]) / pd.Timedelta(hours=1))
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_T}" y2="{H_ - PAD_B}" stroke="var(--muted)" stroke-dasharray="2 3"/>')
    out.append("</svg>")
    return "".join(out)


LEGEND = ('<div class="legend"><span><i class="sw" style="background:var(--actual)"></i>Actual demand</span>'
          '<span><i class="sw" style="background:var(--util)"></i>Utility\'s day-ahead forecast</span>'
          '<span><i class="sw" style="background:var(--ours)"></i>Our forecast</span>'
          '<span><i class="sw" style="background:var(--band90);height:10px"></i>90% range</span>'
          '<span><i class="sw" style="background:var(--band50);height:10px"></i>50% range</span></div>')


def peak_box(pk: dict | None) -> str:
    if not pk:
        return ""
    warn = pk["p_M5"] >= 0.5
    head = "High-peak warning" if warn else "No high-peak warning"
    return (f'<div class="peak{" warn" if warn else ""}"><strong>{head}.</strong> '
            f'Highest demand still to come today: about {mw(pk["peak_M5"])} '
            f'(90% range {mw(pk["lo_M5"])} to {mw(pk["hi_M5"])}). '
            f'High-day line for this month: {mw(pk["threshold"])}. '
            f'Chance of crossing it: <strong>{pct(pk["p_M5"])}</strong>.</div>')


def score_table(sc: dict, pks: dict) -> str:
    rows = []
    for wname, label in (("last_30_days", "Last 30 days"), ("since_start", "Since live start")):
        r = sc.get(wname, {}).get("1-6 h ahead", {})
        if not r or not r.get("hours_scored"):
            continue
        rows.append(f"<tr><td>{label}</td><td>{r['hours_scored']:,}</td><td>{mw(r['mae_M3'])}</td>"
                    f"<td>{mw(r['mae_B4b'])}</td><td>{mw(r['mae_da'])}</td><td>{pct(r.get('cover90'))}</td></tr>")
    if not rows:
        return '<p class="muted">Scores appear once actual demand for forecast hours is published (about 3 hours later).</p>'
    t = ("<table><tr><th>1–6 hours ahead</th><th>hours scored</th><th>our average miss</th>"
         "<th>Trick 2</th><th>utility</th><th>90% range caught</th></tr>" + "".join(rows) + "</table>")
    p = pks.get("since_start", {})
    if p.get("issues_scored"):
        t += (f'<p class="muted">Peak warnings since live start ({p["days"]} days, {p["high_days"]} high days): '
              f'{p["hits"]} hits, {p["misses"]} misses, {p["false_alarms"]} false alarms. '
              f'Average peak miss {mw(p["peak_mae_M5"])} (Trick 2 {mw(p["peak_mae_trick2"])}, '
              f'utility {mw(p["peak_mae_utility"])}).</p>')
    return t


def render(cards: list[dict], status: dict) -> str:
    body = [f"<h1>Grid demand nowcaster</h1>"
            f'<p class="sub">Updated every hour. Latest forecast issued '
            f'{pd.Timestamp(status["issue_utc"]).tz_convert("America/New_York").strftime("%b %d, %I:%M %p %Z")}. '
            f'Models last trained {html.escape(status.get("trained_at", "?")[:10])}.</p>']
    for c in cards:
        tz = c["tz"]
        body.append('<section class="card">')
        body.append(f'<div class="row"><h2>{html.escape(c["name"])}</h2>'
                    f'<span class="muted">{html.escape(c["code"])} · times in {local(c["issue_utc"], tz, "%Z")}</span></div>')
        if c.get("error"):
            body.append(f'<p class="muted">This hour\'s forecast failed: {html.escape(c["error"])}</p>')
        fc = c.get("forecast")
        if fc is None or fc.empty or not np.isfinite(fc["M3"]).any():
            body.append('<p class="muted">Waiting for today\'s day-ahead forecast (it counts as available from '
                        '10 AM Eastern); our forecast starts then.</p>')
            fc = None
        else:
            nxt = fc.sort_values("horizon").iloc[0]
            body.append(f'<p>Next hour (ending {local(nxt["target_utc"], tz, "%I %p")}): '
                        f'<strong>{mw(nxt["M3"])}</strong> '
                        f'<span class="muted">(90% range {mw(nxt["M4_q05"])} to {mw(nxt["M4_q95"])}; '
                        f'utility forecast {mw(nxt["da"])})</span></p>')
        body.append(chart(c["hours"], c["actual"], c["util"], fc, c["issue_utc"], tz) + LEGEND)
        body.append(peak_box(c.get("peak")))
        body.append(score_table(c.get("scores", {}), c.get("peak_scores", {})))
        body.append("</section>")
    body.append('<p class="muted">Our forecast = the utility\'s day-ahead forecast corrected with the latest '
                'published demand, the newest weather forecast and the calendar (model M3). Trick 2 = the '
                'day-ahead forecast plus the latest error and the same-hour-yesterday error. A high day = a '
                'daily peak above the top-10% line of the same month in the two previous years. '
                'Research prototype; not for operational use.</p>')
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Grid Demand Nowcaster</title><style>" + CSS + "</style></head><body><main>"
            + "".join(body) + "</main></body></html>")
