"""Clinician-facing dashboard: a single self-contained HTML file (inline SVG, no scripts, no network).
Shows per-visit push-off/braking impulse with uncertainty, change from baseline against the measurement-error limit,
and every failed quality check. Visits that fail the gate are shown as 'no usable data', never as a number."""
from __future__ import annotations
import html
import numpy as np
from . import longitudinal as lg

PRELIM = "Research prototype. Simulated or bench data only. Not a medical device; not for clinical decisions."


def build_visit(label: str, result: dict, side: str = "") -> dict:
    v = {"label": label, "usable": result["usable"], "flags": list(result["flags"]),
         "warnings": list(result.get("warnings", [])), "side": side or result.get("side", "")}
    if result["usable"]:
        v["push"] = lg.summarize([s["push_imp_bw"] for s in result["steps"]])
        v["brake"] = lg.summarize([s["brake_imp_bw"] for s in result["steps"]])
    return v


def _svg_trend(visits, key, remount_sd, title, unit="BW·s") -> str:
    W, H, L, R, T, B = 560, 220, 56, 16, 28, 44
    pts = [(i, v[key]) for i, v in enumerate(visits) if v["usable"]]
    if not pts:
        return f'<p class="muted">{html.escape(title)}: no usable visits.</p>'
    lo_hi = []
    for _, s in pts:
        half = 1.96 * float(np.sqrt(s.se ** 2 + remount_sd ** 2))
        lo_hi.append((s.mean - half, s.mean + half))
    y0 = min(0.0, min(l for l, _ in lo_hi)); y1 = max(h for _, h in lo_hi) * 1.1 or 1.0
    n = max(len(visits) - 1, 1)
    X = lambda i: L + (W - L - R) * (i / n if len(visits) > 1 else 0.5)
    Y = lambda y: T + (H - T - B) * (1 - (y - y0) / (y1 - y0))
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">',
         f'<text x="{L}" y="16" class="t">{html.escape(title)} ({unit}, mean ±95% interval incl. remount error)</text>',
         f'<line x1="{L}" y1="{Y(y0)}" x2="{W-R}" y2="{Y(y0)}" class="ax"/><line x1="{L}" y1="{T}" x2="{L}" y2="{Y(y0)}" class="ax"/>']
    for k in range(5):
        yy = y0 + (y1 - y0) * k / 4
        g.append(f'<text x="{L-6}" y="{Y(yy)+4}" class="tk" text-anchor="end">{yy:.3f}</text>')
    for i, v in enumerate(visits):
        g.append(f'<text x="{X(i)}" y="{H-24}" class="tk" text-anchor="middle">{html.escape(v["label"])}</text>')
        if not v["usable"]:
            g.append(f'<text x="{X(i)}" y="{H-10}" class="bad" text-anchor="middle">no usable data</text>')
    path = " ".join(f"{X(i):.1f},{Y(s.mean):.1f}" for i, s in pts)
    g.append(f'<polyline points="{path}" class="ln"/>')
    for (i, s), (lo, hi) in zip(pts, lo_hi):
        g.append(f'<line x1="{X(i)}" x2="{X(i)}" y1="{Y(lo)}" y2="{Y(hi)}" class="eb"/>'
                 f'<circle cx="{X(i)}" cy="{Y(s.mean)}" r="4" class="dot"/>')
    g.append("</svg>")
    return "".join(g)


def render(patient_label: str, visits: list, remount_sd_push: float, remount_sd_brake: float) -> str:
    """remount_sd_* are MEASURED remount SDs in BW·s (from the bench remount study); required."""
    e = html.escape
    rows, base = [], next((v for v in visits if v["usable"]), None)
    for v in visits:
        if not v["usable"]:
            rows.append(f'<tr class="fail"><td>{e(v["label"])}</td><td colspan="3">No usable data: {e(", ".join(v["flags"]))}</td></tr>')
            continue
        if v is base:
            rows.append(f'<tr><td>{e(v["label"])}</td><td>{v["push"].mean:.3f}</td><td>baseline</td><td>baseline</td></tr>')
            continue
        c = lg.change(base["push"], v["push"], remount_sd_push); c2 = lg.change(base["brake"], v["brake"], remount_sd_brake)
        fmt = lambda c: (f'{c["delta"]:+.3f} (limit ±{c["mdc95"]:.3f}) ' +
                         ('<b>exceeds measurement error</b>' if c["exceeds_measurement_error"] else 'within measurement error'))
        rows.append(f'<tr><td>{e(v["label"])}</td><td>{v["push"].mean:.3f}</td><td>{fmt(c)}</td><td>{fmt(c2)}</td></tr>')
    warn = "".join(f"<li>{e(v['label'])}: {e(w)}</li>" for v in visits for w in v["warnings"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GaitSense monitoring</title><style>
:root{{--bg:#fff;--fg:#1c2430;--mut:#667;--ln:#2a6fdb;--bad:#b3261e;--bd:#d8dde5}}
@media(prefers-color-scheme:dark){{:root{{--bg:#12161c;--fg:#e6eaf0;--mut:#9aa4b2;--ln:#6aa6ff;--bad:#ff8a80;--bd:#2c3440}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:900px;padding:16px}}
.banner{{border:1px solid var(--bad);color:var(--bad);padding:8px 12px;border-radius:6px;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;display:block;overflow-x:auto}}td,th{{border-bottom:1px solid var(--bd);padding:6px 8px;text-align:left}}
tr.fail td{{color:var(--bad)}}svg{{width:100%;height:auto}}.t{{font-size:12px;fill:var(--fg)}}.tk{{font-size:10px;fill:var(--mut)}}
.ax{{stroke:var(--mut)}}.ln{{fill:none;stroke:var(--ln);stroke-width:2}}.eb{{stroke:var(--ln);stroke-width:2}}.dot{{fill:var(--ln)}}
.bad{{font-size:10px;fill:var(--bad)}}.muted{{color:var(--mut)}}
</style></head><body>
<div class="banner">{e(PRELIM)}</div>
<h1>Rehabilitation monitoring: {e(patient_label)}</h1>
<h2>Push-off impulse</h2>{_svg_trend(visits, "push", remount_sd_push, "Push-off impulse")}
<h2>Braking impulse</h2>{_svg_trend(visits, "brake", remount_sd_brake, "Braking impulse")}
<h2>Change from first usable visit</h2>
<table><tr><th>Visit</th><th>Push-off mean</th><th>Push-off change</th><th>Braking change</th></tr>{"".join(rows)}</table>
<p class="muted">A change inside the limit cannot be distinguished from sampling noise plus sensor remount error. A change outside it is a measured force change, not evidence of recovery, work readiness or treatment effect.</p>
{"<h2>Warnings</h2><ul>"+warn+"</ul>" if warn else ""}
</body></html>"""
