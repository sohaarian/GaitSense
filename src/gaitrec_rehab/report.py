"""Self-contained HTML review page for one patient's rehabilitation stay."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

from .monitoring import FEATURES

LABELS = {
    "peak_loading": ("Weight accepted by injured leg", "body weights"),
    "braking_impulse": ("Braking by injured leg", "body-weight·s"),
    "propulsive_impulse": ("Push-off by injured leg", "body-weight·s"),
}
FLAG_TEXT = {
    "persistent_loading_deficit": "Loading on the injured leg is low for this injury group and week, "
                                  "and has not changed beyond measurement noise since the first assessment.",
    "loading_declined": "Loading on the injured leg is lower than at the first assessment, "
                        "by more than measurement noise.",
    "loading_above_typical_range": "Loading on the injured leg relative to the other leg is above the "
                                   "typical range for this injury group and week.",
}
FIXED_NOTICE = ("Research prototype. These are walking-force measurements, not a diagnosis or an assessment of "
                "recovery, work capacity, causation or impairment. Walking speed was not recorded and can change "
                "these values. Interpret only alongside the treating clinician's assessment.")
INJURY = {"H": "Hip", "K": "Knee", "A": "Ankle", "C": "Heel (calcaneus)"}


def _chart(p: pd.DataFrame, ref: pd.DataFrame | None, f: str) -> str:
    W, H, L, R, T, B = 560, 220, 64, 44, 24, 36
    weeks = p["week"].tolist()
    lo = (p[f] - 1.96 * p[f + "_se"]).tolist()
    hi = (p[f] + 1.96 * p[f + "_se"]).tolist()
    band = []
    if ref is not None:
        for w in weeks:
            pool = ref.loc[(ref["CLASS_LABEL"] == p["CLASS_LABEL"].iat[0]) & (ref["week"] == w), f]
            band.append((np.percentile(pool, 10), np.percentile(pool, 90)) if len(pool) >= 20 else None)
    vals = lo + hi + [0.0] + [v for b in band if b for v in b]
    ymin, ymax = min(vals), max(vals)
    pad = (ymax - ymin) * 0.12 or 0.05
    ymin, ymax = ymin - pad, ymax + pad
    xmax = max(weeks) if len(weeks) > 1 else 2
    X = lambda w: L + (w - 1) / max(xmax - 1, 1) * (W - L - R)
    Y = lambda v: T + (ymax - v) / (ymax - ymin) * (H - T - B)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{LABELS[f][0]} by week">']
    pts = [(X(w), b) for w, b in zip(weeks, band) if b]
    if len(pts) >= 2:
        top = " ".join(f"{x:.1f},{Y(b[1]):.1f}" for x, b in pts)
        bot = " ".join(f"{x:.1f},{Y(b[0]):.1f}" for x, b in reversed(pts))
        s.append(f'<polygon points="{top} {bot}" class="band"/>')
    s.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
    s.append(f'<text x="{L+4}" y="{T-8}" class="tick">Dashed line: legs equal</text>')
    for w in weeks:
        s.append(f'<text x="{X(w):.1f}" y="{H-12}" class="tick" text-anchor="middle">Week {w}</text>')
    line = " ".join(f"{X(w):.1f},{Y(v):.1f}" for w, v in zip(weeks, p[f]))
    s.append(f'<polyline points="{line}" class="trend"/>')
    for w, v, a, b in zip(weeks, p[f], lo, hi):
        s.append(f'<line x1="{X(w):.1f}" x2="{X(w):.1f}" y1="{Y(a):.1f}" y2="{Y(b):.1f}" class="err"/>')
        s.append(f'<circle cx="{X(w):.1f}" cy="{Y(v):.1f}" r="4.5" class="dot"/>')
    span = ymax - ymin
    fmt = "+.2f" if span >= 0.2 else "+.3f" if span >= 0.02 else "+.4f"
    for v in np.linspace(ymin + pad, ymax - pad, 3):
        s.append(f'<text x="{L-8}" y="{Y(v)+4:.1f}" class="tick" text-anchor="end">{v:{fmt}}</text>')
    s.append("</svg>")
    return "".join(s)


def render(patient: pd.DataFrame, changes: pd.DataFrame, flags: list[str],
           reference: pd.DataFrame | None = None, synthetic: bool = False) -> str:
    p = patient.sort_values("week")
    last = changes.iloc[-1]
    rows = []
    for f in FEATURES:
        ch = last[f + "_change"]
        verdict = "changed beyond measurement noise" if last[f + "_beyond_noise"] else "within measurement noise"
        rows.append(f"<tr><td>{LABELS[f][0]}</td><td>{p[f].iat[0]:+.3f}</td><td>{p[f].iat[-1]:+.3f}</td>"
                    f"<td>{ch:+.3f}</td><td>{verdict}</td></tr>")
    flag_html = ("".join(f'<li>{html.escape(FLAG_TEXT[k])}</li>' for k in flags)
                 or "<li>No pre-specified review prompt.</li>")
    charts = "".join(f'<section class="card"><h3>{LABELS[f][0]}</h3><p class="unit">Injured minus uninjured leg, '
                     f'{LABELS[f][1]}. Bars: 95% measurement-noise range. Shaded: 10th–90th percentile of '
                     f'{INJURY.get(p["CLASS_LABEL"].iat[0], "")} patients in the same week.</p>'
                     f'{_chart(p, reference, f)}</section>' for f in FEATURES)
    banner = '<div class="synthetic">Synthetic example. Not patient data.</div>' if synthetic else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Rehabilitation Gait Review</title>
<style>
:root{{--bg:#fbfaf7;--card:#fff;--ink:#1d1d1b;--muted:#5f5e58;--rule:#e3e0d8;--accent:#2f5d8a;--band:#2f5d8a1f;--warn:#9a5b00;--warnbg:#fff4e0}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151514;--card:#1e1e1c;--ink:#ecebe6;--muted:#a8a69e;--rule:#34332f;--accent:#8fb6dd;--band:#8fb6dd26;--warn:#f0b35c;--warnbg:#3a2c12}}}}
:root[data-theme="dark"]{{--bg:#151514;--card:#1e1e1c;--ink:#ecebe6;--muted:#a8a69e;--rule:#34332f;--accent:#8fb6dd;--band:#8fb6dd26;--warn:#f0b35c;--warnbg:#3a2c12}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:24px 16px 48px}} h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}} h3{{font-size:15px;margin:0 0 2px}}
.meta,.unit{{color:var(--muted);font-size:13px;margin:0 0 10px}} .card{{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:16px;margin:12px 0}}
.notice{{border-left:3px solid var(--accent);padding:10px 14px;background:var(--card);font-size:13.5px;color:var(--muted)}}
.flags{{background:var(--warnbg);border:1px solid var(--rule);border-radius:10px;padding:12px 16px}} .flags h2{{margin:0 0 6px;color:var(--warn)}}
.synthetic{{display:inline-block;font-size:12px;font-weight:600;color:var(--warn);border:1px solid var(--warn);border-radius:999px;padding:2px 10px;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;font-size:14px}} th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--rule)}} th{{color:var(--muted);font-weight:600}}
.tablewrap{{overflow-x:auto}} svg{{width:100%;height:auto;display:block}} .band{{fill:var(--band)}} .zero{{stroke:var(--rule);stroke-dasharray:4 4}}
.trend{{fill:none;stroke:var(--accent);stroke-width:2}} .dot{{fill:var(--accent)}} .err{{stroke:var(--accent);stroke-width:1.5}} .tick{{fill:var(--muted);font-size:12px}}
</style></head><body><main>
{banner}<h1>Rehabilitation gait review</h1>
<p class="meta">Patient {html.escape(str(p['SUBJECT_ID'].iat[0]))} · {INJURY.get(p['CLASS_LABEL'].iat[0], '')} injury · {len(p)} assessments over {int(p['day'].iat[-1])} days</p>
<p class="notice">{FIXED_NOTICE}</p>
<div class="flags"><h2>Review prompts</h2><ul>{flag_html}</ul></div>
<h2>First and latest assessment</h2>
<div class="card tablewrap"><table><tr><th>Measure (injured minus uninjured)</th><th>First</th><th>Latest</th><th>Change</th><th>Compared with noise</th></tr>{''.join(rows)}</table></div>
<h2>By week</h2>{charts}
<p class="unit">Measurement-noise ranges come from trial-to-trial variation within each session and do not include day-to-day variation, so they understate total uncertainty.</p>
</main></body></html>"""
