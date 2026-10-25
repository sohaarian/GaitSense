"""Render the review page for a SYNTHETIC patient (no real data), for interface review."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import pandas as pd
from gaitrec_rehab.synthetic import synthetic_trials as _trials
from gaitrec_rehab.monitoring import session_table, changes_from_first, reference_percentiles, review_flags
from gaitrec_rehab.report import render

ref = session_table(pd.concat([_trials(1000 + i, "K", [-0.12 + 0.01*(i%5), -0.07, -0.03, -0.01], seed=i) for i in range(60)]))
p = session_table(_trials(7, "K", [-0.26, -0.25, -0.24, -0.25], noise=0.012, seed=3))
c = changes_from_first(p)
flags = review_flags(c, reference_percentiles(ref, p))
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "demo_review.html")
out.write_text(render(p, c, flags, ref, synthetic=True), encoding="utf-8")
print("flags:", flags, "->", out)
