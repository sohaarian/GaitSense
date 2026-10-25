"""Stage 7a: cache raw stance-phase trials (both legs) of test-partition patient visits for perturbation experiments.

Writes data/derived/ai_raw_test.pkl: {SESSION_ID: {"left": [(fz, fap), ...], "right": [...]}} (newtons).
"""
import pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_ai import ingest

raw = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw"
V = pd.read_csv(ROOT / "results" / "ai" / "visit_level_features.csv.gz")
keep = set(V[(V.split == "test") & (V.role == "cohort")].SESSION_ID)
valid = pd.read_csv(ROOT / "data" / "derived" / "ai_trials.csv.gz", usecols=["SESSION_ID", "TRIAL_ID", "side", "valid"])
ok = set(map(tuple, valid[valid.valid][["SESSION_ID", "TRIAL_ID", "side"]].to_numpy()))
cache = {}
for side in ("left", "right"):
    for keys, fv, fa in ingest.iter_raw(raw, side):
        for k in range(len(keys)):
            sess, trial = int(keys[k, 1]), int(keys[k, 2])
            if sess in keep and (sess, trial, side) in ok:
                z, a, _ = ingest.stance(fv[k], fa[k])
                cache.setdefault(sess, {"left": [], "right": []})[side].append((z.astype(np.float64), a.astype(np.float64)))
with open(ROOT / "data" / "derived" / "ai_raw_test.pkl", "wb") as f:
    pickle.dump(cache, f)
print("sessions cached", len(cache), "of", len(keep))
