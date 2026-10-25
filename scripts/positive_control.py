"""Positive control: plant an outcome driven by push-off asymmetry into the real cohort (same prevalence)
and confirm the pipeline recovers it. Also reports optimistic in-sample AUROCs. Writes results/positive_control.json."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src"))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from gaitrec_rehab.analysis import MODELS, cv_predict, load_data

d = load_data(ROOT / "data"); c = d.cohort; y = c.event.to_numpy(); rows = np.arange(len(c))
out = {}
for m in ("M0", "M2"):
    X = StandardScaler().fit_transform(c[MODELS[m]].to_numpy(float))
    out[f"{m}_in_sample_unpenalised_auroc"] = float(roc_auc_score(y, LogisticRegression(C=1e6, max_iter=5000).fit(X, y).predict_proba(X)[:, 1]))
rng = np.random.default_rng(1)
z = ((c.propulsive_impulse - c.propulsive_impulse.mean()) / c.propulsive_impulse.std()).to_numpy()
a = -3.5
for _ in range(60):
    a += (0.05 - (1 / (1 + np.exp(-(a - 1.2 * z)))).mean()) * 5
ys = (rng.random(len(c)) < 1 / (1 + np.exp(-(a - 1.2 * z)))).astype(int)
o = cv_predict(d, ys, rows, c.SUBJECT_ID.to_numpy(), ["M0", "M2"], 0, False)
out.update({"planted_events": int(ys.sum()), "planted_true_auroc": float(roc_auc_score(ys, -z)),
            "recovered_M0": float(roc_auc_score(ys, o["M0"])), "recovered_M2": float(roc_auc_score(ys, o["M2"]))})
(ROOT / "results" / "positive_control.json").write_text(json.dumps(out, indent=2)); print(out)
