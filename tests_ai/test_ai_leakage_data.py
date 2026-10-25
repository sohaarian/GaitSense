"""Leakage and reproducibility tests on the real GaitSense AI outputs (skipped if the outputs are absent)."""
import json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]; A = ROOT / "results" / "ai"
sys.path.insert(0, str(ROOT / "src"))
HAVE = (A / "predictions_test.csv.gz").exists() and (A / "models.pkl").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="AI outputs not built")


def load():
    return (pd.read_csv(A / "visit_level_features.csv.gz"), pd.read_csv(A / "partition.csv"),
            json.loads((A / "model_selection.json").read_text()))


@pytest.mark.skipif(not HAVE, reason="AI outputs not built")
def test_1_no_patient_in_two_partitions_and_partition_reproducible():
    from gaitsense_ai import cohort, ingest, splits
    V, part, _ = load()
    splits.assert_disjoint(part)
    roles = cohort.patient_roles(ingest.load_metadata(ROOT / "data" / "GRF_metadata.csv"))
    again = splits.make_partition(roles)
    pd.testing.assert_frame_equal(again.sort_values("SUBJECT_ID").reset_index(drop=True),
                                  part.sort_values("SUBJECT_ID").reset_index(drop=True), check_dtype=False)


@pytest.mark.skipif(not HAVE, reason="AI outputs not built")
def test_2_and_7_stored_predictions_use_only_earlier_visits():
    from gaitsense_ai import prediction as P
    V, part, sel = load()
    with open(A / "models.pkl", "rb") as f:
        pk = pickle.load(f)
    PV = V[V.role.isin(["cohort", "temporal"])]
    arr = P.patient_arrays(PV, "d_peak_v", "d_peak_v_se", pk["std"])
    g = pk["fitted"]["d_peak_v"][sel["targets"]["d_peak_v"]["M3_chosen"]]
    D = pd.read_csv(A / "predictions_test.csv.gz")
    D = D[(D.model == "M3") & (D.target == "d_peak_v")].sample(40, random_state=1)
    for _, r in D.iterrows():
        t, y, se2, X = arr[r.SUBJECT_ID][:4]; j = int(r.k) - 1
        y_future = y.copy(); y_future[j:] += 10.0          # corrupt the target and every later visit
        mu, vl = g.predict_next(t[:j], y_future[:j], se2[:j], X[:j], t[j], X[j])
        assert mu == pytest.approx(r.mu, abs=1e-10)
        assert np.sqrt(vl + g.sigma_w ** 2 + r.sbar2) == pytest.approx(r.sd, abs=1e-10)


@pytest.mark.skipif(not HAVE, reason="AI outputs not built")
def test_3_hyperparameters_chosen_without_test_patients():
    V, part, sel = load()
    opened = json.loads((A / "test_opened.json").read_text())["opened"]
    assert sel["finished"] < opened
    for t in sel["targets"].values():
        assert set(t["n_targets"]) == {"train", "calib"}


@pytest.mark.skipif(not HAVE, reason="AI outputs not built")
def test_4_calibration_patients_not_in_final_evaluation():
    D = pd.read_csv(A / "predictions_test.csv.gz"); V, part, _ = load()
    s = D.SUBJECT_ID.map(part.set_index("SUBJECT_ID").split)
    assert set(s.unique()) <= {"test", "temporal"}


@pytest.mark.skipif(not HAVE, reason="AI outputs not built")
def test_5_healthy_reference_uses_training_controls_at_patient_speed_only():
    V, part, _ = load()
    mon = json.loads((A / "monitoring_results.json").read_text())
    hc = V[(V.role == "hc") & (V.split == "train") & (V.SPEED == 2)]
    assert mon["healthy_reference"]["n_sessions"] == len(hc)
    assert mon["healthy_reference"]["mu"] == pytest.approx(hc.d_peak_v.mean())


@pytest.mark.skipif(not HAVE, reason="AI outputs not built")
def test_6_covariate_scaling_fitted_on_training_patients():
    V, part, _ = load()
    with open(A / "models.pkl", "rb") as f:
        std = pickle.load(f)["std"]
    tr = V[(V.role == "cohort") & (V.split == "train")].drop_duplicates("SUBJECT_ID")
    assert std.age[0] == pytest.approx(tr.AGE.mean()) and std.mass[0] == pytest.approx(tr.BODY_MASS.mean())


@pytest.mark.skipif(not (ROOT / "data" / "derived" / "ai_trials.csv.gz").exists(), reason="derived trial data absent (gitignored)")
def test_new_extraction_reproduces_v1_trial_features():
    T = pd.read_csv(ROOT / "data" / "derived" / "ai_trials.csv.gz")
    H = pd.concat([pd.read_csv(ROOT / "data" / "derived" / f"trial_features_{s}.csv") for s in ("left", "right")])
    M = T.merge(H, on=["SUBJECT_ID", "SESSION_ID", "TRIAL_ID", "side"], suffixes=("", "_h"))
    assert len(M) == len(T) == len(H)
    for a, b in (("wa_peak", "peak_loading"), ("braking", "braking_impulse"), ("propulsive", "propulsive_impulse")):
        assert np.allclose(M[a], M[b], atol=1e-12, equal_nan=True)


@pytest.mark.skipif(not HAVE, reason="AI outputs not built")
def test_protocol_file_matches_frozen_hash():
    import hashlib
    want = (ROOT / "docs" / "ai_protocol.sha256").read_text().split()[0]
    assert hashlib.sha256((ROOT / "docs" / "ai_protocol.md").read_bytes()).hexdigest() == want
