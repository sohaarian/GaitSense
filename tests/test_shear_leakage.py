"""Leakage guards for the shear-force study."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from shear_data import patient_split  # noqa: E402

PAIRS = ROOT / "results" / "shear_pairs_test.csv"
RES = ROOT / "results" / "shear_study.json"


def test_patient_split_is_disjoint_complete_and_reproducible():
    ids = np.arange(1000, 2667)
    s = patient_split(np.concatenate([ids, ids[:50]]))  # duplicates must not change the split
    assert set(s) == set(ids)
    groups = {k: {i for i, v in s.items() if v == k} for k in ("train", "val", "test")}
    assert not (groups["train"] & groups["val"]) and not (groups["train"] & groups["test"]) and not (groups["val"] & groups["test"])
    assert [len(groups[k]) for k in ("train", "val", "test")] == [1001, 333, 333]
    assert patient_split(ids[::-1]) == s  # order of input does not matter


@pytest.mark.skipif(not (PAIRS.exists() and RES.exists()), reason="shear study results not present")
def test_evaluation_pairs_come_only_from_test_patients():
    p = pd.read_csv(PAIRS)
    assert (p["split"] == "test").all()
    meta = pd.read_csv(ROOT / "data" / "GRF_metadata.csv")
    sys.path.insert(0, str(ROOT / "src"))
    from gaitrec_rehab.cohort import build_cohort, load_metadata
    cohort, _ = build_cohort(load_metadata(str(ROOT / "data" / "GRF_metadata.csv")))
    s = patient_split(cohort.SUBJECT_ID)
    assert {s[i] for i in p.SUBJECT_ID.unique()} == {"test"}
    r = json.loads(RES.read_text())
    assert r["n_patients"] == {"train": 1001, "val": 333, "test": 333}
    # thresholds and the choice of "the AI" use validation patients only
    assert "tau_from_validation" in r and "validation_push_change_ccc" in r["ai_choice"]
    assert len(meta) > 0
