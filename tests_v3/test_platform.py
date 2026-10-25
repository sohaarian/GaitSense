"""v3.0 platform tests: gate, contracts, linkage leakage, Capability 1 interface, Capability 2 controls."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gaitsense_platform import clinical, gate, linkage, pipeline, synthetic
from gaitsense_platform.capability1 import BiomechForecaster, ConformalCalibrator, load_v2
from gaitsense_platform.contracts import ContractError, DatasetManifest, validate_biomech, validate_outcomes


def real_manifest(tmp_path, **over):
    f = tmp_path / "visits.csv"; f.write_text("patient_id,visit_day,value,value_se\n1,0,0.1,0.01\n")
    d = dict(name="Example", source_url="https://example.org/data", citation="Author 2026", licence_or_agreement="DUA 123",
             access_type="controlled", ethics_or_dua_reference="IRB 456", obtained_by="Researcher", obtained_on="2027-01-01",
             verified_by="Second researcher", verified_on="2027-01-02", is_synthetic=False,
             file_sha256={"visits.csv": gate.sha256(f)})
    d.update(over)
    return DatasetManifest(**d)


def test_gate_refuses_synthetic_without_explicit_permission():
    *_, m = synthetic.make(50, seed=1)
    with pytest.raises(gate.GateError):
        gate.check(m)
    assert gate.check(m, allow_synthetic=True)["banner"] == gate.SYNTHETIC_BANNER


def test_gate_refuses_incomplete_or_placeholder_manifest(tmp_path):
    with pytest.raises(gate.GateError):
        gate.check(real_manifest(tmp_path, ethics_or_dua_reference="TBD"), root=tmp_path)
    with pytest.raises(gate.GateError):
        gate.check(real_manifest(tmp_path, file_sha256={}), root=tmp_path)


def test_gate_detects_changed_or_missing_files(tmp_path):
    m = real_manifest(tmp_path)
    assert gate.check(m, root=tmp_path)["status"] == "verified"
    (tmp_path / "visits.csv").write_text("patient_id,visit_day,value,value_se\n1,0,0.2,0.01\n")
    with pytest.raises(gate.GateError, match="checksum"):
        gate.check(m, root=tmp_path)
    with pytest.raises(gate.GateError, match="not found"):
        gate.check(real_manifest(tmp_path, file_sha256={"missing.csv": "0" * 64}), root=tmp_path)


def test_pipeline_refuses_unverified_data_before_any_modelling():
    v, c, o, m = synthetic.make(60, seed=2)
    cfg = pipeline.Config("meaningful_improvement", 42, 90, 14, 120, True, ("age", "severity"))
    with pytest.raises(gate.GateError):
        pipeline.run(m, v, c, o, cfg)


def test_outcomes_must_be_independent_of_biomechanics():
    o = pd.DataFrame({"patient_id": [1], "outcome": ["x"], "instrument": ["gait-derived"], "value": [1.0],
                      "assessment_day": [90], "independent_of_biomechanics": [False]})
    with pytest.raises(ContractError, match="independently"):
        validate_outcomes(o)


def test_biomech_contract_rejects_bad_tables():
    with pytest.raises(ContractError):
        validate_biomech(pd.DataFrame({"patient_id": [1], "visit_day": [0], "value": [np.nan], "value_se": [0.01]}))
    with pytest.raises(ContractError):
        validate_biomech(pd.DataFrame({"patient_id": [1, 1], "visit_day": [0, 0], "value": [0.1, 0.2], "value_se": [0.01, 0.01]}))


def test_landmark_features_ignore_visits_after_the_landmark():
    v, *_ = synthetic.make(120, seed=3)
    fc = BiomechForecaster().fit(v[v.visit_day <= 21], restarts=1)
    f1 = fc.landmark_features(v, 21, 90)
    v2 = v.copy(); v2.loc[v2.visit_day > 21, "value"] += 5.0
    f2 = fc.landmark_features(v2, 21, 90)
    pd.testing.assert_frame_equal(f1, f2)


def test_forecast_rejects_future_history():
    v, *_ = synthetic.make(40, seed=4)
    fc = BiomechForecaster().fit(v, restarts=1)
    p = v[v.patient_id == v.patient_id.iloc[0]]
    with pytest.raises(ValueError):
        fc.forecast(p, day=-1)


def test_linkage_excludes_outcomes_before_landmark_plus_gap():
    o = pd.DataFrame({"patient_id": [1, 1, 2], "outcome": "f", "instrument": "scale", "value": [10.0, 20.0, 30.0],
                      "assessment_day": [40, 70, 50], "independent_of_biomechanics": True})
    r = linkage.outcome_at(o, "f", landmark_day=42, min_gap_days=14, window_end_day=120)
    assert r.patient_id.tolist() == [1] and r.value.tolist() == [20.0]


def test_covariates_after_landmark_are_dropped():
    c = pd.DataFrame({"patient_id": [1, 2], "available_day": [0, 60], "x": [1.0, 2.0]})
    assert linkage.clinical_at(c, 42).patient_id.tolist() == [1]


def test_negative_control_finds_no_added_value_and_positive_control_does():
    cfg = pipeline.Config("meaningful_improvement", 42, 90, 14, 120, True, ("age", "severity"))
    v, c, o, m = synthetic.make(800, bio_effect=0.0, seed=11)
    neg = pipeline.run(m, v, c, o, cfg, allow_synthetic=True)["results"]["C_minus_A"]["auroc"]
    v, c, o, m = synthetic.make(800, bio_effect=1.5, seed=11)
    pos = pipeline.run(m, v, c, o, cfg, allow_synthetic=True)["results"]["C_minus_A"]["auroc"]
    assert neg["ci95"][0] < 0 < neg["ci95"][1]
    assert pos["ci95"][0] > 0


def test_continuous_endpoint_comparison_runs():
    rng = np.random.default_rng(0); n = 300
    t = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}); t["outcome"] = 2 * t.a + rng.normal(size=n)
    r = clinical.compare_models(t, ["a"], ["b"], binary=False, B=200)
    assert r["C_minus_A"]["mae"]["ci95"][0] < 0.05 and r["models"]["A_clinical"]["mae"] < r["models"]["B_biomech"]["mae"]


def test_meaningful_improvement_endpoint():
    assert clinical.meaningful_improvement(pd.Series([10, 10]), pd.Series([20, 12]), mcid=9).tolist() == [1, 0]
    assert clinical.meaningful_improvement(pd.Series([5, 5]), pd.Series([1, 4]), mcid=2, higher_is_better=False).tolist() == [1, 0]


def test_conformal_calibrator_gives_finite_sample_coverage():
    rng = np.random.default_rng(1)
    y_cal = rng.standard_t(3, 4000); y_te = rng.standard_t(3, 20000)
    cal = ConformalCalibrator().fit(y_cal, 0.0, 1.0)
    for lv in (0.5, 0.8, 0.95):
        assert np.mean(np.abs(y_te) <= cal.quantile(lv)) == pytest.approx(lv, abs=0.015)


@pytest.mark.skipif(not (ROOT / "results" / "ai" / "models.pkl").exists(), reason="v2.0 outputs absent")
def test_capability1_v2_models_load_unchanged():
    v2 = load_v2(ROOT / "results" / "ai")
    sel = json.loads((ROOT / "results" / "ai" / "model_selection.json").read_text())
    assert set(v2["fitted"]) == set(sel["targets"])
    k = sel["targets"]["d_peak_v"]["M3_chosen"]
    assert v2["fitted"]["d_peak_v"][k].hyperparameters()["sigma_w"] == pytest.approx(
        sel["targets"]["d_peak_v"]["hyperparameters"][k]["sigma_w"])
