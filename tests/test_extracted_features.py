from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1] / "data"
FILES = [ROOT / "derived" / f"trial_features_{s}.csv" for s in ("left", "right")]


@pytest.mark.skipif(not all(f.exists() for f in FILES), reason="extracted features not present")
def test_extraction_matches_metadata_and_is_plausible():
    t = pd.concat([pd.read_csv(f) for f in FILES])
    meta = pd.read_csv(ROOT / "GRF_metadata.csv")
    assert len(t) == 2 * 75_732
    assert set(t["SESSION_ID"]) == set(meta["SESSION_ID"])
    chk = t.merge(meta[["SESSION_ID", "SUBJECT_ID"]], on="SESSION_ID", suffixes=("", "_m"))
    assert (chk["SUBJECT_ID"] == chk["SUBJECT_ID_m"]).all()
    v = t[t["valid"]]
    assert v["peak_loading"].between(0.5, 2.5).all()
    assert v["stance_s"].median() == pytest.approx(0.73, abs=0.05)


SESSIONS = ROOT / "derived" / "first_stay_session_differences.csv"


@pytest.mark.skipif(not SESSIONS.exists(), reason="session table not present")
def test_affected_side_coding_points_the_right_way():
    """If AFFECTED_SIDE were read backwards, the injured leg would look stronger on average."""
    st = pd.read_csv(SESSIONS)
    first = st.sort_values("day").groupby("SUBJECT_ID").first()
    for f in ("peak_loading", "braking_impulse", "propulsive_impulse"):
        assert (first[f] < 0).mean() > 0.6
        assert (first.groupby("CLASS_LABEL")[f].mean() < 0).all()
