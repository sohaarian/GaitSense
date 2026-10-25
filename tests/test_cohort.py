from pathlib import Path

import pandas as pd
import pytest

from gaitrec_rehab.cohort import CohortRules, build_cohort

REAL = Path(__file__).parents[1] / "data" / "GRF_metadata.csv"


def _row(sid, sess, date, readm=0, side=0, shod=1, cls="K"):
    return dict(SUBJECT_ID=sid, SESSION_ID=sess, CLASS_LABEL=cls, SEX=1, AGE=40, BODY_MASS=80.0,
                AFFECTED_SIDE=side, SHOD_CONDITION=shod, ORTHOPEDIC_INSOLE=0,
                READMISSION=readm, SESSION_DATE=pd.Timestamp(date))


@pytest.fixture
def toy():
    rows = [
        _row(1, 1, "2010-01-01"), _row(1, 2, "2010-03-01", readm=1),      # event at day 59
        _row(2, 3, "2010-01-01"), _row(2, 4, "2011-06-01", readm=1),      # return after window: no event
        _row(3, 5, "2010-01-01", side=2),                                 # both sides: excluded
        _row(4, 6, "2010-01-01", shod=0),                                 # barefoot: excluded
        _row(5, 7, "2012-06-01"),                                         # under 365 days of window: excluded
        _row(6, 8, "2010-02-01", readm=1),                                # no first-stay session: excluded
        _row(7, 9, "2010-01-01", cls="HC"),                               # healthy control: ignored
        _row(8, 10, "2012-09-01"),                                        # last patient session (end date)
    ]
    return pd.DataFrame(rows)


def test_flow_and_events(toy):
    cohort, flow = build_cohort(toy)
    assert dict(flow)["Patients in GaitRec"] == 7
    assert sorted(cohort["SUBJECT_ID"]) == [1, 2]
    assert cohort.set_index("SUBJECT_ID")["event"].to_dict() == {1: 1, 2: 0}


def test_event_uses_first_later_session_only(toy):
    toy = pd.concat([toy, pd.DataFrame([_row(1, 11, "2010-02-01", readm=2)])])
    cohort, _ = build_cohort(toy)
    assert cohort.set_index("SUBJECT_ID").loc[1, "days_to_event"] == 31


def test_index_is_earliest_first_stay_session(toy):
    toy = pd.concat([toy, pd.DataFrame([_row(1, 12, "2009-12-25")])])
    cohort, _ = build_cohort(toy)
    assert cohort.set_index("SUBJECT_ID").loc[1, "index_session"] == 12


def test_window_rule_respects_explicit_end_date(toy):
    cohort, _ = build_cohort(toy, CohortRules(end_date=pd.Timestamp("2010-06-01")))
    assert cohort.empty


@pytest.mark.skipif(not REAL.exists(), reason="GaitRec metadata not present")
def test_reproduces_audited_cohort():
    from gaitrec_rehab.cohort import load_metadata
    cohort, flow = build_cohort(load_metadata(str(REAL)))
    assert [n for _, n in flow] == [2084, 2076, 1975, 1759, 1667, 84]
    by_group = cohort.groupby("CLASS_LABEL")["event"].agg(["size", "sum"])
    assert by_group.loc["K"].tolist() == [525, 32]
    assert by_group.loc["A"].tolist() == [511, 27]
    assert by_group.loc["H"].tolist() == [370, 14]
    assert by_group.loc["C"].tolist() == [261, 11]
