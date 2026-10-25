import numpy as np
import pandas as pd
import pytest

from gaitrec_rehab.safeguards import (LeakageError, assert_no_future_information,
                                       assert_single_patient_sessions, patient_folds)


def test_merged_patients_in_one_session_are_rejected():
    trials = pd.DataFrame({"SESSION_ID": [1, 1, 2], "SUBJECT_ID": [10, 11, 12]})
    with pytest.raises(LeakageError):
        assert_single_patient_sessions(trials)


def test_future_features_are_rejected():
    idx = pd.Series(pd.to_datetime(["2010-01-01", "2010-01-01"]))
    ok = pd.Series(pd.to_datetime(["2010-01-01", "2009-12-31"]))
    late = pd.Series(pd.to_datetime(["2010-01-01", "2010-01-02"]))
    assert_no_future_information(ok, idx)
    with pytest.raises(LeakageError):
        assert_no_future_information(late, idx)


def test_folds_keep_each_patient_on_one_side():
    rng = np.random.default_rng(0)
    subjects = np.repeat(np.arange(200), 3)          # three rows per patient
    y = np.repeat(rng.random(200) < 0.1, 3).astype(int)
    for train, test in patient_folds(subjects, y):
        assert not set(subjects[train]) & set(subjects[test])
