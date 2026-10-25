"""Checks that stop the three failures that would invalidate the study."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class LeakageError(RuntimeError):
    pass


def assert_single_patient_sessions(trials: pd.DataFrame) -> None:
    """Hazard H1: trials from two patients merged into one session."""
    per_session = trials.groupby("SESSION_ID")["SUBJECT_ID"].nunique()
    bad = per_session[per_session > 1]
    if len(bad):
        raise LeakageError(f"sessions with trials from more than one patient: {list(bad.index[:5])}")


def assert_no_future_information(feature_dates: pd.Series, index_dates: pd.Series) -> None:
    """Hazard H2: any predictor dated after the prediction time."""
    late = feature_dates > index_dates
    if late.any():
        raise LeakageError(f"{int(late.sum())} features dated after the index assessment")


def patient_folds(subject_ids, y, n_splits: int = 5, seed: int = 0):
    """Hazard H3: patient-level stratified folds, verified disjoint before use."""
    subject_ids, y = np.asarray(subject_ids), np.asarray(y)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in cv.split(np.zeros(len(y)), y, groups=subject_ids):
        if set(subject_ids[train]) & set(subject_ids[test]):
            raise LeakageError("patient present in both training and test data")
        yield train, test
