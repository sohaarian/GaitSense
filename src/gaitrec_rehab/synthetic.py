"""Synthetic trial data for tests and interface demos. Never patient data."""
import numpy as np
import pandas as pd


def synthetic_trials(sid, cls, weekly_diffs, noise=0.01, n=5, seed=0):
    """Synthetic trials: unaffected leg at 1.10 BW, affected leg at 1.10 + diff."""
    rng, rows = np.random.default_rng(seed), []
    for w, d in enumerate(weekly_diffs):
        date = pd.Timestamp("2015-01-05") + pd.Timedelta(days=7 * w)
        for leg, level in (("affected", 1.10 + d), ("unaffected", 1.10)):
            for _ in range(n):
                rows.append(dict(SUBJECT_ID=sid, SESSION_ID=sid * 100 + w, SESSION_DATE=date,
                                 CLASS_LABEL=cls, leg=leg,
                                 peak_loading=level + rng.normal(0, noise),
                                 braking_impulse=0.05 + rng.normal(0, noise / 10),
                                 propulsive_impulse=0.06 + rng.normal(0, noise / 10)))
    return pd.DataFrame(rows)
