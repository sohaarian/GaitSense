import numpy as np
import pandas as pd
import pytest

from gaitrec_rehab.synthetic import synthetic_trials as _trials
from gaitrec_rehab.monitoring import (changes_from_first, reference_percentiles,
                                      review_flags, session_table)


def test_session_difference_and_weeks():
    t = session_table(_trials(1, "K", [-0.25, -0.18, -0.09], noise=0.0001))
    assert t["peak_loading"].round(2).tolist() == [-0.25, -0.18, -0.09]
    assert t["week"].tolist() == [1, 2, 3]


def test_change_within_noise_is_not_called_change():
    t = session_table(_trials(1, "K", [-0.20, -0.195], noise=0.05))
    c = changes_from_first(t)
    assert not c.iloc[-1]["peak_loading_beyond_noise"]


def test_large_improvement_is_beyond_noise():
    t = session_table(_trials(1, "K", [-0.25, -0.05], noise=0.01))
    c = changes_from_first(t)
    assert c.iloc[-1]["peak_loading_beyond_noise"] and c.iloc[-1]["peak_loading_change"] > 0


def test_reference_must_exclude_patient():
    t = session_table(_trials(1, "K", [-0.2, -0.1]))
    with pytest.raises(ValueError):
        reference_percentiles(t, t)


def _reference(n=40):
    frames = [_trials(1000 + i, "K", [-0.10, -0.06, -0.03], seed=i) for i in range(n)]
    return session_table(pd.concat(frames))


def test_persistent_deficit_flag():
    ref = _reference()
    p = session_table(_trials(1, "K", [-0.30, -0.30, -0.30], noise=0.01))
    flags = review_flags(changes_from_first(p), reference_percentiles(ref, p))
    assert "persistent_loading_deficit" in flags


def test_typical_recovery_raises_no_flag():
    ref = _reference()
    p = session_table(_trials(2, "K", [-0.10, -0.06, -0.03], noise=0.01, seed=99))
    assert review_flags(changes_from_first(p), reference_percentiles(ref, p)) == []


def test_decline_flag():
    ref = _reference()
    p = session_table(_trials(3, "K", [-0.05, -0.10, -0.25], noise=0.01))
    assert "loading_declined" in review_flags(changes_from_first(p), reference_percentiles(ref, p))


def test_above_typical_range_flag():
    ref = _reference()
    p = session_table(_trials(4, "K", [0.10, 0.12, 0.15], noise=0.01))
    assert "loading_above_typical_range" in review_flags(changes_from_first(p), reference_percentiles(ref, p))


def test_between_session_sd_recovers_planted_day_to_day_variation():
    from gaitrec_rehab.monitoring import between_session_sd
    rng = np.random.default_rng(5)
    frames = []
    for i in range(150):
        day_effects = list(-0.1 + rng.normal(0, 0.03, 4))      # session-level shifts, SD 0.03
        frames.append(_trials(2000 + i, "K", day_effects, noise=0.01, seed=i))
    est = between_session_sd(session_table(pd.concat(frames)))["peak_loading"]
    # a change between two sessions carries two session effects: sqrt(2) * 0.03 = 0.042
    assert est == pytest.approx(0.042, abs=0.006)


def test_corrected_limit_is_wider_than_within_session_limit():
    from gaitrec_rehab.monitoring import changes_from_first
    p = session_table(_trials(1, "K", [-0.20, -0.25], noise=0.01))
    narrow = changes_from_first(p).iloc[-1]["peak_loading_noise_limit"]
    wide = changes_from_first(p, {"peak_loading": 0.04, "braking_impulse": 0, "propulsive_impulse": 0}).iloc[-1]["peak_loading_noise_limit"]
    assert wide > narrow and wide == pytest.approx(1.96 * np.sqrt((narrow / 1.96) ** 2 + 0.04 ** 2))


def test_corrected_limit_misses_a_decline_the_original_catches():
    """Documents the trade-off in hazard H14: a 0.06 BW decline is flagged under the
    same-day limit but not under the corrected limit (tau = 0.038)."""
    p = session_table(_trials(1, "K", [-0.20, -0.26], noise=0.01))
    tau = {"peak_loading": 0.038, "braking_impulse": 0, "propulsive_impulse": 0}
    assert changes_from_first(p).iloc[-1]["peak_loading_beyond_noise"]
    assert not changes_from_first(p, tau).iloc[-1]["peak_loading_beyond_noise"]
