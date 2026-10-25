import numpy as np
import pytest

from gaitrec_rehab.features import is_valid_trial, session_difference, trial_features

FS, BW, T = 1000.0, 800.0, 0.6        # sample rate (Hz), body weight (N), stance time (s)
t = np.arange(0, T, 1 / FS)


def test_impulses_match_analytic_values():
    amp = 0.2 * BW
    fap = -amp * np.sin(2 * np.pi * t / T)           # braking first half, propulsion second half
    fz = BW * np.sin(np.pi * t / T)
    f = trial_features(fz, fap, FS, BW)
    expected = 0.2 * T / np.pi                       # area of one half-sine lobe, in BW*s
    assert f["braking_impulse"] == pytest.approx(expected, rel=1e-3)
    assert f["propulsive_impulse"] == pytest.approx(expected, rel=1e-3)
    assert f["peak_loading"] == pytest.approx(1.0, rel=1e-3)


def test_rejects_mismatched_or_short_curves():
    with pytest.raises(ValueError):
        trial_features(np.ones(100), np.ones(99), FS, BW)
    with pytest.raises(ValueError):
        trial_features(np.ones(5), np.ones(5), FS, BW)


def test_quality_rule():
    good = BW * np.sin(np.pi * t / T)
    assert is_valid_trial(good, BW)
    assert not is_valid_trial(good * 0.3, BW)        # implausibly low peak
    bad = good.copy(); bad[10] = np.nan
    assert not is_valid_trial(bad, BW)


def test_session_difference_needs_three_valid_trials_per_leg():
    trial = {"peak_loading": 1.1, "braking_impulse": 0.05, "propulsive_impulse": 0.06}
    weaker = {"peak_loading": 1.0, "braking_impulse": 0.04, "propulsive_impulse": 0.03}
    assert session_difference([weaker] * 2, [trial] * 5) is None
    d = session_difference([weaker] * 3, [trial] * 3)
    assert d["propulsive_impulse"] == pytest.approx(-0.03)
