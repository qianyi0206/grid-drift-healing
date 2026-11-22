import numpy as np
import pandas as pd
import pytest

from grid_drift_healing.drift import PinballDegradationDetector, PSIDriftDetector, psi
from grid_drift_healing.gate import diebold_mariano, promote
from grid_drift_healing.metrics import crps_from_quantiles, interval_coverage, pinball


def test_pinball_known_values():
    assert pinball(np.array([10.0]), np.array([8.0]), 0.9)[0] == pytest.approx(1.8)
    assert pinball(np.array([10.0]), np.array([12.0]), 0.9)[0] == pytest.approx(0.2)


def test_crps_quantile_approx_reasonable():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20000)
    qs = np.linspace(0.01, 0.99, 99)
    from scipy.stats import norm
    preds = {q: np.full_like(y, norm.ppf(q)) for q in qs}
    # exact CRPS of N(0,1) evaluated at N(0,1) draws = 1/sqrt(pi) ~ 0.564
    assert crps_from_quantiles(y, preds).mean() == pytest.approx(1 / np.sqrt(np.pi), rel=0.03)


def test_coverage():
    assert interval_coverage([1, 2, 3, 4], [0, 0, 0, 5], [5, 5, 5, 6]) == 0.75


def test_psi_same_vs_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert psi(ref, rng.normal(size=5000)) < 0.02
    assert psi(ref, rng.normal(1.0, 1, size=5000)) > 0.25


def test_psi_detector():
    rng = np.random.default_rng(1)
    d = PSIDriftDetector(["a"], threshold=0.25)
    d.set_reference(pd.DataFrame({"a": rng.normal(size=2000)}))
    assert not d.check(pd.DataFrame({"a": rng.normal(size=500)}))[0]
    assert d.check(pd.DataFrame({"a": rng.normal(2, 1, size=500)}))[0]


def test_degradation_detector():
    d = PinballDegradationDetector(window=10, ratio=1.25)
    d.reset(1.0)
    d.update(np.ones(10))
    assert not d.check()
    d.update(np.full(10, 2.0))
    assert d.check()


def test_dm_detects_better_candidate():
    rng = np.random.default_rng(2)
    inc = rng.gamma(2, 1.0, 500) + 0.5
    cand = inc - 0.4 + rng.normal(0, 0.2, 500)
    ok, res = promote(inc, cand, alpha=0.05, h=24)
    assert ok and res.p_value < 0.01


def test_dm_rejects_equal_or_worse():
    rng = np.random.default_rng(3)
    inc = rng.gamma(2, 1.0, 500)
    assert not promote(inc, inc + rng.normal(0, 0.5, 500), h=24)[0]
    assert not promote(inc, inc + 0.3, h=24)[0]


def test_dm_size_under_null():
    """Under H0 the one-sided test should reject roughly alpha of the time."""
    rng = np.random.default_rng(4)
    rej = 0
    for _ in range(400):
        a, b = rng.normal(size=200), rng.normal(size=200)
        rej += diebold_mariano(a, b, h=1).p_value < 0.05
    assert 0.02 < rej / 400 < 0.09


def test_dm_needs_data():
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(3), np.ones(3))


def test_degradation_relative_is_scale_free():
    """Relative mode: doubling both model and reference losses (seasonal scale) must not fire."""
    d = PinballDegradationDetector(window=10, ratio=1.25)
    d.reset(0.5)
    d.update(np.full(10, 2.0), np.full(10, 4.0))
    assert d.rolling == pytest.approx(0.5) and not d.check()
    d.update(np.full(10, 3.0), np.full(10, 4.0))
    assert d.check()
