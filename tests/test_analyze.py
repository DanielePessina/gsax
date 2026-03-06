import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gsax
from gsax.benchmarks.ishigami import ANALYTICAL_S1, ANALYTICAL_S2, ANALYTICAL_ST, PROBLEM, evaluate


@pytest.fixture(scope="module")
def ishigami_result():
    """Run Ishigami analysis once for all tests in this module."""
    sr = gsax.sample(PROBLEM, n_samples=2**14 * 8, seed=42)
    Y = evaluate(jnp.asarray(sr.samples))
    return gsax.analyze(sr, Y)


def test_s1_accuracy(ishigami_result):
    S1 = np.array(ishigami_result.S1)
    for i, expected in enumerate(ANALYTICAL_S1):
        if expected == 0.0:
            assert abs(S1[i]) < 0.05, f"S1[{i}] = {S1[i]}, expected ~0"
        else:
            rel_err = abs(S1[i] - expected) / abs(expected)
            assert rel_err < 0.10, f"S1[{i}] = {S1[i]}, expected {expected}, rel_err={rel_err:.3f}"


def test_st_accuracy(ishigami_result):
    ST = np.array(ishigami_result.ST)
    for i, expected in enumerate(ANALYTICAL_ST):
        rel_err = abs(ST[i] - expected) / abs(expected)
        assert rel_err < 0.10, f"ST[{i}] = {ST[i]}, expected {expected}, rel_err={rel_err:.3f}"


def test_s2_accuracy(ishigami_result):
    S2 = np.array(ishigami_result.S2)
    # x1-x3 interaction
    expected = ANALYTICAL_S2[(0, 2)]
    rel_err = abs(S2[0, 2] - expected) / abs(expected)
    assert rel_err < 0.15, f"S2[0,2] = {S2[0, 2]}, expected {expected}, rel_err={rel_err:.3f}"
    # Other interactions should be ~0
    assert abs(S2[0, 1]) < 0.05, f"S2[0,1] = {S2[0, 1]}, expected ~0"
    assert abs(S2[1, 2]) < 0.05, f"S2[1,2] = {S2[1, 2]}, expected ~0"


@pytest.fixture(scope="module")
def ishigami_bootstrap_result():
    """Run Ishigami analysis with bootstrap CIs once for all tests."""
    sr = gsax.sample(PROBLEM, n_samples=2**14 * 8, seed=42)
    Y = evaluate(jnp.asarray(sr.samples))
    return gsax.analyze(sr, Y, num_resamples=200, key=jax.random.key(0))


def test_bootstrap_ci_contains_point_estimate(ishigami_bootstrap_result):
    """Point estimates should lie within their bootstrap CIs."""
    r = ishigami_bootstrap_result
    assert np.all(np.array(r.S1_conf[0]) <= np.array(r.S1))
    assert np.all(np.array(r.S1) <= np.array(r.S1_conf[1]))
    assert np.all(np.array(r.ST_conf[0]) <= np.array(r.ST))
    assert np.all(np.array(r.ST) <= np.array(r.ST_conf[1]))


def test_bootstrap_ci_contains_analytical(ishigami_bootstrap_result):
    """95% bootstrap CIs should contain the known analytical Ishigami values."""
    r = ishigami_bootstrap_result
    S1_lo, S1_hi = np.array(r.S1_conf[0]), np.array(r.S1_conf[1])
    ST_lo, ST_hi = np.array(r.ST_conf[0]), np.array(r.ST_conf[1])
    for i, expected in enumerate(ANALYTICAL_S1):
        assert S1_lo[i] <= expected <= S1_hi[i], (
            f"S1[{i}]: analytical {expected} not in CI [{S1_lo[i]}, {S1_hi[i]}]"
        )
    for i, expected in enumerate(ANALYTICAL_ST):
        assert ST_lo[i] <= expected <= ST_hi[i], (
            f"ST[{i}]: analytical {expected} not in CI [{ST_lo[i]}, {ST_hi[i]}]"
        )
