import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gsax
from gsax.benchmarks.ishigami import ANALYTICAL_S1, ANALYTICAL_S2, ANALYTICAL_ST, PROBLEM, evaluate
from gsax.sampling import SamplingResult


@pytest.fixture(scope="module")
def ishigami_result():
    """Run Ishigami analysis once for all tests in this module."""
    sr = gsax.sample(PROBLEM, n_samples=2**14 * 8, seed=42, verbose=False)
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
    upper = np.triu_indices_from(S2, k=1)
    lower = (upper[1], upper[0])

    assert np.all(np.isnan(np.diag(S2))), f"S2 diagonal should be NaN, got {np.diag(S2)}"
    assert np.allclose(S2[upper], S2[lower]), "S2 should be symmetric off-diagonal"

    # x1-x3 interaction
    expected = ANALYTICAL_S2[(0, 2)]
    rel_err = abs(S2[0, 2] - expected) / abs(expected)
    assert rel_err < 0.15, f"S2[0,2] = {S2[0, 2]}, expected {expected}, rel_err={rel_err:.3f}"

    # Other upper-triangular interactions should be ~0
    assert abs(S2[0, 1]) < 0.05, f"S2[0,1] = {S2[0, 1]}, expected ~0"
    assert abs(S2[1, 2]) < 0.05, f"S2[1,2] = {S2[1, 2]}, expected ~0"


@pytest.fixture(scope="module")
def ishigami_bootstrap_result():
    """Run Ishigami analysis with bootstrap CIs once for all tests."""
    sr = gsax.sample(PROBLEM, n_samples=2**14 * 8, seed=42, verbose=False)
    Y = evaluate(jnp.asarray(sr.samples))
    return gsax.analyze(sr, Y, num_resamples=200, key=jax.random.key(0))


def test_bootstrap_ci_contains_point_estimate(ishigami_bootstrap_result):
    """Point estimates should lie within their bootstrap CIs."""
    r = ishigami_bootstrap_result
    assert np.all(np.array(r.S1_conf[0]) <= np.array(r.S1))
    assert np.all(np.array(r.S1) <= np.array(r.S1_conf[1]))
    assert np.all(np.array(r.ST_conf[0]) <= np.array(r.ST))
    assert np.all(np.array(r.ST) <= np.array(r.ST_conf[1]))

    S2 = np.array(r.S2)
    S2_lo = np.array(r.S2_conf[0])
    S2_hi = np.array(r.S2_conf[1])
    upper = np.triu_indices_from(S2, k=1)
    lower = (upper[1], upper[0])

    assert np.all(np.isnan(np.diag(S2_lo))), f"S2_conf lower diagonal should be NaN, got {np.diag(S2_lo)}"
    assert np.all(np.isnan(np.diag(S2_hi))), f"S2_conf upper diagonal should be NaN, got {np.diag(S2_hi)}"
    assert np.allclose(S2_lo[upper], S2_lo[lower]), "Lower S2_conf bound should be symmetric"
    assert np.allclose(S2_hi[upper], S2_hi[lower]), "Upper S2_conf bound should be symmetric"
    assert np.all(S2_lo[upper] <= S2[upper])
    assert np.all(S2[upper] <= S2_hi[upper])


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


def test_repeated_no_bootstrap_calls_identical():
    """Repeated identical no-bootstrap calls should return identical outputs."""
    sr = gsax.sample(PROBLEM, n_samples=2**10, seed=7)
    Y = evaluate(jnp.asarray(sr.samples))

    first = gsax.analyze(sr, Y)
    second = gsax.analyze(sr, Y)

    np.testing.assert_allclose(first.S1, second.S1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first.ST, second.ST, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first.S2, second.S2, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_repeated_bootstrap_calls_identical():
    """Repeated identical bootstrap calls should preserve point estimates and CI shapes."""
    sr = gsax.sample(PROBLEM, n_samples=2**10, seed=11)
    Y = evaluate(jnp.asarray(sr.samples))
    key = jax.random.key(123)

    first = gsax.analyze(sr, Y, num_resamples=20, key=key)
    second = gsax.analyze(sr, Y, num_resamples=20, key=key)

    np.testing.assert_allclose(first.S1, second.S1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first.ST, second.ST, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first.S2, second.S2, rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(first.S1_conf, second.S1_conf, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first.ST_conf, second.ST_conf, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(first.S2_conf, second.S2_conf, rtol=1e-6, atol=1e-6, equal_nan=True)


def _legacy_sampling_result(sr: SamplingResult) -> SamplingResult:
    expanded_n_total = sr.expanded_n_total
    expanded_samples = sr.samples[sr.expanded_to_unique]
    return SamplingResult(
        samples=expanded_samples,
        sample_ids=np.arange(expanded_n_total, dtype=np.int64),
        expanded_n_total=expanded_n_total,
        expanded_to_unique=np.arange(expanded_n_total, dtype=np.int64),
        base_n=sr.base_n,
        n_params=sr.n_params,
        calc_second_order=sr.calc_second_order,
        problem=sr.problem,
    )


def test_unique_analysis_matches_expanded_layout():
    sr = gsax.sample(PROBLEM, n_samples=1024, seed=7, verbose=False)
    Y_unique = evaluate(jnp.asarray(sr.samples))
    result_unique = gsax.analyze(sr, Y_unique)

    legacy_sr = _legacy_sampling_result(sr)
    Y_expanded = Y_unique[sr.expanded_to_unique]
    result_expanded = gsax.analyze(legacy_sr, Y_expanded)

    assert np.allclose(np.asarray(result_unique.S1), np.asarray(result_expanded.S1))
    assert np.allclose(np.asarray(result_unique.ST), np.asarray(result_expanded.ST))
    assert np.allclose(
        np.asarray(result_unique.S2),
        np.asarray(result_expanded.S2),
        equal_nan=True,
    )


def test_unique_bootstrap_matches_expanded_layout():
    sr = gsax.sample(PROBLEM, n_samples=1024, seed=7, verbose=False)
    Y_unique = evaluate(jnp.asarray(sr.samples))
    key = jax.random.key(123)
    result_unique = gsax.analyze(sr, Y_unique, num_resamples=50, key=key)

    legacy_sr = _legacy_sampling_result(sr)
    Y_expanded = Y_unique[sr.expanded_to_unique]
    result_expanded = gsax.analyze(legacy_sr, Y_expanded, num_resamples=50, key=key)

    assert np.allclose(np.asarray(result_unique.S1), np.asarray(result_expanded.S1))
    assert np.allclose(np.asarray(result_unique.ST), np.asarray(result_expanded.ST))
    assert np.allclose(
        np.asarray(result_unique.S2),
        np.asarray(result_expanded.S2),
        equal_nan=True,
    )
    assert np.allclose(np.asarray(result_unique.S1_conf), np.asarray(result_expanded.S1_conf))
    assert np.allclose(np.asarray(result_unique.ST_conf), np.asarray(result_expanded.ST_conf))
    assert np.allclose(
        np.asarray(result_unique.S2_conf),
        np.asarray(result_expanded.S2_conf),
        equal_nan=True,
    )
