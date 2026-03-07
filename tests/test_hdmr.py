"""Tests for RS-HDMR sensitivity analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gsax.analyze_hdmr import analyze_hdmr, emulate_hdmr
from gsax.benchmarks.ishigami import (
    ANALYTICAL_S1,
    ANALYTICAL_ST,
    PROBLEM,
    evaluate,
)


@pytest.fixture(scope="module")
def ishigami_data():
    """Generate random Ishigami (X, Y) data for HDMR tests."""
    key = jax.random.PRNGKey(42)
    N = 2000
    bounds = jnp.array(PROBLEM.bounds)
    X = jax.random.uniform(key, shape=(N, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
    Y = evaluate(X)
    return X, Y


@pytest.fixture(scope="module")
def hdmr_result(ishigami_data):
    """Run HDMR analysis once for all tests."""
    X, Y = ishigami_data
    return analyze_hdmr(
        PROBLEM, X, Y,
        maxorder=2,
        m=2,
    )


# ---------------------------------------------------------------------------
# Accuracy tests
# ---------------------------------------------------------------------------

def test_st_accuracy(hdmr_result):
    """Total-order indices should approximate analytical values."""
    ST = np.array(hdmr_result.ST)
    for i, expected in enumerate(ANALYTICAL_ST):
        rel_err = abs(ST[i] - expected) / max(abs(expected), 0.01)
        assert rel_err < 0.25, (
            f"ST[{i}] = {ST[i]:.4f}, expected {expected}, rel_err={rel_err:.3f}"
        )


def test_s1_via_sa(hdmr_result):
    """First-order Sa terms should approximate analytical S1."""
    Sa = np.array(hdmr_result.Sa)
    D = PROBLEM.num_vars
    for i in range(D):
        expected = ANALYTICAL_S1[i]
        if expected == 0.0:
            assert abs(Sa[i]) < 0.1, f"Sa[{i}] = {Sa[i]}, expected ~0"
        else:
            rel_err = abs(Sa[i] - expected) / abs(expected)
            assert rel_err < 0.30, (
                f"Sa[{i}] = {Sa[i]:.4f}, expected {expected}, rel_err={rel_err:.3f}"
            )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_shapes_1d(ishigami_data):
    """Y shape (N,) -> index shapes (n_terms,) and (D,)."""
    X, Y = ishigami_data
    result = analyze_hdmr(
        PROBLEM, X, Y, maxorder=2, m=2,
    )
    D = PROBLEM.num_vars
    n_terms = D + D * (D - 1) // 2  # D + C(D,2)
    assert result.Sa.shape == (n_terms,)
    assert result.Sb.shape == (n_terms,)
    assert result.S.shape == (n_terms,)
    assert result.ST.shape == (D,)


def test_shapes_2d(ishigami_data):
    """Y shape (N, K) -> index shapes (K, n_terms) and (K, D)."""
    X, Y = ishigami_data
    K = 2
    Y_2d = jnp.stack([Y, Y * 0.5], axis=1)  # (N, 2)
    result = analyze_hdmr(
        PROBLEM, X, Y_2d, maxorder=2, m=2,
    )
    D = PROBLEM.num_vars
    n_terms = D + D * (D - 1) // 2
    assert result.Sa.shape == (K, n_terms)
    assert result.ST.shape == (K, D)


def test_shapes_3d(ishigami_data):
    """Y shape (N, T, K) -> index shapes (T, K, n_terms) and (T, K, D)."""
    X, Y = ishigami_data
    T, K = 2, 3
    Y_3d = jnp.broadcast_to(Y[:, None, None], (Y.shape[0], T, K))
    result = analyze_hdmr(
        PROBLEM, X, Y_3d, maxorder=2, m=2,
    )
    D = PROBLEM.num_vars
    n_terms = D + D * (D - 1) // 2
    assert result.Sa.shape == (T, K, n_terms)
    assert result.ST.shape == (T, K, D)
    assert result.rmse.shape == (T * K,)


def test_chunk_size_regression(ishigami_data):
    """Chunked and unchunked HDMR paths should agree for multi-output Y."""
    X, Y = ishigami_data
    Y_2d = jnp.stack([Y, Y * 0.5], axis=1)

    result_default = analyze_hdmr(PROBLEM, X, Y_2d, maxorder=2, m=2)
    result_chunked = analyze_hdmr(
        PROBLEM, X, Y_2d, maxorder=2, m=2, chunk_size=1,
    )

    np.testing.assert_allclose(result_default.Sa, result_chunked.Sa, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result_default.Sb, result_chunked.Sb, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result_default.S, result_chunked.S, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result_default.ST, result_chunked.ST, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result_default.rmse, result_chunked.rmse, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# maxorder tests
# ---------------------------------------------------------------------------

def test_maxorder_1(ishigami_data):
    """maxorder=1 should produce D terms."""
    X, Y = ishigami_data
    result = analyze_hdmr(
        PROBLEM, X, Y, maxorder=1, m=2,
    )
    assert result.Sa.shape == (PROBLEM.num_vars,)
    assert len(result.terms) == PROBLEM.num_vars


def test_maxorder_2(ishigami_data):
    """maxorder=2 should produce D + C(D,2) terms."""
    X, Y = ishigami_data
    D = PROBLEM.num_vars
    result = analyze_hdmr(
        PROBLEM, X, Y, maxorder=2, m=2,
    )
    expected_n = D + D * (D - 1) // 2
    assert result.Sa.shape == (expected_n,)
    assert len(result.terms) == expected_n


def test_maxorder_3(ishigami_data):
    """maxorder=3 should produce D + C(D,2) + C(D,3) terms."""
    X, Y = ishigami_data
    D = PROBLEM.num_vars
    result = analyze_hdmr(
        PROBLEM, X, Y, maxorder=3, m=2,
    )
    expected_n = D + D * (D - 1) // 2 + D * (D - 1) * (D - 2) // 6
    assert result.Sa.shape == (expected_n,)
    assert len(result.terms) == expected_n


# ---------------------------------------------------------------------------
# Emulator tests
# ---------------------------------------------------------------------------

def test_emulator_prediction(hdmr_result, ishigami_data):
    """Emulator predictions on training data should have low RMSE."""
    X, Y = ishigami_data
    Y_pred = emulate_hdmr(hdmr_result, X)
    assert Y_pred.shape == Y.shape
    rmse = float(jnp.sqrt(jnp.mean(jnp.square(Y - Y_pred))))
    # HDMR surrogate should capture most of the variance
    assert rmse < 1.5, f"Emulator RMSE = {rmse:.3f}, expected < 1.5"


def test_emulator_reasonable(hdmr_result, ishigami_data):
    """Emulator output mean should be close to data mean."""
    X, Y = ishigami_data
    Y_pred = emulate_hdmr(hdmr_result, X)
    assert abs(float(jnp.mean(Y_pred)) - float(jnp.mean(Y))) < 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_term_labels(hdmr_result):
    """Term labels should contain parameter names and interaction terms."""
    terms = hdmr_result.terms
    names = PROBLEM.names
    # First-order terms should be parameter names
    for name in names:
        assert name in terms
    # Should have interaction terms
    assert any("/" in t for t in terms)


def test_select_and_rmse(hdmr_result):
    """select and rmse should be present."""
    assert hdmr_result.select is not None
    assert hdmr_result.rmse is not None
    assert hdmr_result.select.shape[0] > 0


def test_s1_property(hdmr_result):
    """S1 property should extract first D terms of Sa."""
    D = PROBLEM.num_vars
    S1 = hdmr_result.S1
    assert S1.shape == (D,)
    np.testing.assert_array_equal(S1, hdmr_result.Sa[:D])


def test_s1_accuracy(hdmr_result):
    """S1 property should approximate analytical first-order Sobol indices."""
    S1 = np.array(hdmr_result.S1)
    for i, expected in enumerate(ANALYTICAL_S1):
        if expected == 0.0:
            assert abs(S1[i]) < 0.1, f"S1[{i}] = {S1[i]}, expected ~0"
        else:
            rel_err = abs(S1[i] - expected) / abs(expected)
            assert rel_err < 0.30, (
                f"S1[{i}] = {S1[i]:.4f}, expected {expected}, rel_err={rel_err:.3f}"
            )


def test_constant_y():
    """Constant Y should not produce NaN indices."""
    N = 500
    D = PROBLEM.num_vars
    key = jax.random.PRNGKey(99)
    bounds = jnp.array(PROBLEM.bounds)
    X = jax.random.uniform(key, shape=(N, D), minval=bounds[:, 0], maxval=bounds[:, 1])
    Y = jnp.ones(N) * 42.0  # constant output
    result = analyze_hdmr(PROBLEM, X, Y, maxorder=2, m=2)
    assert not jnp.any(jnp.isnan(result.Sa))
    assert not jnp.any(jnp.isnan(result.ST))


def test_validation_errors():
    """Input validation should raise on bad inputs."""
    X = jnp.ones((100, 3))
    Y = jnp.ones(100)

    with pytest.raises(ValueError, match="at least 300"):
        analyze_hdmr(PROBLEM, X, Y)

    X = jnp.ones((500, 3))
    Y = jnp.ones(500)

    with pytest.raises(ValueError, match="maxorder"):
        analyze_hdmr(PROBLEM, X, Y, maxorder=4)

    with pytest.raises(ValueError, match="chunk_size"):
        analyze_hdmr(PROBLEM, X, Y, chunk_size=0)
