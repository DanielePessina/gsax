"""End-to-end HSIC checks on deterministic dependence fixtures."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import ishigami, linear
from jaxgsa.hsic import analyze, indices
from jaxgsa.problem import Problem
from jaxgsa.sampling import monte_carlo


def _linear_data(n: int = 512):
    X = jnp.asarray(monte_carlo(linear.PROBLEM, n=n, seed=4))
    return X, linear.evaluate(X)


def _ishigami_data(n: int = 512):
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=n, seed=5))
    return X, ishigami.evaluate(X)


def test_linear_fixture_has_bounded_ranked_dependence():
    with jax.enable_x64():
        X, Y = _linear_data()
        result = analyze(linear.PROBLEM, X, Y, n_perms=40, key=jax.random.key(0), verbose=False)
    assert np.all((np.asarray(result.R2_HSIC) >= 0.0) & (np.asarray(result.R2_HSIC) <= 1.0))
    assert np.all(np.asarray(result.p_values) <= 0.05)
    assert result.R2_HSIC[2] > result.R2_HSIC[1] > result.R2_HSIC[0]


def test_ishigami_fixture_detects_nonlinear_influence():
    with jax.enable_x64():
        X, Y = _ishigami_data()
        result = analyze(ishigami.PROBLEM, X, Y, n_perms=40, key=jax.random.key(1), verbose=False)
    assert result.R2_HSIC[0] > result.R2_HSIC[2]
    assert np.all(np.isfinite(np.asarray(result.T_HSIC)))


def test_multi_output_matches_the_scalar_slice():
    with jax.enable_x64():
        X, Y = _linear_data()
        multi = jnp.column_stack([Y, Y**2])
        scalar = analyze(linear.PROBLEM, X, Y, n_perms=20, key=jax.random.key(2), verbose=False)
        result = analyze(
            linear.PROBLEM, X, multi, n_perms=20, key=jax.random.key(2), verbose=False
        )
    assert result.R2_HSIC.shape == (2, 3)
    np.testing.assert_allclose(result.R2_HSIC[0], scalar.R2_HSIC, atol=1e-8)


def test_bandwidth_is_a_real_analysis_setting():
    with jax.enable_x64():
        X, Y = _linear_data(256)
        narrow = analyze(linear.PROBLEM, X, Y, bandwidth=0.25, n_perms=10, key=jax.random.key(3))
        wide = analyze(linear.PROBLEM, X, Y, bandwidth=2.0, n_perms=10, key=jax.random.key(3))
    assert not np.allclose(narrow.R2_HSIC, wide.R2_HSIC)


def test_affine_output_rescaling_does_not_change_hsic():
    with jax.enable_x64():
        X, Y = _linear_data(256)
        base = analyze(linear.PROBLEM, X, Y, n_perms=10, key=jax.random.key(4))
        scaled = analyze(linear.PROBLEM, X, 7.0 * Y + 3.0, n_perms=10, key=jax.random.key(4))
    np.testing.assert_allclose(base.R2_HSIC, scaled.R2_HSIC, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(base.T_HSIC, scaled.T_HSIC, rtol=1e-10, atol=1e-12)


def test_zero_variance_output_is_nan_with_a_warning():
    X, _ = _linear_data(128)
    with pytest.warns(JaxgsaWarning, match="zero variance"):
        result = analyze(linear.PROBLEM, X, jnp.ones(X.shape[0]), n_perms=5, key=jax.random.key(5))
    assert np.all(np.isnan(np.asarray(result.R2_HSIC)))


def test_missing_key_is_rejected():
    X, Y = _linear_data(128)
    with pytest.raises(ValueError, match="key"):
        analyze(linear.PROBLEM, X, Y, n_perms=5, verbose=False)


def test_correlated_inputs_are_accepted_but_categorical_inputs_are_refused():
    X, Y = _linear_data(128)
    correlation = np.eye(3)
    correlation[0, 1] = correlation[1, 0] = 0.5
    correlated = linear.PROBLEM.with_correlation(correlation)
    with jax.enable_x64():
        result = analyze(correlated, X, Y, n_perms=5, key=jax.random.key(6), verbose=False)
    assert result.R2_HSIC.shape == (3,)

    categorical = Problem.from_dict(
        {
            "a": {"dist": "categorical", "probs": [0.5, 0.5]},
            "b": (0.0, 1.0),
        }
    )
    X_cat = jnp.asarray(monte_carlo(categorical, n=128, seed=7))
    with pytest.raises(ValueError, match="categorical"):
        analyze(categorical, X_cat, X_cat[:, 0], n_perms=5, key=jax.random.key(7))


def test_traceable_core_matches_the_public_result():
    with jax.enable_x64():
        X, Y = _linear_data(128)
        result = analyze(linear.PROBLEM, X, Y, n_perms=5, key=jax.random.key(8), verbose=False)
        core = indices(linear.PROBLEM, X, Y, n_perms=5, key=jax.random.key(8))
    np.testing.assert_allclose(result.R2_HSIC, core[0], atol=1e-8)
    np.testing.assert_allclose(result.T_HSIC, core[1], atol=1e-8)
