"""End-to-end Borgonovo delta checks on analytical output fixtures."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import gaussian_linear, ishigami
from jaxgsa.borgonovo import analyze
from jaxgsa.problem import Problem
from jaxgsa.sampling import monte_carlo


@pytest.fixture(scope="module")
def ishigami_data():
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=2048, seed=2))
    return X, ishigami.evaluate(X)


@pytest.fixture(scope="module")
def gaussian_linear_data():
    X = jnp.asarray(monte_carlo(gaussian_linear.PROBLEM, n=4096, seed=3))
    return X, gaussian_linear.evaluate(X)


def test_gaussian_linear_matches_closed_form_delta_and_s1(gaussian_linear_data):
    X, Y = gaussian_linear_data
    result = analyze(gaussian_linear.PROBLEM, X, Y, n_bootstrap=0, verbose=False)
    np.testing.assert_allclose(result.delta, gaussian_linear.ANALYTICAL_DELTA, atol=0.04)
    np.testing.assert_allclose(result.S1, gaussian_linear.ANALYTICAL_S1, atol=0.04)


def test_ishigami_indices_are_bounded_and_ranked(ishigami_data):
    X, Y = ishigami_data
    result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0, verbose=False)
    assert np.all((np.asarray(result.delta) >= -0.05) & (np.asarray(result.delta) <= 1.05))
    assert result.delta[1] > result.delta[0] > result.delta[2]


def test_grid_tiling_does_not_change_the_estimate(ishigami_data):
    X, Y = ishigami_data
    full = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0, verbose=False)
    tiled = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0, slice_chunk_size=1, verbose=False)
    np.testing.assert_allclose(full.delta, tiled.delta, rtol=1e-5, atol=1e-7)


def test_bootstrap_is_reproducible_and_bracketed(ishigami_data):
    X, Y = ishigami_data
    kwargs = dict(n_bootstrap=8, key=jax.random.key(4), verbose=False)
    first = analyze(ishigami.PROBLEM, X, Y, **kwargs)
    second = analyze(ishigami.PROBLEM, X, Y, **kwargs)
    assert first.ci is not None
    np.testing.assert_array_equal(first.delta_conf, second.delta_conf)
    assert np.all(np.asarray(first.delta_conf[0]) <= np.asarray(first.delta_conf[1]))


def test_zero_variance_output_is_zero_with_a_warning(ishigami_data):
    X, _ = ishigami_data
    with pytest.warns(JaxgsaWarning, match="zero variance"):
        result = analyze(ishigami.PROBLEM, X, jnp.ones(X.shape[0]), n_bootstrap=0)
    np.testing.assert_array_equal(result.delta, 0.0)
    np.testing.assert_array_equal(result.S1, 0.0)


def test_discrete_output_is_rejected():
    problem = Problem(("x1", "x2"), ((0.0, 1.0),) * 2)
    X = jnp.asarray(monte_carlo(problem, n=256, seed=5))
    with pytest.raises(ValueError, match="discrete|continuous"):
        analyze(problem, X, (X[:, 0] > 0.5).astype(jnp.float32), n_bootstrap=0)


def test_invalid_policy_drop_returns_finite_indices(ishigami_data):
    X, Y = ishigami_data
    bad = Y.at[7].set(jnp.nan)
    with pytest.raises(ValueError, match="non-finite"):
        analyze(ishigami.PROBLEM, X, bad, n_bootstrap=0, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=JaxgsaWarning)
        result = analyze(ishigami.PROBLEM, X, bad, n_bootstrap=0, on_invalid="drop")
    assert np.all(np.isfinite(np.asarray(result.delta)))


def test_multi_output_and_time_series_layouts(ishigami_data):
    X, Y = ishigami_data
    multi = jnp.column_stack([Y, Y**2])
    series = jnp.stack([multi, multi + 1.0], axis=1)
    result = analyze(ishigami.PROBLEM, X, series, n_bootstrap=0, verbose=False)
    assert result.delta.shape == (2, 2, 3)


def test_xarray_export_has_named_axes(ishigami_data):
    X, Y = ishigami_data
    result = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=0, verbose=False)
    dataset = result.to_dataset()
    assert dataset["delta"].dims == ("param",)
    assert list(dataset.coords["param"].values) == list(ishigami.PROBLEM.names)
