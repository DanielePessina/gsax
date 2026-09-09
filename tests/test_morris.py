"""End-to-end Morris screening checks on known models."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import ishigami, linear, sobol_g
from jaxgsa.morris import analyze, indices, sample


def test_trajectory_design_and_linear_fixture_are_consistent():
    design = sample(linear.PROBLEM, n_trajectories=32, seed=2, verbose=False)
    result = analyze(design, linear.evaluate(jnp.asarray(design.samples)), verbose=False)
    assert design.samples.ndim == 2
    assert result.mu_star.shape == (3,)
    assert np.all(np.asarray(result.mu_star) > 0.0)
    np.testing.assert_allclose(result.mu, result.mu_star, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(result.sigma, 0.0, atol=1e-5)


def test_radial_design_runs_end_to_end():
    design = sample(ishigami.PROBLEM, n_trajectories=16, method="radial", seed=3, verbose=False)
    result = analyze(design, ishigami.evaluate(jnp.asarray(design.samples)), verbose=False)
    assert result.mu_star.shape == (3,)
    assert np.all(np.isfinite(np.asarray(result.mu_star)))


def test_ishigami_and_sobol_g_fixtures_rank_influential_inputs():
    ish_design = sample(ishigami.PROBLEM, n_trajectories=40, seed=4, verbose=False)
    ish = analyze(ish_design, ishigami.evaluate(jnp.asarray(ish_design.samples)), verbose=False)
    assert np.argmax(np.asarray(ish.mu_star)) in {0, 1}
    assert np.asarray(ish.sigma)[2] > 0.0

    g_design = sample(sobol_g.PROBLEM, n_trajectories=24, seed=5, verbose=False)
    g = analyze(g_design, sobol_g.evaluate(jnp.asarray(g_design.samples)), verbose=False)
    mu = np.asarray(g.mu_star)
    assert mu[0] > mu[1] > mu[2] > mu[3]
    assert np.all(mu[3] > mu[4:])


def test_multi_output_and_time_series_shapes():
    design = sample(linear.PROBLEM, n_trajectories=16, seed=6, verbose=False)
    y = linear.evaluate(jnp.asarray(design.samples))
    outputs = jnp.stack([y, 2.0 * y], axis=-1)
    series = jnp.stack([outputs, outputs + 1.0], axis=1)
    result = analyze(design, series, verbose=False)
    assert result.mu_star.shape == (2, 2, 3)
    assert result.sigma.shape == (2, 2, 3)


def test_bootstrap_is_reproducible_and_bracketed():
    design = sample(linear.PROBLEM, n_trajectories=20, seed=7, verbose=False)
    Y = linear.evaluate(jnp.asarray(design.samples))
    kwargs = dict(n_bootstrap=8, key=jax.random.key(9), verbose=False)
    first = analyze(design, Y, **kwargs)
    second = analyze(design, Y, **kwargs)
    assert first.ci is not None
    np.testing.assert_array_equal(first.mu_star_conf, second.mu_star_conf)
    assert np.all(np.asarray(first.mu_star_conf[0]) <= np.asarray(first.mu_star_conf[1]))


def test_physical_units_recover_linear_coefficients():
    design = sample(linear.PROBLEM, n_trajectories=20, seed=8, verbose=False)
    Y = linear.evaluate(jnp.asarray(design.samples))
    result = analyze(design, Y, verbose=False).to_physical_units()
    np.testing.assert_allclose(result.mu_star, np.array([1.0, 2.0, 3.0]), atol=0.05)


def test_invalid_policy_drop_removes_a_whole_trajectory():
    design = sample(linear.PROBLEM, n_trajectories=12, seed=9, verbose=False)
    Y = linear.evaluate(jnp.asarray(design.samples)).at[0].set(jnp.nan)
    with pytest.raises(ValueError, match="non-finite"):
        analyze(design, Y, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=JaxgsaWarning)
        result = analyze(design, Y, on_invalid="drop", verbose=False)
    assert result.invalid.n_invalid == 1
    assert np.all(np.isfinite(np.asarray(result.mu_star)))


def test_traceable_core_matches_the_public_result():
    design = sample(linear.PROBLEM, n_trajectories=12, seed=10, verbose=False)
    Y = linear.evaluate(jnp.asarray(design.samples))
    result = analyze(design, Y, verbose=False)
    core = indices(design, Y)
    np.testing.assert_allclose(result.mu_star, core[1], atol=1e-6)
