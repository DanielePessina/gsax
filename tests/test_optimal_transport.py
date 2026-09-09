"""End-to-end optimal-transport checks against analytical fixtures."""

from __future__ import annotations

import warnings
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.benchmarks import gaussian_linear, ishigami
from jaxgsa.optimal_transport import analyze
from jaxgsa.optimal_transport._result import OTResult
from jaxgsa.sampling import monte_carlo


def _regular(result: Any) -> OTResult:
    """Narrow a rectangular-output result for type checking."""
    assert isinstance(result, OTResult)
    return result


@pytest.fixture(scope="module")
def ishigami_data():
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=2048, seed=42))
    return X, ishigami.evaluate(X)


@pytest.fixture(scope="module")
def gaussian_linear_data():
    X = jnp.asarray(monte_carlo(gaussian_linear.PROBLEM, n=8192, seed=1))
    return X, gaussian_linear.evaluate(X)


def test_univariate_ishigami_is_bounded_and_decomposes(ishigami_data):
    X, Y = ishigami_data
    result = _regular(analyze(ishigami.PROBLEM, X, Y, n_partitions=8, verbose=False))
    for field in (result.ot, result.advective, result.diffusive):
        assert np.all((np.asarray(field) >= 0.0) & (np.asarray(field) <= 1.0))
    np.testing.assert_allclose(result.ot, result.advective + result.diffusive, atol=1e-6)
    assert result.ot[0] > result.ot[2]
    assert result.ot[1] > result.ot[2]


def test_gaussian_linear_matches_closed_form_fixture(gaussian_linear_data):
    X, Y = gaussian_linear_data
    result = _regular(analyze(gaussian_linear.PROBLEM, X, Y, n_partitions=16, verbose=False))
    np.testing.assert_allclose(result.ot, gaussian_linear.ANALYTICAL_OT, atol=0.04)
    np.testing.assert_allclose(
        2.0 * np.asarray(result.advective), gaussian_linear.ANALYTICAL_S1, atol=0.04
    )


def test_joint_modes_return_their_documented_layouts(ishigami_data):
    X, Y = ishigami_data
    Y2 = jnp.stack([Y, Y**2], axis=-1)
    Y3 = jnp.stack([Y2, Y2 + 1.0], axis=1)
    multivariate = _regular(
        analyze(
            ishigami.PROBLEM, X[:512], Y2[:512], mode="multivariate", n_partitions=4, verbose=False
        )
    )
    trajectory = _regular(
        analyze(
            ishigami.PROBLEM, X[:512], Y3[:512], mode="trajectory", n_partitions=4, verbose=False
        )
    )
    assert multivariate.ot.shape == (3,)
    assert trajectory.ot.shape == (2, 3)
    np.testing.assert_allclose(
        np.asarray(multivariate.ot),
        np.asarray(multivariate.advective + multivariate.diffusive),
        atol=1e-5,
    )


def test_bootstrap_is_reproducible_and_dummy_exposes_a_floor(ishigami_data):
    X, Y = ishigami_data
    first = _regular(
        analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_partitions=6,
            n_bootstrap=4,
            dummy=True,
            key=jax.random.key(7),
            verbose=False,
        )
    )
    second = _regular(
        analyze(
            ishigami.PROBLEM,
            X,
            Y,
            n_partitions=6,
            n_bootstrap=4,
            dummy=True,
            key=jax.random.key(7),
            verbose=False,
        )
    )
    np.testing.assert_array_equal(first.ot, second.ot)
    np.testing.assert_array_equal(first.ot_conf, second.ot_conf)
    assert first.ot_dummy is not None
    assert first.above_dummy is not None


def test_correlated_observations_keep_the_correlation_inclusive_reading():
    rho = 0.8
    correlation = np.array([[1.0, rho], [rho, 1.0]])
    problem = jaxgsa.Problem.from_dict(
        {
            "x1": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
            "x2": {"dist": "gaussian", "mean": 0.0, "variance": 1.0},
        },
        correlation=correlation,
    )
    X = jnp.asarray(monte_carlo(problem, n=2048, seed=0))
    result = _regular(analyze(problem, X, X[:, 0], n_partitions=8, verbose=False))
    assert result.ot[0] > result.ot[1] > 0.05


def test_invalid_policy_drop_removes_bad_rows():
    problem = jaxgsa.Problem(("a", "b"), ((0.0, 1.0),) * 2)
    X = jnp.asarray(monte_carlo(problem, n=128, seed=3))
    Y = jnp.sin(3.0 * X[:, 0]) + 0.3 * X[:, 1]
    bad = Y.at[5].set(jnp.nan)
    with pytest.raises(ValueError, match="non-finite"):
        analyze(problem, X, bad, n_partitions=4, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=jaxgsa.JaxgsaWarning)
        result = _regular(
            analyze(problem, X, bad, n_partitions=4, on_invalid="drop", verbose=False)
        )
    assert np.all(np.isfinite(np.asarray(result.ot)))


def test_to_dataset_preserves_time_coordinates(ishigami_data):
    X, Y = ishigami_data
    times = np.array([0.0, 0.5, 1.0])
    Y3 = jnp.stack(
        [Y[:, None] * (1.0 + times[None, :]), jnp.broadcast_to(Y[:, None], (Y.shape[0], 3))],
        axis=-1,
    )
    result = _regular(analyze(ishigami.PROBLEM, X, Y3, verbose=False))
    dataset = result.to_dataset(time_coords=times)
    assert dataset["ot"].dims == ("time", "output", "param")
    np.testing.assert_array_equal(dataset["time"].values, times)


def test_bad_mode_is_rejected(ishigami_data):
    X, Y = ishigami_data
    with pytest.raises(ValueError, match="mode"):
        cast(Callable[..., Any], analyze)(ishigami.PROBLEM, X, Y, mode="unknown", verbose=False)
