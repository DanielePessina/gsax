"""End-to-end VKOGA checks using fixed analytical and layout fixtures."""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest
from _linear_gaussian import (
    A_COEF,
    A_COEF_ASYM,
    ASYM_PROBLEM,
    GAUSS_PROBLEM,
    R_ASYM,
    R_GAUSS,
    analytic_indices,
)
from conftest import single_precision_warning

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa.problem import Problem

UNIFORM_PROBLEM = Problem(names=("u1", "u2"), bounds=((0.0, 1.0),) * 2)

SMALL_KWARGS = dict(
    gamma=3.0,
    ridge=1e-6,
    max_centers=64,
    n_outer=64,
    n_inner=16,
    n_variance=512,
    key=jax.random.key(0),
)

TINY_KWARGS = dict(
    gamma=3.0,
    ridge=1e-6,
    max_centers=32,
    n_outer=32,
    n_inner=8,
    n_variance=128,
    key=jax.random.key(0),
)


def _uniform_scalar(X: np.ndarray) -> np.ndarray:
    """A smooth model used for public layout and policy checks."""
    return X[:, 0] + X[:, 1] ** 2 + 0.5 * X[:, 0] * X[:, 1]


def _uniform_outputs(X: np.ndarray) -> dict[str, np.ndarray]:
    """The same model in scalar, multi-output, and time-series layouts."""
    y0 = _uniform_scalar(X)
    multi = np.stack([y0, 2.0 * y0, 1.0 - y0], axis=-1)
    return {"scalar": y0, "multi": np.stack([multi, multi + 0.5], axis=1)}


@pytest.fixture(scope="module")
def gauss_result():
    """VKOGA analysis of the correlated linear-Gaussian fixture."""
    with jax.enable_x64():
        X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 1536, seed=7)
        Y = X @ A_COEF
        return jaxgsa.vkoga.analyze(
            GAUSS_PROBLEM,
            X,
            Y,
            correlation=R_GAUSS,
            gamma=2.0,
            ridge=1e-10,
            max_centers=150,
            n_outer=256,
            n_inner=64,
            n_variance=4096,
            key=jax.random.key(0),
        )


@pytest.fixture(scope="module")
def asym_result():
    """VKOGA analysis of the asymmetric correlated fixture."""
    with jax.enable_x64():
        X = jaxgsa.sampling.monte_carlo(ASYM_PROBLEM, 2048, seed=7)
        Y = X @ A_COEF_ASYM
        return jaxgsa.vkoga.analyze(
            ASYM_PROBLEM,
            X,
            Y,
            correlation=R_ASYM,
            gamma=2.0,
            ridge=1e-10,
            max_centers=200,
            n_outer=256,
            n_inner=64,
            n_variance=4096,
            key=jax.random.key(0),
        )


@pytest.fixture(scope="module")
def uniform_fits():
    """Small fits used to exercise every public output layout once."""
    X = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 256, seed=11)
    fits = {}
    for label, Y in _uniform_outputs(X).items():
        with single_precision_warning():
            fits[label] = jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, **SMALL_KWARGS)
    return X, fits


def test_correlated_linear_fixture_matches_closed_form(gauss_result):
    s_tc, s_tu, variance = analytic_indices(A_COEF, R_GAUSS)
    np.testing.assert_allclose(gauss_result.S_TC, s_tc, atol=1e-2)
    np.testing.assert_allclose(gauss_result.S_TU, s_tu, atol=2e-2)
    np.testing.assert_allclose(float(gauss_result.variance), variance, rtol=5e-2)
    np.testing.assert_allclose(gauss_result.S_IU, 0.0, atol=3e-2)
    np.testing.assert_allclose(gauss_result.S_U, gauss_result.S_TU, atol=3e-2)
    assert gauss_result.is_correlated


def test_asymmetric_fixture_preserves_parameter_order(asym_result):
    s_tc, s_tu, variance = analytic_indices(A_COEF_ASYM, R_ASYM)
    np.testing.assert_allclose(asym_result.S_TC, s_tc, atol=2.5e-2)
    np.testing.assert_allclose(asym_result.S_TU, s_tu, atol=2.5e-2)
    np.testing.assert_allclose(float(asym_result.variance), variance, rtol=5e-2)
    assert list(np.argsort(asym_result.S_TC)) == list(np.argsort(s_tc))


def test_independent_linear_fixture_uses_classic_split():
    expected = A_COEF**2 / np.sum(A_COEF**2)
    with jax.enable_x64():
        X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 1536, seed=7)
        Y = X @ A_COEF
        result = jaxgsa.vkoga.analyze(
            GAUSS_PROBLEM,
            X,
            Y,
            gamma=2.0,
            ridge=1e-10,
            max_centers=150,
            n_outer=256,
            n_inner=64,
            n_variance=4096,
            key=jax.random.key(0),
        )
    assert not result.is_correlated
    np.testing.assert_allclose(result.S_TC, expected, atol=3e-2)
    np.testing.assert_allclose(result.S_C, 0.0, atol=3e-2)
    np.testing.assert_allclose(result.S_TU, result.S_TC, atol=3e-2)


def test_interaction_projection_clips_negative_interaction_parts():
    correlation = np.array([[1.0, -0.32, -0.12], [-0.32, 1.0, -0.75], [-0.12, -0.75, 1.0]])
    with jax.enable_x64():
        X = np.asarray(jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 1024, seed=5))
        Y = X @ np.array([1.0, 0.7, 0.4]) + 0.5 * X[:, 0] * X[:, 1] + 0.3 * np.tanh(X[:, 2])
        with pytest.warns(UserWarning, match="S_U exceeded S_TU"):
            result = jaxgsa.vkoga.analyze(
                GAUSS_PROBLEM,
                X,
                Y,
                correlation=correlation,
                gamma=2.0,
                ridge=1e-10,
                max_centers=150,
                n_outer=256,
                n_inner=64,
                n_variance=4096,
                key=jax.random.key(0),
            )
    assert np.all(np.asarray(result.S_U) <= np.asarray(result.S_TU) + 1e-12)
    assert np.all(np.asarray(result.S_IU) >= 0.0)


def test_failed_surrogate_warns_on_an_oscillatory_fixture():
    problem = Problem(names=("u1", "u2", "u3"), bounds=((0.0, 1.0),) * 3)
    X = jaxgsa.sampling.monte_carlo(problem, 512, seed=1)
    Y = np.sin(2.0 * np.pi * 12.0 * np.asarray(X)[:, 0]) + 0.5 * np.asarray(X)[:, 1]
    with pytest.warns(UserWarning, match="cross-validated surrogate error"):
        result = jaxgsa.vkoga.analyze(
            problem,
            X,
            Y,
            ridge=1e-6,
            max_centers=100,
            n_folds=4,
            n_outer=64,
            n_inner=16,
            n_variance=512,
            key=jax.random.key(0),
        )
    assert result.cv_rmse is not None and result.cv_rmse > 0.5 * float(Y.std())


def test_problem_correlation_is_default_but_override_wins():
    X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 256, seed=2)
    Y = X @ A_COEF
    declared = GAUSS_PROBLEM.with_correlation(R_GAUSS)
    with single_precision_warning():
        from_problem = jaxgsa.vkoga.analyze(declared, X, Y, **SMALL_KWARGS)
        override = jaxgsa.vkoga.analyze(declared, X, Y, correlation=np.eye(3), **SMALL_KWARGS)
    assert from_problem.is_correlated
    assert not override.is_correlated
    np.testing.assert_allclose(override.correlation, np.eye(3), atol=1e-12)


def test_output_layout_prediction_and_batching(uniform_fits):
    X, fits = uniform_fits
    expected = {"scalar": (2,), "multi": (2, 3, 2)}
    for label, result in fits.items():
        assert result.S_TC.shape == expected[label]
        assert result.S_TU.shape == expected[label]
    X_new = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 128, seed=5)
    pred = np.asarray(fits["scalar"].predict(X_new))
    np.testing.assert_allclose(
        fits["scalar"].predict(X_new, batch_size=37), pred, rtol=2e-5, atol=1e-6
    )
    assert pred.shape == (128,)


def test_dataset_carries_correlation_and_parameter_coordinates(gauss_result):
    dataset = gauss_result.to_dataset()
    assert list(dataset.coords["param"].values) == ["x1", "x2", "x3"]
    np.testing.assert_allclose(dataset["correlation"], R_GAUSS, atol=1e-12)
    assert dataset.attrs["is_correlated"] is True


def test_scalar_arguments_are_validated_before_fitting():
    X = jaxgsa.sampling.monte_carlo(GAUSS_PROBLEM, 32, seed=0)
    Y = X @ A_COEF
    cases = [
        ({"n_folds": 1}, "n_folds must be >= 2"),
        ({"n_outer": 1}, "n_outer must be >= 2"),
        ({"n_inner": 1}, "n_inner must be >= 2"),
        ({"n_variance": 1}, "n_variance must be >= 2"),
        ({"max_centers": 0}, "max_centers must be >= 1"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"key": None}, "key is required"),
        ({"gamma": -1.0}, "gamma must be a finite positive number"),
        ({"ridge": 0.0}, "ridge must be a finite positive number"),
    ]
    for overrides, match in cases:
        with pytest.raises(ValueError, match=match):
            jaxgsa.vkoga.analyze(GAUSS_PROBLEM, X, Y, **dict(SMALL_KWARGS, **overrides))


def test_analysis_is_deterministic_from_the_key():
    X = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 256, seed=11)
    Y = _uniform_scalar(X)
    correlation = np.array([[1.0, 0.4], [0.4, 1.0]])

    def run(seed):
        with single_precision_warning():
            return jaxgsa.vkoga.analyze(
                UNIFORM_PROBLEM,
                X,
                Y,
                correlation=correlation,
                **dict(SMALL_KWARGS, key=jax.random.key(seed)),
            )

    first, again = run(0), run(0)
    np.testing.assert_array_equal(first.S_TC, again.S_TC)
    other = run(1)
    assert not np.allclose(first.S_TC, other.S_TC, atol=1e-6)


def test_invalid_y_raise_and_drop_are_reported():
    X = np.random.default_rng(5).uniform(size=(256, 2))
    Y = _uniform_scalar(X)
    bad = Y.copy()
    bad[4] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, bad, **SMALL_KWARGS)
    with pytest.warns(JaxgsaWarning, match="dropped 1 of 256 rows"):
        result = jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, bad, on_invalid="drop", **SMALL_KWARGS)
    assert np.all(np.isfinite(np.asarray(result.S_TC)))
    assert result.invalid.unit_indices == (4,)


def test_invalid_x_names_the_source():
    X = np.random.default_rng(5).uniform(size=(256, 2))
    Y = _uniform_scalar(X)
    X[6, 0] = np.nan
    with pytest.raises(ValueError, match=r"in X\."):
        jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, **SMALL_KWARGS)


def test_bootstrap_metadata_bounds_and_key_requirement():
    X = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 128, seed=11)
    Y = _uniform_scalar(X)
    with pytest.raises(ValueError, match="key is required"):
        jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, n_bootstrap=4, **{**TINY_KWARGS, "key": None})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JaxgsaWarning)
        result = jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, n_bootstrap=8, **TINY_KWARGS)
    assert result.ci is not None
    assert result.ci.n_bootstrap == 8
    for name in ("S_TC", "S_TU", "S_U", "S_C", "S_IU"):
        conf = np.asarray(getattr(result, f"{name}_conf"))
        assert conf.shape[0] == 2
        assert np.all(conf[0] <= conf[1] + 1e-6)


def test_bootstrap_replicates_are_reproducible_and_exported():
    X = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 128, seed=11)
    Y = _uniform_scalar(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JaxgsaWarning)
        first = jaxgsa.vkoga.analyze(
            UNIFORM_PROBLEM, X, Y, n_bootstrap=4, keep_replicates=True, **TINY_KWARGS
        )
        second = jaxgsa.vkoga.analyze(
            UNIFORM_PROBLEM, X, Y, n_bootstrap=4, keep_replicates=True, **TINY_KWARGS
        )
    np.testing.assert_array_equal(first.ci.replicates["S_TC"], second.ci.replicates["S_TC"])
    assert first.ci.replicates["S_TC"].shape == (4, 2)
    dataset = first.to_dataset()
    assert "S_TC_lower" in dataset and "S_TC_upper" in dataset


def test_multi_output_bootstrap_keeps_layout():
    X = jaxgsa.sampling.monte_carlo(UNIFORM_PROBLEM, 128, seed=12)
    Y = _uniform_outputs(X)["multi"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", JaxgsaWarning)
        result = jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, n_bootstrap=4, **TINY_KWARGS)
    assert np.asarray(result.S_TC).shape == (2, 3, 2)
    assert np.asarray(result.S_TC_conf).shape == (2, 2, 3, 2)


def test_training_design_correlation_warning_is_actionable():
    rng = np.random.default_rng(3)
    latent = rng.standard_normal((256, 2))
    latent[:, 1] = 0.8 * latent[:, 0] + 0.6 * latent[:, 1]
    X = jax.scipy.special.ndtr(latent)
    Y = _uniform_scalar(np.asarray(X))
    correlation = np.array([[1.0, 0.7], [0.7, 1.0]])
    with pytest.warns(JaxgsaWarning, match="training X is itself correlated"):
        jaxgsa.vkoga.analyze(UNIFORM_PROBLEM, X, Y, correlation=correlation, **SMALL_KWARGS)
