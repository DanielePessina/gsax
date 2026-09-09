"""Tests for the samplers: design layout, marginals, correlation, downsampling.

Tier T4 (internal consistency) except where noted: most tests pin design
invariants — uniqueness, nesting, determinism, bounds. The truncated-Gaussian
moment checks compare live against ``scipy.stats.truncnorm`` (Tier T2), and
the copula tests check recovered rank correlations against the declared
targets (T4: the target is our own input).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import truncnorm

import jaxgsa
from jaxgsa import JaxgsaWarning
from jaxgsa._core.sampling import _next_power_of_2
from jaxgsa.problem import GaussianInputSpec, InputSpecValue, Problem, UniformInputSpec
from jaxgsa.sampling import correlate, fit_correlation, monte_carlo
from jaxgsa.sobol import sample
from jaxgsa.sobol._sampling import _saltelli_step


def test_next_power_of_2():
    assert _next_power_of_2(1) == 1
    assert _next_power_of_2(2) == 2
    assert _next_power_of_2(3) == 4
    assert _next_power_of_2(5) == 8
    assert _next_power_of_2(1024) == 1024
    assert _next_power_of_2(1025) == 2048


def test_sobol_design_layout_is_unique_bounded_and_power_of_two():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)})
    result = sample(p, n_samples=100, seed=42, verbose=False)
    assert result.n_runs >= 100
    assert result.samples.shape == (result.n_runs, p.num_vars)
    assert np.unique(result.samples, axis=0).shape[0] == result.n_runs
    assert result.expanded_to_unique.shape == (result.n_expanded,)
    assert result.expanded_to_unique.max() < result.n_runs
    assert result.n_params == p.num_vars
    assert result.calc_second_order is True
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=100, seed=42, verbose=False)
    assert result.base_n & (result.base_n - 1) == 0
    with pytest.raises(ValueError, match=r"power of 2 .*got 1000.*nearest valid: 512 or 1024"):
        sample(p, n_samples=100, base_n=1000, seed=42, verbose=False)


def test_tiny_design_warns_about_a_degenerate_base():
    """A resolved base_n below the floor is degenerate, and says so.

    ``n_samples=4`` on a 3-D second-order design resolves to ``base_n=1``
    (``ceil(4 / 8)`` rounded up to a power of 2), which the doubling loop may
    raise later but which never carries a usable estimate on its own.
    """
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)})
    with pytest.warns(JaxgsaWarning, match="degenerate"):
        result = sample(p, n_samples=4, seed=0, verbose=False)
    assert result.base_n < 16


def test_single_parameter_mapping_collapses_duplicates():
    p = Problem.from_dict({"x1": (0.0, 1.0)})

    with pytest.warns(JaxgsaWarning, match="degenerate"):
        first_only = sample(p, n_samples=16, calc_second_order=False, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, False)
    for i in range(first_only.base_n):
        group = first_only.expanded_to_unique[i * step : (i + 1) * step]
        assert group[1] == group[2]
        assert group[0] != group[1]

    with pytest.warns(JaxgsaWarning, match="degenerate"):
        second_order = sample(p, n_samples=16, calc_second_order=True, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, True)
    for i in range(second_order.base_n):
        group = second_order.expanded_to_unique[i * step : (i + 1) * step]
        assert group[0] == group[2]
        assert group[1] == group[3]
        assert group[0] != group[1]


def test_two_parameter_second_order_mapping_collapses_cross_duplicates():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    with pytest.warns(JaxgsaWarning, match="degenerate"):
        result = sample(p, n_samples=32, calc_second_order=True, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, True)

    for i in range(result.base_n):
        group = result.expanded_to_unique[i * step : (i + 1) * step]
        assert group[1] == group[4]
        assert group[2] == group[3]
        assert len(set(group.tolist())) == 4


def test_mixed_marginals_respect_bounds_and_known_moments():
    p = Problem.from_dict(
        {
            "uniform": UniformInputSpec(dist="uniform", low=-3.0, high=2.0),
            "gaussian": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
        }
    )

    result = sample(p, n_samples=256, seed=1, verbose=False)
    assert np.all(result.samples[:, 0] >= -3.0)
    assert np.all(result.samples[:, 0] <= 2.0)
    p_gaussian = Problem.from_dict(
        {"x1": GaussianInputSpec(dist="gaussian", mean=1.5, variance=2.25), "x2": (0.0, 1.0)}
    )
    gaussian = sample(p_gaussian, n_samples=4096, calc_second_order=False, seed=123, verbose=False)
    assert abs(np.mean(gaussian.samples[:, 0]) - 1.5) < 0.05
    assert abs(np.var(gaussian.samples[:, 0]) - 2.25) < 0.08


def test_truncated_gaussian_columns_respect_one_sided_and_two_sided_bounds():
    p = Problem.from_dict(
        {
            "lower_only": GaussianInputSpec(
                dist="gaussian",
                mean=0.0,
                variance=1.0,
                low=-0.25,
            ),
            "upper_only": GaussianInputSpec(
                dist="gaussian",
                mean=0.0,
                variance=1.0,
                high=0.5,
            ),
            "two_sided": GaussianInputSpec(
                dist="gaussian",
                mean=0.0,
                variance=1.0,
                low=-1.0,
                high=1.0,
            ),
        }
    )

    result = sample(p, n_samples=512, calc_second_order=False, seed=7, verbose=False)
    assert np.all(result.samples[:, 0] >= -0.25)
    assert np.all(result.samples[:, 1] <= 0.5)
    assert np.all(result.samples[:, 2] >= -1.0)
    assert np.all(result.samples[:, 2] <= 1.0)


# ---------------------------------------------------------------------------
# Prefix downsampling tests
# ---------------------------------------------------------------------------


class TestSamplingResultDownsample:
    """Tests for SobolSamples.downsample()."""

    def _make_sr(self, D: int = 3, base_n: int = 32, second_order: bool = True, seed: int = 42):
        names: dict[str, InputSpecValue] = {f"x{i}": (0.0, 1.0) for i in range(D)}
        p = Problem.from_dict(names)
        return sample(
            p, n_samples=1, base_n=base_n, calc_second_order=second_order, seed=seed, verbose=False
        )

    def test_identity_and_prefix_properties(self):
        sr_identity = self._make_sr(base_n=16)
        assert sr_identity.downsample(16) is sr_identity
        sr_full = self._make_sr(base_n=64)
        sr_small = sr_full.downsample(16)
        assert np.array_equal(sr_small.samples, sr_full.samples[: sr_small.n_runs])

    def test_multiple_rungs_are_nested(self):
        sr_full = self._make_sr(base_n=64)
        sr_32 = sr_full.downsample(32)
        sr_16 = sr_full.downsample(16)
        sr_8 = sr_full.downsample(8)
        assert sr_8.n_runs <= sr_16.n_runs <= sr_32.n_runs <= sr_full.n_runs
        assert np.array_equal(sr_8.samples, sr_16.samples[: sr_8.n_runs])
        assert np.array_equal(sr_16.samples, sr_32.samples[: sr_16.n_runs])

    def test_upsample_raises(self):
        sr = self._make_sr(base_n=16)
        with pytest.raises(ValueError, match="Cannot upsample"):
            sr.downsample(32)

    def test_non_power_of_two_raises(self):
        sr = self._make_sr(base_n=16)
        with pytest.raises(ValueError, match=r"power of 2 .*nearest valid: 8 or 16"):
            sr.downsample(12)

    def test_with_Y_returns_tuple(self):
        sr_full = self._make_sr(base_n=32)
        Y = np.arange(sr_full.n_runs * 4, dtype=np.float64).reshape(sr_full.n_runs, 4)
        sr_small, Y_small = sr_full.downsample(8, Y)
        assert Y_small.shape == (sr_small.n_runs, 4)
        assert np.array_equal(Y_small, Y[: sr_small.n_runs])

    def test_with_Y_misaligned_raises(self):
        sr_full = self._make_sr(base_n=32)
        Y_wrong = np.zeros((sr_full.n_runs + 5, 3))
        with pytest.raises(ValueError, match="does not match n_runs"):
            sr_full.downsample(8, Y_wrong)

    def test_downsample_is_bit_identical_to_direct_draw(self):
        """Prefix property: downsampling to K equals drawing K base points directly.

        This backs the ``downsample`` docstring claim that the first K base
        points of a draw with N > K base points are bit-identical to drawing
        K base points with the same seed and scramble.
        """
        p = Problem.from_dict(
            {
                "uniform": UniformInputSpec(dist="uniform", low=-2.0, high=3.0),
                "gaussian": GaussianInputSpec(
                    dist="gaussian", mean=1.0, variance=4.0, low=-1.0, high=4.0
                ),
            }
        )
        N, K, seed = 64, 16, 1234

        sr_small = sample(p, n_samples=1, base_n=N, seed=seed, verbose=False).downsample(K)
        sr_direct = sample(p, n_samples=1, base_n=K, seed=seed, verbose=False)

        np.testing.assert_array_equal(sr_small.samples, sr_direct.samples)
        np.testing.assert_array_equal(sr_small.expanded_to_unique, sr_direct.expanded_to_unique)
        assert sr_small.n_expanded == sr_direct.n_expanded
        assert sr_small.base_n == sr_direct.base_n == K


# ---------------------------------------------------------------------------
# Correlated sampling (Gaussian copula on Problem.correlation)
# ---------------------------------------------------------------------------


def _spearman_of(X: np.ndarray) -> np.ndarray:
    """Sample Spearman rank-correlation matrix of the columns of X."""
    ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float64)
    return np.corrcoef(ranks, rowvar=False)


def test_monte_carlo_gaussian_marginals_reproduce_mvn_covariance():
    """All-Gaussian marginals + latent R: the copula *is* the MVN N(mu, DRD)."""
    rho = 0.6
    sigma = np.array([1.5, 2.0])
    problem = Problem.from_dict(
        {
            "a": GaussianInputSpec(dist="gaussian", mean=1.0, variance=float(sigma[0] ** 2)),
            "b": GaussianInputSpec(dist="gaussian", mean=-2.0, variance=float(sigma[1] ** 2)),
        },
        correlation=[[1.0, rho], [rho, 1.0]],
    )
    X = monte_carlo(problem, 100_000, seed=42)
    expected = sigma[:, None] * np.array([[1.0, rho], [rho, 1.0]]) * sigma[None, :]
    np.testing.assert_allclose(np.cov(X, rowvar=False), expected, atol=0.08)
    np.testing.assert_allclose(np.mean(X, axis=0), [1.0, -2.0], atol=0.03)


def test_spearman_kind_recovers_rank_correlation_with_uniform_marginals():
    rho_s = 0.7
    R = [[1.0, rho_s], [rho_s, 1.0]]
    spearman_problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=R, correlation_type="spearman"
    )
    X = monte_carlo(spearman_problem, 100_000, seed=3)
    assert abs(_spearman_of(X)[0, 1] - rho_s) < 0.01

    # The latent kind would *not* hit the rank target: a latent 0.7 yields
    # a Spearman correlation of (6/pi) asin(0.7/2) ~ 0.683, which is what
    # justifies exposing the kind switch at all.
    latent_problem = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=R)
    X_latent = monte_carlo(latent_problem, 100_000, seed=3)
    expected_spearman = (6.0 / np.pi) * np.arcsin(rho_s / 2.0)
    assert abs(_spearman_of(X_latent)[0, 1] - expected_spearman) < 0.01
    assert abs(_spearman_of(X_latent)[0, 1] - rho_s) > 0.01


def test_correlated_sampling_preserves_marginals():
    """Coupling must not disturb the declared marginals (moments + bounds)."""
    R = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ]
    )
    problem = Problem.from_dict(
        {
            "uniform": UniformInputSpec(dist="uniform", low=-3.0, high=2.0),
            "gaussian": GaussianInputSpec(dist="gaussian", mean=1.5, variance=2.25),
            "two_sided": GaussianInputSpec(
                dist="gaussian", mean=0.5, variance=1.44, low=-0.5, high=1.5
            ),
        },
        correlation=R,
    )
    X = monte_carlo(problem, 16_384, seed=123)

    assert np.all(X[:, 0] >= -3.0)
    assert np.all(X[:, 0] <= 2.0)
    assert abs(np.mean(X[:, 1]) - 1.5) < 0.05
    assert abs(np.var(X[:, 1]) - 2.25) < 0.08
    assert np.all(X[:, 2] >= -0.5)
    assert np.all(X[:, 2] <= 1.5)
    std = np.sqrt(1.44)
    a = (-0.5 - 0.5) / std
    b = (1.5 - 0.5) / std
    assert abs(np.var(X[:, 2]) - truncnorm.var(a, b, loc=0.5, scale=std)) < 0.03


def test_correlated_monte_carlo_determinism_and_generator_seed():
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, 0.8], [0.8, 1.0]]
    )
    np.testing.assert_array_equal(
        monte_carlo(problem, 128, seed=5), monte_carlo(problem, 128, seed=5)
    )
    assert not np.array_equal(monte_carlo(problem, 128, seed=5), monte_carlo(problem, 128, seed=6))
    from_generator = monte_carlo(problem, 128, seed=np.random.default_rng(5))
    np.testing.assert_array_equal(from_generator, monte_carlo(problem, 128, seed=5))


def test_correlate_is_a_per_column_permutation_hitting_the_target():
    rho = 0.8
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, rho], [rho, 1.0]]
    )
    X = monte_carlo(problem.with_correlation(None), 8192, seed=21)
    X_corr = correlate(X, problem, seed=22)

    # Exact permutation of each column: sample values are fully preserved.
    np.testing.assert_array_equal(np.sort(X_corr, axis=0), np.sort(X, axis=0))
    # And the re-pairing achieves the declared latent correlation.
    assert abs(fit_correlation(problem, X_corr)[0, 1] - rho) < 0.05


def test_correlate_determinism_and_validation():
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, 0.5], [0.5, 1.0]]
    )
    X = monte_carlo(problem.with_correlation(None), 256, seed=1)
    np.testing.assert_array_equal(correlate(X, problem, seed=2), correlate(X, problem, seed=2))

    with pytest.raises(ValueError, match="requires problem.correlation"):
        correlate(X, problem.with_correlation(None))
    with pytest.raises(ValueError, match=r"X must be \(N, 2\)"):
        correlate(X[:, :1], problem)


def test_correlate_rejects_non_finite_X():
    """Tier T4 (input-contract): NaN rows must be rejected, not silently paired.

    np.sort puts NaN last, so every NaN row would deterministically take the
    highest van der Waerden scores and bias the correlation of the finite
    rows even if the NaNs were dropped afterwards.
    """
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, 0.5], [0.5, 1.0]]
    )
    X = monte_carlo(problem.with_correlation(None), 64, seed=5)
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        correlate(X, problem, seed=5)
    X[3, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        correlate(X, problem, seed=5)


def test_correlate_handles_degenerate_row_counts():
    """Too few rows to estimate corr(M): warn and fall back, do not divide by zero.

    ``n <= D`` or ``n < 3`` skips the de-correlation step, so those designs
    warn that the achieved pairing does not follow the declared matrix.
    """
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, 0.5], [0.5, 1.0]]
    )
    for n in (1, 2):
        X = monte_carlo(problem.with_correlation(None), n, seed=n)
        with pytest.warns(JaxgsaWarning, match="does not follow the declared correlation"):
            out = correlate(X, problem, seed=n)
        assert np.isfinite(out).all()
        np.testing.assert_array_equal(np.sort(out, axis=0), np.sort(X, axis=0))
    # n=3 with D=2 can estimate corr(M): the de-correlation runs, silently.
    X = monte_carlo(problem.with_correlation(None), 3, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", JaxgsaWarning)
        out = correlate(X, problem, seed=3)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(np.sort(out, axis=0), np.sort(X, axis=0))


def test_correlated_end_to_end_ot_borgonovo_hdmr():
    """Y = X1 with corr(X1, X2) = 0.8: the unused X2 must earn a clearly
    non-zero index under the correlation-inclusive given-data methods, and
    HDMR's ANCOVA Sb term must flag the correlation-induced variance."""
    rho = 0.8
    problem = Problem.from_dict(
        {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}, correlation=[[1.0, rho], [rho, 1.0]]
    )
    X = monte_carlo(problem, 2048, seed=99)
    Y = jnp.asarray(X[:, 0])
    Xj = jnp.asarray(X)

    ot = jaxgsa.optimal_transport.analyze(problem, Xj, Y)
    assert float(ot.ot[0]) > 0.5  # Y is fully determined by X1
    assert float(ot.ot[1]) > 0.1  # X2 unused, but correlated with X1

    delta = jaxgsa.borgonovo.analyze(problem, Xj, Y, key=jax.random.key(0))
    assert float(delta.delta[0]) > 0.5
    assert float(delta.delta[1]) > 0.1

    # For Y = X1 backfitting attributes everything to the x1 component, so
    # the correlative share is clearest on an additive model of both inputs:
    # each first-order term covaries with the other through corr(X1, X2).
    Y_sum = jnp.asarray(X[:, 0] + X[:, 1])
    hdmr = jaxgsa.hdmr.analyze(problem, Xj, Y_sum)
    assert float(jnp.max(jnp.abs(hdmr.Sb))) > 0.1

    # Same model on an independent draw: the correlative share collapses.
    independent = problem.with_correlation(None)
    X0 = monte_carlo(independent, 2048, seed=98)
    hdmr_indep = jaxgsa.hdmr.analyze(
        independent, jnp.asarray(X0), jnp.asarray(X0[:, 0] + X0[:, 1])
    )
    assert float(jnp.max(jnp.abs(hdmr_indep.Sb))) < 0.05
