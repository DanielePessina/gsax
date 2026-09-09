"""Tests for PAWN sensitivity analysis."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgsa import JaxgsaWarning
from jaxgsa.benchmarks import ishigami
from jaxgsa.pawn import analyze, indices
from jaxgsa.problem import InputSpecValue, Problem
from jaxgsa.sampling import monte_carlo


@pytest.fixture(scope="module")
def ishigami_data():
    """Generate Ishigami test data."""
    N = 5000
    X = jnp.asarray(monte_carlo(ishigami.PROBLEM, n=N, seed=42))
    Y = ishigami.evaluate(X)
    return X, Y


class TestPAWNBasic:
    def test_values_are_bounded_and_rank_the_ishigami_fixture(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y)
        pawn = np.asarray(result.pawn)
        assert np.all(pawn >= 0.0)
        assert np.all(pawn <= 1.0)
        assert pawn[0] > pawn[2], "x1 should be more important than x3"
        assert pawn[1] > pawn[2], "x2 should be more important than x3"
        median = np.asarray(analyze(ishigami.PROBLEM, X, Y, statistic="median").pawn)
        maximum = np.asarray(analyze(ishigami.PROBLEM, X, Y, statistic="max").pawn)
        assert np.all(maximum >= median - 1e-6)

    def test_slice_chunk_size_invariance(self, ishigami_data):
        """Tier T4 (internal consistency): chunking changes no index.

        ``slice_chunk_size`` splits the flattened ``T*K`` output columns into
        separate kernel calls. Every column is independent of every other, so
        the chunked result must equal the unchunked one exactly, not merely
        to a tolerance. The bootstrap intervals are compared too: they run
        through the same loop, once per resample.

        The test is not vacuous. The output has ``T*K == 6`` columns and the
        chunk size is 4, so the loop runs twice and the second chunk is a
        short one. Both asserts below state that, and they fail if a future
        edit shrinks the output or raises the chunk size past it.
        """
        X, Y = ishigami_data
        X = X[:500]
        Y = Y[:500]
        # (N, T, K) with T = 3, K = 2, so six flattened output columns.
        Y_3d = jnp.stack(
            [
                jnp.stack([Y, 2.0 * Y], axis=-1),
                jnp.stack([jnp.sin(Y), Y**2], axis=-1),
                jnp.stack([-Y, jnp.cos(Y)], axis=-1),
            ],
            axis=1,
        )
        total = Y_3d.shape[1] * Y_3d.shape[2]
        chunk = 4
        assert total > chunk, "chunk must be smaller than T*K or nothing is split"
        assert total % chunk != 0, "an uneven split exercises the short trailing chunk"

        full = analyze(
            ishigami.PROBLEM, X, Y_3d, n_bootstrap=8, conf_level=0.9, key=jax.random.key(3)
        )
        chunked = analyze(
            ishigami.PROBLEM,
            X,
            Y_3d,
            n_bootstrap=8,
            conf_level=0.9,
            key=jax.random.key(3),
            slice_chunk_size=chunk,
        )

        assert full.pawn.shape == (3, 2, 3)
        assert full.pawn_conf is not None
        assert chunked.pawn_conf is not None
        np.testing.assert_array_equal(np.asarray(chunked.pawn), np.asarray(full.pawn))
        np.testing.assert_array_equal(np.asarray(chunked.pawn_conf), np.asarray(full.pawn_conf))
        assert chunked.problem is full.problem

    def test_slice_chunk_size_must_be_positive(self, ishigami_data):
        """Tier T4 (internal consistency): an unusable chunk size is refused."""
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="slice_chunk_size must be >= 1"):
            analyze(ishigami.PROBLEM, X[:200], Y[:200], slice_chunk_size=0)


class TestPAWNBootstrap:
    def test_bootstrap_lower_leq_upper(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=20, conf_level=0.95, key=jax.random.key(0)
        )
        assert result.pawn_conf is not None
        lower = np.asarray(result.pawn_conf[0])
        upper = np.asarray(result.pawn_conf[1])
        assert np.all(lower <= upper + 1e-6)

    def test_bootstrap_without_a_key_raises(self, ishigami_data):
        """Tier T4. A bootstrap needs a key, and an int seed is not one."""
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="key is required"):
            analyze(ishigami.PROBLEM, X, Y, n_bootstrap=4)

    def test_same_key_same_interval(self, ishigami_data):
        """Tier T4. Determinism is a property of the key, not of a seed."""
        X, Y = ishigami_data
        r1 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=8, key=jax.random.key(11))
        r2 = analyze(ishigami.PROBLEM, X, Y, n_bootstrap=8, key=jax.random.key(11))
        assert r1.pawn_conf is not None and r2.pawn_conf is not None
        np.testing.assert_array_equal(np.asarray(r1.pawn_conf), np.asarray(r2.pawn_conf))

    def test_gaussian_ci_is_centred_on_the_estimate(self, ishigami_data):
        """Tier T0 (closed form): the Gaussian interval is symmetric.

        ``estimate +/- z * sd`` puts the point estimate exactly at the
        midpoint of the two endpoints, which the percentile interval does
        not do. That is what tells the two branches apart.
        """
        X, Y = ishigami_data
        result = analyze(
            ishigami.PROBLEM, X, Y, n_bootstrap=20, ci_method="gaussian", key=jax.random.key(0)
        )
        assert result.ci is not None
        assert result.ci.method == "gaussian"
        assert result.pawn_conf is not None
        midpoint = 0.5 * (np.asarray(result.pawn_conf[0]) + np.asarray(result.pawn_conf[1]))
        np.testing.assert_allclose(midpoint, np.asarray(result.pawn), atol=1e-6)

    def test_unknown_ci_method_rejected(self, ishigami_data):
        X, Y = ishigami_data
        with pytest.raises(ValueError, match="ci_method"):
            analyze(
                ishigami.PROBLEM,
                X,
                Y,
                n_bootstrap=4,
                ci_method=cast(Any, "bca"),
                key=jax.random.key(0),
            )


class TestPAWNTiedOutputs:
    """Tied or constant outputs keep the public statistic well-defined."""

    def test_constant_output_zero_ks(self):
        """All-equal Y => conditional == unconditional => KS is 0."""
        problem = Problem(names=("a",), bounds=((0.0, 1.0),))
        X = np.linspace(0.0, 1.0, 100).reshape(-1, 1)
        y = np.full(100, 3.0)
        result = analyze(problem, jnp.asarray(X), jnp.asarray(y), n_bins=5)
        np.testing.assert_allclose(np.asarray(result.pawn), 0.0, atol=1e-6)


class TestPAWNEmptyBinWarning:
    def test_warns_once_not_per_bootstrap(self):
        """The empty-bin warning fires once, not once per bootstrap resample."""
        import warnings as _warnings

        problem = Problem(names=("a",), bounds=((0.0, 1.0),))
        # N < n_bins with spread inputs => every bin has < 2 samples.
        X = np.linspace(0.02, 0.98, 8).reshape(-1, 1)
        y = np.arange(8, dtype=float)
        with _warnings.catch_warnings(record=True) as rec:
            _warnings.simplefilter("always")
            analyze(
                problem,
                jnp.asarray(X),
                jnp.asarray(y),
                n_bins=20,
                n_bootstrap=5,
                key=jax.random.key(0),
            )
        msgs = [r for r in rec if "all bins empty" in str(r.message)]
        assert len(msgs) == 1


def _invalid_sample(n: int = 200, seed: int = 0):
    """Build a clean two-parameter PAWN sample for the on_invalid tests."""
    problem = Problem(names=("a", "b"), bounds=((0.0, 1.0), (0.0, 1.0)))
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))
    Y = X[:, 0] ** 2 + 0.5 * X[:, 1]
    return problem, jnp.asarray(X), jnp.asarray(Y)


class TestPAWNInvalidPolicy:
    """T4 (behaviour): jaxgsa.pawn.analyze honours the shared on_invalid policy."""

    def test_raise_is_the_default_and_names_the_rows(self):
        """T4: a non-finite Y row refuses the analysis and says which row it is.

        The row index is the whole point of the message: it names the model
        evaluation the user has to investigate.
        """
        problem, X, Y = _invalid_sample()
        Y = Y.at[7].set(jnp.nan)
        with pytest.raises(ValueError) as exc:
            analyze(problem, X, Y)
        message = str(exc.value)
        assert "jaxgsa.pawn.analyze" in message
        assert "1 of 200 rows" in message
        assert "[7]" in message

    def test_drop_removes_the_row_and_returns_finite_indices(self):
        """T4: 'drop' analyzes the remainder and the indices come back usable."""
        problem, X, Y = _invalid_sample()
        Y = Y.at[7].set(jnp.nan)
        with pytest.warns(JaxgsaWarning, match="dropped 1 of 200 rows"):
            result = analyze(problem, X, Y, on_invalid="drop")
        assert result.invalid.n_kept == 199
        assert result.invalid.unit_indices == (7,)
        assert np.all(np.isfinite(np.asarray(result.pawn)))


class TestPureCore:
    """The transformable core ``pawn.indices``.

    Tier T4 throughout (internal consistency and transformability). The KS
    statistic itself is checked against ``scipy.stats.ks_2samp`` elsewhere in
    this file, and ``indices`` runs the same kernel ``analyze`` runs.
    """

    def test_matches_analyze_scalar(self, ishigami_data):
        """T4: ``indices`` returns exactly what ``analyze`` reports."""
        X, Y = ishigami_data

        result = analyze(ishigami.PROBLEM, X, Y, n_bins=8)
        (pawn,) = indices(ishigami.PROBLEM, X, Y, n_bins=8)

        np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))

    def test_matches_analyze_for_every_statistic(self, ishigami_data):
        """T4: the aggregation choice is a Python branch, shared by both paths."""
        X, Y = ishigami_data
        for statistic in ("median", "max", "mean"):
            kwargs: dict[str, Any] = {"n_bins": 8, "statistic": statistic}
            result = analyze(ishigami.PROBLEM, X, Y, **kwargs)
            (pawn,) = indices(ishigami.PROBLEM, X, Y, **kwargs)
            np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))

    def test_matches_analyze_time_series(self, ishigami_data):
        """T4: the chunked ``(N, T, K)`` path agrees with ``analyze``."""
        X, base = ishigami_data
        Y = jnp.stack(
            [
                jnp.stack([base, 2.0 * base + X[:, 0]], axis=-1),
                jnp.stack([base**2, X[:, 1]], axis=-1),
            ],
            axis=1,
        )
        assert Y.shape == (X.shape[0], 2, 2)

        result = analyze(ishigami.PROBLEM, X, Y, n_bins=8)
        (pawn,) = indices(ishigami.PROBLEM, X, Y, n_bins=8)

        assert pawn.shape == (2, 2, 3)
        np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))

    def test_matches_analyze_with_a_categorical_parameter(self):
        """T4: the traced level-code path bins a categorical column as ``analyze`` does."""
        problem = Problem.from_dict(
            {
                "a": {"dist": "categorical", "probs": [0.4, 0.6]},
                "b": (0.0, 1.0),
            }
        )
        X = jnp.asarray(monte_carlo(problem, n=1000, seed=4))
        Y = 2.0 * X[:, 0] + X[:, 1]

        result = analyze(problem, X, Y, n_bins=5)
        (pawn,) = indices(problem, X, Y, n_bins=5)

        np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))

    # Each marginal family reaches ``cdf_to_unit_interval`` by a different
    # branch, and two of those branches used to break tracing in different
    # ways: a truncated Gaussian read ``X`` on the host through SciPy, and
    # *every* Gaussian, truncated or not, took ``float()`` of a tracer while
    # standardising. They were separate defects, so they get separate cases.
    _MARGINALS: dict[str, InputSpecValue] = {
        "uniform": (0.0, 1.0),
        "gaussian": {"dist": "gaussian", "mean": 0.3, "variance": 2.0},
        "truncated": {
            "dist": "gaussian",
            "mean": 0.0,
            "variance": 1.0,
            "low": -1.5,
            "high": 2.0,
        },
        # Both bounds far out in the upper tail. The direct
        # ``(Phi(z) - Phi(a)) / (Phi(b) - Phi(a))`` form cancels to noise here,
        # so this case pins the survival-function branch of the transform.
        "upper_tail": {
            "dist": "gaussian",
            "mean": 0.0,
            "variance": 1.0,
            "low": 5.0,
            "high": 6.0,
        },
    }

    def test_every_marginal_traces_and_matches_analyze(self):
        """T4: ``jit`` holds for every marginal family, not only uniform.

        A uniform-only contract test passes while the Gaussian branch of the
        unit-interval transform cannot be traced at all, which is exactly the
        state this package shipped in. The eager comparison against ``analyze``
        and the jitted comparison against the eager core are both needed: the
        first is the value contract, the second is the traceability one, and a
        host read inside the transform breaks only the second.
        """
        for marginal in self._MARGINALS.values():
            problem = Problem.from_dict({"a": marginal, "b": (0.0, 1.0)})
            X = jnp.asarray(monte_carlo(problem, n=1200, seed=17))
            Y = jnp.sin(X[:, 0]) + 0.5 * X[:, 1]
            result = analyze(problem, X, Y, n_bins=6)
            (pawn,) = indices(problem, X, Y, n_bins=6)
            np.testing.assert_array_equal(np.asarray(pawn), np.asarray(result.pawn))
            (pawn_jit,) = jax.jit(lambda outputs: indices(problem, X, outputs, n_bins=6))(Y)
            np.testing.assert_allclose(np.asarray(pawn_jit), np.asarray(pawn), rtol=1e-6)


class TestBinCountDiagnostic:
    """Tier T4 (internal consistency): the contributing-bin diagnostic."""

    def test_full_bins_report_n_bins_and_match_pawn_shape(self, ishigami_data):
        X, Y = ishigami_data
        result = analyze(ishigami.PROBLEM, X, Y, n_bins=10)
        n_valid = np.asarray(result.n_valid_bins)
        assert n_valid.shape == np.asarray(result.pawn).shape
        # 5000 samples over 10 equal-probability bins: every bin holds
        # hundreds of samples, so every bin contributes.
        np.testing.assert_array_equal(n_valid, np.full_like(n_valid, 10))

    def test_diagnostic_broadcasts_over_output_axes(self, ishigami_data):
        """Bin occupancy never reads Y, so the count repeats across (T, K)."""
        X, Y = ishigami_data
        Y3 = jnp.stack([jnp.stack([Y, Y**2], axis=1)] * 2, axis=1)  # (N, 2, 2)
        result = analyze(ishigami.PROBLEM, X, Y3, n_bins=8)
        n_valid = np.asarray(result.n_valid_bins)
        assert n_valid.shape == (2, 2, 3)
        assert (n_valid == n_valid[0, 0]).all()

    def test_to_dataset_carries_the_diagnostic(self, ishigami_data):
        X, Y = ishigami_data
        ds = analyze(ishigami.PROBLEM, X, Y).to_dataset()
        assert "n_valid_bins" in ds
        assert ds["n_valid_bins"].dims == ("param",)

    def test_sparse_bins_warn_and_name_the_fix(self):
        """A parameter whose samples pile into few bins gets one warning."""
        problem = Problem(names=("narrow", "wide"), bounds=((0.0, 1.0), (0.0, 1.0)))
        rng = np.random.default_rng(0)
        X = np.column_stack(
            [
                # All mass in the first of 10 equal-width bins: 1/10 bins
                # contribute, which is below half.
                rng.uniform(0.0, 0.05, size=64),
                rng.uniform(0.0, 1.0, size=64),
            ]
        )
        Y = X[:, 0] + X[:, 1]
        with pytest.warns(JaxgsaWarning, match="fewer bins .* or more samples"):
            result = analyze(problem, jnp.asarray(X), jnp.asarray(Y), n_bins=10)
        n_valid = np.asarray(result.n_valid_bins)
        assert n_valid[0] == 1
        assert n_valid[1] > 5
