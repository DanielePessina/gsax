"""Tests for categorical (unordered discrete) input support.

Mostly Tier T4 (internal consistency): validation, sampling, persistence,
the partition layer, and method gating check the library's own stated
contract, not an external reference. The "OT + Borgonovo estimates" section
is the exception -- it compares against a closed-form true value (Tier T0)
built from balanced categorical levels with known offsets, with a tolerance
wide enough for the estimator's own measured bias.
"""

import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa import (
    borgonovo,
    dgsm,
    efast,
    hdmr,
    hsic,
    morris,
    optimal_transport,
    pawn,
    shapley,
    sobol,
)
from jaxgsa import pce as pce_mod
from jaxgsa._core.partition import (
    _extract_categorical_codes,
)
from jaxgsa._core.sampling import _inverse_transform_samples
from jaxgsa._core.transforms import cdf_to_unit_interval
from jaxgsa.borgonovo._analyze import _warn_conf_out_of_range
from jaxgsa.problem import (
    CategoricalSpec,
    Problem,
    _categorical_dims,
    _normalized_input_to_dict,
)

PROBS = [0.5, 0.3, 0.2]
OFFSETS = np.array([0.0, 2.0, -1.0])


def _mixed_problem():
    """One uniform and one 3-level categorical parameter."""
    return Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": PROBS},
        }
    )


def _mixed_data(n=4000, seed=1, noise=0.1):
    """Sample the mixed problem and evaluate the per-level-offset model."""
    problem = _mixed_problem()
    X = jaxgsa.sampling.monte_carlo(problem, n, seed=seed)
    codes = X[:, 1].astype(int)
    rng = np.random.default_rng(seed)
    Y = X[:, 0] + OFFSETS[codes] + noise * rng.standard_normal(n)
    return problem, X, Y


def _analytic_s1():
    """Closed-form S1 of (x1, c) for the per-level-offset model."""
    pr = np.array(PROBS)
    m = (pr * OFFSETS).sum()
    v_cat = (pr * (OFFSETS - m) ** 2).sum()
    v_x1 = 1.0 / 12.0
    v_y = v_cat + v_x1 + 0.01  # + noise variance
    return v_x1 / v_y, v_cat / v_y


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_categorical_spec_normalizes_and_stores_probs_and_labels():
    """Tier T4 (internal consistency): the normalized spec keeps the declared
    probabilities and supplies default string labels.

    The stored payload is also checked directly, not through
    ``_normalized_input_to_dict``. That helper rebuilds ``probs`` and
    ``labels`` as lists for the public view, so it cannot see whether the
    stored spec is still hashable. ``Problem`` is frozen and its specs go into
    cache keys, so both must stay tuples.
    """
    p = Problem.from_dict({"c": {"dist": "categorical", "probs": [0.25, 0.25, 0.5]}})
    stored = p.input_specs[0]
    assert isinstance(stored, CategoricalSpec)
    assert isinstance(stored.probs, tuple)
    assert isinstance(stored.labels, tuple)
    spec = _normalized_input_to_dict(p.input_specs[0])
    assert spec["dist"] == "categorical"
    assert list(spec["probs"]) == [0.25, 0.25, 0.5]
    assert list(spec["labels"]) == ["0", "1", "2"]
    assert p.has_categorical_inputs is True
    assert p.has_non_uniform_inputs is True
    assert p.bounds is None
    assert _categorical_dims(p) == ((0, 3),)


def test_categorical_spec_renormalizes_small_prob_error():
    """Tier T0 (closed form): stored probabilities sum to exactly one."""
    p = Problem.from_dict({"c": {"dist": "categorical", "probs": [1 / 3, 1 / 3, 1 / 3]}})
    probs = _normalized_input_to_dict(p.input_specs[0])["probs"]
    assert sum(probs) == pytest.approx(1.0, abs=1e-15)


def test_categorical_spec_labels_stored_as_strings():
    p = Problem.from_dict(
        {"c": {"dist": "categorical", "probs": [0.5, 0.5], "labels": ["red", 7]}}
    )
    assert p.categorical_labels == {"c": ("red", "7")}


def test_categorical_spec_rejects_invalid_input():
    cases = [
        ({"dist": "categorical", "probs": [1.0]}, "at least 2 levels"),
        ({"dist": "categorical", "probs": [0.5, -0.5, 1.0]}, "positive"),
        ({"dist": "categorical", "probs": [0.5, 0.0, 0.5]}, "positive"),
        ({"dist": "categorical", "probs": [0.5, np.nan]}, "positive"),
        ({"dist": "categorical", "probs": [0.3, 0.3]}, "sum to 1"),
        ({"dist": "categorical", "probs": [0.5, 0.5], "labels": ["a"]}, "labels length"),
        ({"dist": "categorical", "probs": [0.5, 0.5], "labels": ["a", "a"]}, "unique"),
    ]
    for spec, match in cases:
        with pytest.raises(ValueError, match=match):
            Problem.from_dict({"c": spec})


# ---------------------------------------------------------------------------
# Correlation x categorical
# ---------------------------------------------------------------------------


def test_correlation_touching_categorical_raises():
    with pytest.raises(ValueError, match="polychoric"):
        Problem.from_dict(
            {"x": (0.0, 1.0), "c": {"dist": "categorical", "probs": [0.5, 0.5]}},
            correlation=[[1.0, 0.3], [0.3, 1.0]],
        )


def test_correlation_with_identity_categorical_rows_passes():
    p = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": [0.5, 0.5]},
        },
        correlation=[[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert p.has_correlated_inputs is True
    X = jaxgsa.sampling.monte_carlo(p, 20_000, seed=0)
    freqs = np.bincount(X[:, 2].astype(int), minlength=2) / X.shape[0]
    np.testing.assert_allclose(freqs, [0.5, 0.5], atol=0.02)
    assert abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1]) > 0.3


def test_with_correlation_touching_categorical_raises():
    p = _mixed_problem()
    with pytest.raises(ValueError, match="'c'"):
        p.with_correlation([[1.0, -0.2], [-0.2, 1.0]])


def test_identity_categorical_rows_survive_psd_repair():
    """The PSD repair's float noise must not read as a categorical coupling."""
    p = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": (0.0, 1.0),
            "x3": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": [0.5, 0.5]},
        }
    )
    # Mildly non-PSD continuous block; the categorical row is exact identity.
    # The repair has to engage, but must stay under the "declared" policy's
    # material threshold so it warns rather than raising — a matrix that moved
    # further would be rejected before the categorical reset ever ran.
    R = np.array(
        [
            [1.0, 0.52, 0.52, 0.0],
            [0.52, 1.0, -0.52, 0.0],
            [0.52, -0.52, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.linalg.eigvalsh(R).min() < 0  # the repair must engage
    with pytest.warns(UserWarning, match="not positive definite"):
        p2 = p.with_correlation(R)
    stored = p2.correlation
    assert stored is not None
    # The stored matrix carries exact identity categorical rows and columns.
    assert np.all(stored[3, :3] == 0.0)
    assert np.all(stored[:3, 3] == 0.0)
    assert stored[3, 3] == 1.0
    # A genuine coupling in the same problem still raises.
    R_bad = np.eye(4)
    R_bad[0, 3] = R_bad[3, 0] = 0.3
    with pytest.raises(ValueError, match="polychoric"):
        p.with_correlation(R_bad)


def test_fit_correlation_roundtrip_on_mixed_problem():
    """with_correlation(fit_correlation(...)) must work end to end."""
    p = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": PROBS},
        },
        correlation=[[1.0, 0.6, 0.0], [0.6, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    X = jaxgsa.sampling.monte_carlo(p, 2000, seed=0)
    with pytest.warns(UserWarning, match="categorical"):
        fitted = jaxgsa.sampling.fit_correlation(p, X)
    p2 = p.with_correlation(fitted)  # must not raise
    R2 = p2.correlation
    assert R2 is not None
    assert abs(R2[0, 1] - 0.6) < 0.1
    assert np.all(R2[2, :2] == 0.0) and np.all(R2[:2, 2] == 0.0)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_monte_carlo_preserves_categorical_and_continuous_marginals():
    p = Problem.from_dict({"c": {"dist": "categorical", "probs": PROBS}})
    X = jaxgsa.sampling.monte_carlo(p, 10_000, seed=0)
    codes = X[:, 0]
    assert np.all(codes == np.round(codes))
    freqs = np.bincount(codes.astype(int), minlength=3) / X.shape[0]
    np.testing.assert_allclose(freqs, PROBS, atol=0.025)
    p = _mixed_problem()
    X = jaxgsa.sampling.monte_carlo(p, 10_000, seed=2)
    assert X[:, 0].min() >= 0.0 and X[:, 0].max() <= 1.0
    np.testing.assert_allclose(X[:, 0].mean(), 0.5, atol=0.025)
    freqs = np.bincount(X[:, 1].astype(int), minlength=3) / X.shape[0]
    np.testing.assert_allclose(freqs, PROBS, atol=0.025)


def test_continuous_transform_helpers_reject_categorical_inputs():
    p = _mixed_problem()
    X = jaxgsa.sampling.monte_carlo(p, 16, seed=0)
    with pytest.raises(ValueError, match="'c'"):
        _inverse_transform_samples(p, X)
    with pytest.raises(ValueError, match="'c'"):
        cdf_to_unit_interval(X, p)


# ---------------------------------------------------------------------------
# Persistence round-trips
# ---------------------------------------------------------------------------


def test_problem_meta_json_round_trip_carries_probs_and_labels():
    from jaxgsa._core.samples import _problem_from_meta, _problem_to_meta

    p = Problem.from_dict(
        {
            "x": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": PROBS, "labels": ["a", "b", "c"]},
        }
    )
    meta = json.loads(json.dumps(_problem_to_meta(p)))  # force a real JSON round-trip
    restored = _problem_from_meta(meta)
    assert restored == p
    assert restored.categorical_labels == {"c": ("a", "b", "c")}


def test_sobol_npz_round_trip_carries_categorical_problem(tmp_path):
    p = Problem.from_dict(
        {
            "x": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": PROBS, "labels": ["lo", "mid", "hi"]},
        }
    )
    sr = sobol.sample(p, 64, seed=0, verbose=False)
    path = tmp_path / "design.npz"
    sr.save(path)
    loaded = sobol.SobolSamples.load(path)
    assert loaded.problem == p
    assert loaded.problem.categorical_labels == {"c": ("lo", "mid", "hi")}
    np.testing.assert_array_equal(loaded.samples, sr.samples)


# ---------------------------------------------------------------------------
# Partition layer
# ---------------------------------------------------------------------------


def test_extract_categorical_codes_rejects_non_code_values():
    p = _mixed_problem()
    dims_levels = _categorical_dims(p)
    X = np.asarray(jaxgsa.sampling.monte_carlo(p, 32, seed=0))
    bad = X.copy()
    bad[0, 1] = 0.5
    with pytest.raises(ValueError, match="'c'"):
        _extract_categorical_codes(p, bad, dims_levels)
    bad = X.copy()
    bad[0, 1] = 3.0  # out of range for 3 levels
    with pytest.raises(ValueError, match="'c'"):
        _extract_categorical_codes(p, bad, dims_levels)


def test_empty_declared_level_warns_and_completes():
    p = Problem.from_dict(
        {"c": {"dist": "categorical", "probs": [0.5, 0.4, 0.1]}},
    )
    rng = np.random.default_rng(0)
    n = 400
    codes = rng.integers(0, 2, size=n)  # level 2 never observed
    X = codes[:, None].astype(np.float64)
    Y = 1.0 * codes + 0.1 * rng.standard_normal(n)
    with pytest.warns(UserWarning, match="no\\s+samples at level"):
        res = borgonovo.analyze(p, X, Y, n_bootstrap=4, key=jax.random.key(0))
    assert np.all(np.isfinite(np.asarray(res.delta)))
    assert float(res.S1[0]) > 0.8


# ---------------------------------------------------------------------------
# OT + Borgonovo estimates
# ---------------------------------------------------------------------------


def test_borgonovo_degenerate_class_recovers_delta():
    """A near-atomic per-level output must not bias delta low.

    Y = offsets[code] with three balanced levels has true delta = 2/3.
    Without the bandwidth floor the degenerate (near-zero-variance) classes
    drop out of the integrand and delta collapses to ~0.33. With the floor
    the estimate lands at 0.60-0.62 for N in [1e3, 1e4] (measured error
    <= 0.07; asserted with headroom). The jitter is 1e-9, far below the
    atom spacing, so the classes stay degenerate; an exactly noise-free Y
    is a discrete output and ``analyze`` refuses it (see
    ``test_borgonovo_refuses_a_discrete_output``).
    """
    p, X, Y = _atom_data(1e-9)
    with pytest.warns(UserWarning, match="bandwidth"):
        res = borgonovo.analyze(p, X, Y, key=jax.random.key(0))
    assert abs(float(res.delta[0]) - 2.0 / 3.0) < 0.11
    assert float(res.S1[0]) == pytest.approx(1.0, abs=1e-6)


def _atom_data(noise, n=3000, seed=0):
    """Three balanced atoms plus jitter; true delta is 2/3 at every noise."""
    p = Problem.from_dict({"c": {"dist": "categorical", "probs": [1 / 3, 1 / 3, 1 / 3]}})
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 3, n)
    Y = OFFSETS[codes] + noise * np.random.default_rng(42).standard_normal(n)
    return p, codes[:, None].astype(np.float64), Y


def test_borgonovo_refuses_a_discrete_output():
    """Zero jitter makes Y discrete, which the estimator does not support.

    The two guards are complementary. A discrete output is caught here, up
    front, by the distinct-value check. An output with a tiny jitter has
    about N distinct values, passes this check, and is caught later by the
    delta range check if its conditional density aliases on the grid.
    """
    p, X, Y = _atom_data(0.0)
    assert len(np.unique(Y)) == 3
    with pytest.raises(ValueError, match="continuous output distribution only"):
        jaxgsa.borgonovo.analyze(p, X, Y, key=jax.random.key(0), n_bootstrap=0)


def test_borgonovo_accepts_a_rounded_continuous_output():
    """Rounding to 2 decimals leaves many distinct values, so it is allowed."""
    p = Problem(names=("a", "b"), bounds=((0.0, 1.0), (0.0, 1.0)))
    rng = np.random.default_rng(1)
    X = rng.random((3000, 2))
    Y = np.round(X[:, 0] + 0.5 * X[:, 1], 2)
    assert len(np.unique(Y)) > 20
    res = jaxgsa.borgonovo.analyze(p, X, Y, n_bootstrap=0)
    assert float(np.asarray(res.delta)[0]) > float(np.asarray(res.delta)[1])


def test_borgonovo_discrete_output_names_the_offending_column():
    """With several outputs the message says which column is discrete."""
    p = Problem(names=("a",), bounds=((0.0, 1.0),), output_names=("smooth", "atoms"))
    rng = np.random.default_rng(2)
    X = rng.random((2000, 1))
    Y = np.stack([X[:, 0], np.floor(3 * X[:, 0])], axis=1)
    with pytest.raises(ValueError, match=r"k=1 \('atoms'\)"):
        jaxgsa.borgonovo.analyze(p, X, Y, n_bootstrap=0)


def test_borgonovo_atomic_class_delta_stays_in_range():
    """The atoms are separated far beyond the jitter, so delta is 2/3.

    Before the degenerate-class tolerance was raised to 1e-2 the class
    bandwidth could sit orders of magnitude below the output grid step. The
    conditional density then aliased on the grid and the trapezoid integral
    exploded: noise 1e-5 returned delta 121 with no warning at all. Zero
    noise is not swept here: it makes Y discrete, and the estimator refuses
    a discrete output up front.
    """
    for noise in (1e-9, 1e-5, 0.1):
        p, X, Y = _atom_data(noise)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = jaxgsa.borgonovo.analyze(p, X, Y, key=jax.random.key(0), n_bootstrap=0)
        delta = float(np.asarray(res.delta).ravel()[0])
        assert 0.0 <= delta <= 1.0
        assert abs(delta - 2.0 / 3.0) < 0.11


def test_borgonovo_aliasing_delta_raises_out_of_range():
    """An unresolvable class must never return a huge delta.

    A delta outside [0, 1] is a failed computation, so it is an error, not
    a number the caller can plot. The message names both knobs.
    """
    p, X, Y = _atom_data(1e-5)
    # degenerate_tol=1e-6 restores the old, too-low detection threshold.
    with pytest.raises(ValueError, match=r"delta is a half L1 distance") as excinfo:
        jaxgsa.borgonovo.analyze(
            p, X, Y, key=jax.random.key(0), n_bootstrap=0, degenerate_tol=1e-6
        )
    message = str(excinfo.value)
    assert "'c'" in message or "c:" in message
    assert "grid_size (currently 100)" in message
    assert "degenerate_bandwidth" in message


def test_borgonovo_out_of_range_confidence_bound_only_warns():
    """The interval is a diagnostic, so a bad bound warns instead of raising."""
    p = Problem(names=("a", "b"), bounds=((0.0, 1.0), (0.0, 1.0)))
    conf = np.array([[-0.4, 0.1], [0.6, 1.9]])  # lower bad in a, upper bad in b
    with pytest.warns(UserWarning, match="bootstrap confidence") as record:
        _warn_conf_out_of_range(p, jnp.asarray(conf))
    message = str(record[0].message)
    assert "lower bound for a" in message
    assert "upper bound for b" in message


def test_borgonovo_rejects_invalid_degenerate_settings():
    p, X, Y = _atom_data(0.1, n=200)
    cases = [
        ({"degenerate_tol": 1.0}, "degenerate_tol"),
        ({"degenerate_tol": -0.1}, "degenerate_tol"),
        ({"degenerate_bandwidth": 0.0}, "degenerate_bandwidth"),
        ({"degenerate_bandwidth": "wide"}, "degenerate_bandwidth"),
        ({"degenerate_bandwidth": True}, "degenerate_bandwidth"),
    ]
    for kwargs, match in cases:
        with pytest.raises(ValueError, match=match):
            borgonovo.analyze(p, X, Y, n_bootstrap=0, **kwargs)


def test_categorical_borgonovo_and_ot_match_the_fixture():
    problem, X, Y = _mixed_data(n=8000)
    s1_x1, s1_cat = _analytic_s1()
    borgonovo_result = borgonovo.analyze(problem, X, Y, key=jax.random.key(0))
    ot_result = optimal_transport.analyze(problem, X, Y)
    np.testing.assert_allclose(np.asarray(borgonovo_result.S1), [s1_x1, s1_cat], atol=0.03)
    np.testing.assert_allclose(2.0 * np.asarray(ot_result.advective), [s1_x1, s1_cat], atol=0.03)
    assert float(ot_result.ot[1]) > float(ot_result.ot[0])


def test_level_permutation_invariance():
    """The acid test: relabeling levels must not change the indices."""
    problem, X, Y = _mixed_data(n=4000)
    codes = X[:, 1].astype(int)

    perm = np.array([2, 0, 1])  # old code -> new code
    inv = np.argsort(perm)
    X_perm = X.copy()
    X_perm[:, 1] = perm[codes]
    probs_perm = [PROBS[inv[level]] for level in range(3)]
    problem_perm = Problem.from_dict(
        {"x1": (0.0, 1.0), "c": {"dist": "categorical", "probs": probs_perm}}
    )

    cases = [
        (
            borgonovo.analyze(problem, X, Y, n_bootstrap=16, key=jax.random.key(0)),
            borgonovo.analyze(problem_perm, X_perm, Y, n_bootstrap=16, key=jax.random.key(0)),
            ("delta", "S1"),
        ),
        (
            optimal_transport.analyze(problem, X, Y, n_bootstrap=16, key=jax.random.key(0)),
            optimal_transport.analyze(
                problem_perm, X_perm, Y, n_bootstrap=16, key=jax.random.key(0)
            ),
            ("ot", "advective"),
        ),
    ]
    for result, permuted, fields in cases:
        for field in fields:
            np.testing.assert_allclose(
                np.asarray(getattr(result, field)), np.asarray(getattr(permuted, field)), atol=1e-5
            )


# ---------------------------------------------------------------------------
# PAWN with categorical inputs
# ---------------------------------------------------------------------------


def test_pawn_accepts_a_mixed_problem():
    problem, X, Y = _mixed_data(n=6000)
    res = pawn.analyze(problem, X, Y, n_bootstrap=8, key=jax.random.key(0))
    values = np.asarray(res.pawn)
    assert values.shape == (2,)
    assert np.isfinite(values).all()
    # The level offsets dominate the unit-uniform x1 term.
    assert values[1] > values[0]
    lo, hi = np.asarray(res.pawn_conf)
    assert np.all(lo <= hi)


def test_pawn_accepts_an_all_categorical_problem():
    p, X, Y = _all_categorical_data(n=4000)
    res = pawn.analyze(p, X, Y)
    values = np.asarray(res.pawn)
    assert np.isfinite(values).all()
    # Y depends on c1 only, so c2 is near zero.
    assert values[0] > 0.45
    assert values[1] < 0.15


def test_pawn_rejects_non_code_values_in_a_categorical_column():
    problem, X, Y = _mixed_data(n=200)
    bad = np.asarray(X).copy()
    bad[0, 1] = 0.5
    with pytest.raises(ValueError, match="integer level codes"):
        pawn.analyze(problem, bad, Y)


def test_pawn_level_relabeling_is_invariant():
    """Relabeling levels only reorders the per-bin KS values.

    Median, max and mean over bins are permutation-invariant, so median and
    max are exactly equal. The mean sums the same three floats in a
    different order, which costs at most one float32 ULP.
    """
    problem, X, Y = _mixed_data(n=4000)
    codes = np.asarray(X)[:, 1].astype(int)
    perm = np.array([2, 0, 1])
    inv = np.argsort(perm)
    X_perm = np.asarray(X).copy()
    X_perm[:, 1] = perm[codes]
    problem_perm = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": [PROBS[inv[i]] for i in range(3)]},
        }
    )

    for statistic in ("median", "max", "mean"):
        base = np.asarray(pawn.analyze(problem, X, Y, statistic=statistic).pawn)
        other = np.asarray(pawn.analyze(problem_perm, X_perm, Y, statistic=statistic).pawn)
        if statistic == "mean":
            np.testing.assert_allclose(base, other, rtol=0, atol=np.finfo(np.float32).eps)
        else:
            np.testing.assert_array_equal(base, other)


def test_ot_modes_and_dummy_baseline_support_categorical_inputs():
    problem, X, Y = _mixed_data(n=2000)
    rng = np.random.default_rng(3)
    Y2 = np.stack([Y, -Y + 0.05 * rng.standard_normal(Y.shape[0])], axis=1)
    res = optimal_transport.analyze(problem, X, Y2, mode="multivariate", max_iter=5000)
    assert res.ot.shape == (2,)
    assert float(res.ot[1]) > float(res.ot[0])
    dummy = optimal_transport.analyze(problem, X, Y, dummy=True, key=jax.random.key(0))
    assert dummy.ot_dummy is not None
    assert float(dummy.ot_dummy[1]) < float(dummy.ot[1])


def _all_categorical_data(n=2000, seed=2):
    p = Problem.from_dict(
        {
            "c1": {"dist": "categorical", "probs": [0.5, 0.5]},
            "c2": {"dist": "categorical", "probs": [0.25, 0.25, 0.5]},
        }
    )
    X = jaxgsa.sampling.monte_carlo(p, n, seed=seed)
    rng = np.random.default_rng(seed)
    Y = 3.0 * X[:, 0] + 0.5 * rng.standard_normal(n)
    return p, X, Y


def test_all_categorical_partition_options_are_validated_and_adaptive():
    p, X, Y = _all_categorical_data()
    with pytest.raises(ValueError, match="n_partitions"):
        optimal_transport.analyze(p, X, Y, n_partitions=10**9)
    with pytest.raises(ValueError, match="n_classes"):
        borgonovo.analyze(p, X, Y, n_classes=10**9, n_bootstrap=0)
    with pytest.warns(UserWarning, match="n_partitions is ignored"):
        res = optimal_transport.analyze(p, X, Y, n_partitions=10)
    assert float(res.ot[0]) > 0.5
    assert float(res.ot[1]) < 0.05
    with pytest.warns(UserWarning, match="n_classes is ignored"):
        res_b = borgonovo.analyze(p, X, Y, n_classes=10, n_bootstrap=0)
    assert float(res_b.S1[0]) > 0.7
    res = optimal_transport.analyze(p, X, Y)
    assert float(res.ot[0]) > 0.5
    # With a dummy the passed value is used; out of range names n_partitions.
    # M4: no dummy reads n_partitions any more (each categorical column gets
    # its own permutation floor), so the scope text this used to match is
    # gone; the bound itself still applies.
    with pytest.raises(ValueError, match="n_partitions"):
        optimal_transport.analyze(p, X, Y, n_partitions=10**9, dummy=True, key=jax.random.key(0))
    # The dummy default adapts to small N instead of raising over a bare 25.
    p_small, X_small, Y_small = _all_categorical_data(n=40, seed=5)
    res_small = optimal_transport.analyze(
        p_small, X_small, Y_small, dummy=True, key=jax.random.key(0)
    )
    assert res_small.ot_dummy is not None


def test_mixed_partition_bootstrap_cis_are_finite_and_ordered():
    problem, X, Y = _mixed_data(n=2000)
    res = borgonovo.analyze(problem, X, Y, n_bootstrap=32, key=jax.random.key(0))
    assert res.delta_conf is not None
    lo, hi = np.asarray(res.delta_conf)
    assert np.all(lo <= hi)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    res_o = optimal_transport.analyze(problem, X, Y, n_bootstrap=32, key=jax.random.key(0))
    assert res_o.ot_conf is not None
    lo_o, hi_o = np.asarray(res_o.ot_conf)
    assert np.all(lo_o <= hi_o)


# ---------------------------------------------------------------------------
# Sobol design + analysis
# ---------------------------------------------------------------------------


def test_sobol_saltelli_matches_analytic_reference():
    problem = _mixed_problem()
    sr = sobol.sample(problem, 2**13, seed=0, verbose=False)
    codes = sr.samples[:, 1].astype(int)
    assert np.array_equal(sr.samples[:, 1], codes)  # codes, never physical values
    Y = sr.samples[:, 0] + OFFSETS[codes]
    res = sobol.analyze(sr, Y)

    pr = np.array(PROBS)
    m = (pr * OFFSETS).sum()
    v_cat = (pr * (OFFSETS - m) ** 2).sum()
    v_y = v_cat + 1.0 / 12.0
    expected = np.array([(1.0 / 12.0) / v_y, v_cat / v_y])
    np.testing.assert_allclose(np.asarray(res.S1), expected, atol=5e-2)
    np.testing.assert_allclose(np.asarray(res.ST), expected, atol=5e-2)


def test_sobol_design_inflation_guard_warns_and_completes():
    p = Problem.from_dict(
        {
            "c1": {"dist": "categorical", "probs": [0.5, 0.5]},
            "c2": {"dist": "categorical", "probs": [0.3, 0.7]},
        }
    )
    with pytest.warns(UserWarning, match="possible distinct rows"):
        sr = sobol.sample(p, 1000, seed=0, verbose=False)
    assert sr.n_runs <= 4  # at most prod(L_d) distinct rows
    Y = sr.samples[:, 0] + 0.5 * sr.samples[:, 1]
    res = sobol.analyze(sr, Y)
    # V1 = Var(Bernoulli(0.5)) = 0.25; V2 = 0.25 * Var(Bernoulli(0.7)).
    v1, v2 = 0.25, 0.25 * 0.21
    np.testing.assert_allclose(np.asarray(res.S1), [v1 / (v1 + v2), v2 / (v1 + v2)], atol=5e-2)


def test_sobol_unique_rows_distort_the_marginal_but_expanded_does_not():
    """Pin the documented ``sr.samples`` caveat.

    ``sr.samples`` is a deduplicated evaluation set. Its empirical
    frequencies do not match the declared marginal; only the expanded
    design, which ``analyze`` reconstructs, carries the declared one.
    """
    p = Problem.from_dict({"c": {"dist": "categorical", "probs": [0.9, 0.1]}})
    with pytest.warns(UserWarning, match="possible distinct rows"):
        sr = sobol.sample(p, 2048, seed=0, verbose=False)

    unique_freq = np.bincount(sr.samples[:, 0].astype(int), minlength=2) / sr.n_runs
    expanded_codes = sr.samples[sr.expanded_to_unique, 0].astype(int)
    expanded_freq = np.bincount(expanded_codes, minlength=2) / sr.n_expanded

    np.testing.assert_allclose(expanded_freq, [0.9, 0.1], atol=0.01)
    assert abs(unique_freq[0] - 0.9) > 0.02


# ---------------------------------------------------------------------------
# Method gating
# ---------------------------------------------------------------------------


def test_categorical_design_error_is_honest_about_correlation():
    """The refusal must not point to a sampler that refuses the same problem.

    For a plain categorical problem the message may recommend
    jaxgsa.sobol.sample. For a categorical problem that also declares a
    correlation, sobol.sample would refuse too — the message must instead
    say that no variance-based route exists and name the given-data methods.
    """
    problem = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": PROBS},
        }
    )
    # kucherenko.sample is exempt from the correlated-design guard, so its
    # categorical refusal is where the combined case actually surfaces.
    with pytest.raises(ValueError, match=r"jaxgsa\.sobol\.sample"):
        jaxgsa.kucherenko.sample(problem, 64)

    R = np.eye(3)
    R[0, 1] = R[1, 0] = 0.5  # couples the continuous pair; identity row for c
    correlated = problem.with_correlation(R)
    with pytest.raises(ValueError, match="no variance-based method") as excinfo:
        jaxgsa.kucherenko.sample(correlated, 64)
    message = str(excinfo.value)
    assert "sobol.sample" not in message
    assert "optimal_transport" in message and "borgonovo" in message


def _categorical_and_correlated_problem():
    """A problem that is categorical AND correlated -- the combined dead end."""
    problem = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": (0.0, 1.0),
            "c": {"dist": "categorical", "probs": PROBS},
        }
    )
    R = np.eye(3)
    R[0, 1] = R[1, 0] = 0.5  # couples the continuous pair; identity row for c
    return problem.with_correlation(R)


def _categorical_and_correlated_data(n=128):
    """The combined-dead-end problem plus an aligned (X, Y) sample."""
    problem = _categorical_and_correlated_problem()
    X = jaxgsa.sampling.monte_carlo(problem, n, seed=0)
    codes = np.asarray(X)[:, 2].astype(int)
    Y = np.asarray(X)[:, 0] + OFFSETS[codes]
    return problem, X, Y


def test_combined_categorical_and_correlated_message():
    """Every design sampler must name the combined dead end, not one half.

    A correlated-only message would recommend jaxgsa.vkoga or
    jaxgsa.kucherenko, which then refuse the problem for being categorical.
    A categorical-only message would recommend jaxgsa.sobol.sample, which
    then refuses it for being correlated. Both gates route this case to the
    combined text, so the order the two checks run in does not matter.
    """
    problem = _categorical_and_correlated_problem()
    calls = {
        "morris": lambda: morris.sample(problem, 8),
        "efast": lambda: efast.sample(problem, 65),
        "sobol": lambda: sobol.sample(problem, 64, seed=0, verbose=False),
        "kucherenko": lambda: jaxgsa.kucherenko.sample(problem, 64),
    }
    for name, call in calls.items():
        with pytest.raises(ValueError, match="no variance-based method") as excinfo:
            call()
        message = str(excinfo.value)
        assert message.startswith(f"jaxgsa.{name}.sample ")
        assert "'c'" in message and "problem.correlation" in message
        advice = message.split("this problem:", 1)[1]
        assert "sobol.sample" not in advice
        assert "jaxgsa.vkoga" not in advice
        assert "jaxgsa.kucherenko" not in advice
        assert "jaxgsa.optimal_transport" in advice
        assert "jaxgsa.borgonovo" in advice
        assert "jaxgsa.pawn" in advice


def test_combined_message_on_the_analysis_side():
    """Every gated analyzer must name the combined dead end too.

    The analysis-side gates have the same fault the design-side ones had. A
    correlated-only message recommends jaxgsa.vkoga and jaxgsa.kucherenko,
    which refuse a categorical parameter. A categorical-only message
    recommends the design-based jaxgsa.sobol pipeline, which refuses a
    declared correlation. Both analysis gates route the combination to the
    shared message, so the order they run in does not matter. Note that the
    correlation-tolerant analyzers (hdmr, hsic, vkoga) reach it through
    the categorical gate, and the correlation-naive ones (dgsm, pce, shapley)
    through whichever gate their caller runs first.
    """
    problem, X, Y = _categorical_and_correlated_data()
    calls = {
        "dgsm": lambda: dgsm.analyze(problem, fn=lambda x: x[:, 0], X=X),
        "pce": lambda: pce_mod.analyze(problem, X, Y),
        "hdmr": lambda: hdmr.analyze(problem, X, Y),
        "hsic": lambda: hsic.analyze(problem, X, Y),
        "shapley": lambda: shapley.analyze(problem, X, Y),
        "vkoga": lambda: jaxgsa.vkoga.analyze(problem, X, Y),
    }
    for name, call in calls.items():
        with pytest.raises(ValueError, match="no variance-based method") as excinfo:
            call()
        message = str(excinfo.value)
        assert message.startswith(f"jaxgsa.{name}.analyze ")
        assert "'c'" in message and "problem.correlation" in message
        advice = message.split("this problem:", 1)[1]
        assert "sobol" not in advice
        assert "jaxgsa.vkoga" not in advice
        assert "jaxgsa.kucherenko" not in advice
        assert "jaxgsa.optimal_transport" in advice
        assert "jaxgsa.borgonovo" in advice
        assert "jaxgsa.pawn" in advice
        assert "jaxgsa.sampling.monte_carlo" not in advice


def test_combined_case_is_accepted_by_the_recommended_methods():
    """The advice must be true: every named method actually runs."""
    problem, X, Y = _categorical_and_correlated_data()
    for name, field in (("borgonovo", "delta"), ("optimal_transport", "ot"), ("pawn", "pawn")):
        kwargs = {"key": jax.random.key(0)} if name == "borgonovo" else {}
        result = getattr(jaxgsa, name).analyze(problem, X, Y, **kwargs)
        assert np.asarray(getattr(result, field)).shape == (3,)


def test_correlated_only_message_is_unchanged_without_categoricals():
    """A purely correlated problem keeps the correlated-only recommendation."""
    problem = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match="correlation-tolerant") as excinfo:
        morris.sample(problem.with_correlation(R), 8)
    assert "jaxgsa.vkoga" in str(excinfo.value)

    X = jaxgsa.sampling.monte_carlo(problem.with_correlation(R), 128, seed=0)
    Y = np.asarray(X)[:, 0]
    with pytest.raises(ValueError, match="correlation-tolerant") as excinfo:
        pce_mod.analyze(problem.with_correlation(R), X, Y)
    assert "jaxgsa.vkoga" in str(excinfo.value)


def test_categorical_only_analysis_message_is_unchanged():
    """A purely categorical problem keeps the categorical-only recommendation."""
    problem, X, Y = _mixed_data(n=256)
    with pytest.raises(ValueError, match="categorical-aware method") as excinfo:
        hsic.analyze(problem, X, Y)
    assert "jaxgsa.sobol" in str(excinfo.value)
