"""End-to-end checks for the Sobol estimator menu.

The analytical benchmark fixtures are the oracle. External estimator parity
tests are intentionally not part of the ordinary suite: they were slow,
dependency-sensitive, and repeated the same convention checks.
"""

from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

import jaxgsa
from jaxgsa.benchmarks import ishigami, sobol_g
from jaxgsa.sobol._estimators import DEFAULT_ESTIMATOR, ESTIMATORS, Estimator
from jaxgsa.sobol._result import SobolResult


def _regular(result: Any) -> SobolResult:
    """Narrow a rectangular-output result for type checking."""
    assert isinstance(result, SobolResult)
    return result


@pytest.fixture(scope="module")
def ishigami_design():
    """A single deterministic design shared by the estimator cases."""
    samples = jaxgsa.sobol.sample(
        ishigami.PROBLEM, n_samples=1, base_n=2048, seed=11, verbose=False
    )
    return samples, ishigami.evaluate(jnp.asarray(samples.samples))


@pytest.fixture(scope="module")
def sobol_g_design():
    """A deterministic Sobol-G fixture with known indices."""
    samples = jaxgsa.sobol.sample(
        sobol_g.PROBLEM, n_samples=1, base_n=2048, seed=13, verbose=False
    )
    return samples, sobol_g.evaluate(jnp.asarray(samples.samples))


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_every_estimator_reaches_the_ishigami_fixture(ishigami_design, estimator: Estimator):
    samples, outputs = ishigami_design
    result = _regular(jaxgsa.sobol.analyze(samples, outputs, estimator=estimator, verbose=False))
    np.testing.assert_allclose(result.S1, ishigami.ANALYTICAL_S1, atol=0.06)
    np.testing.assert_allclose(result.ST, ishigami.ANALYTICAL_ST, atol=0.06)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_every_estimator_reaches_the_sobol_g_fixture(sobol_g_design, estimator: Estimator):
    samples, outputs = sobol_g_design
    result = _regular(jaxgsa.sobol.analyze(samples, outputs, estimator=estimator, verbose=False))
    np.testing.assert_allclose(result.S1, sobol_g.ANALYTICAL_S1, atol=0.08)
    np.testing.assert_allclose(result.ST, sobol_g.ANALYTICAL_ST, atol=0.08)


def test_second_order_interaction_is_preserved(ishigami_design):
    samples, outputs = ishigami_design
    result = _regular(jaxgsa.sobol.analyze(samples, outputs, verbose=False))
    assert result.S2 is not None
    np.testing.assert_allclose(result.S2[0, 2], ishigami.ANALYTICAL_S2[0, 2], atol=0.06)


def test_default_estimator_is_explicit_and_recorded(ishigami_design):
    samples, outputs = ishigami_design
    implicit = _regular(jaxgsa.sobol.analyze(samples, outputs, verbose=False))
    explicit = _regular(
        jaxgsa.sobol.analyze(samples, outputs, estimator=DEFAULT_ESTIMATOR, verbose=False)
    )
    assert DEFAULT_ESTIMATOR == "saltelli-jansen"
    assert implicit.estimator == DEFAULT_ESTIMATOR
    np.testing.assert_array_equal(implicit.S1, explicit.S1)
    np.testing.assert_array_equal(implicit.ST, explicit.ST)


def test_unknown_estimator_is_rejected(ishigami_design):
    samples, outputs = ishigami_design
    with pytest.raises(ValueError, match="estimator must be one of"):
        jaxgsa.sobol.analyze(
            samples, outputs, estimator=cast(Estimator, "not-an-estimator"), verbose=False
        )


def test_azzini_rosati_enforces_samplewise_order():
    samples = jaxgsa.sobol.sample(sobol_g.PROBLEM, n_samples=1, base_n=32, seed=0, verbose=False)
    outputs = sobol_g.evaluate(jnp.asarray(samples.samples))
    result = _regular(
        jaxgsa.sobol.analyze(samples, outputs, estimator="azzini-rosati", verbose=False)
    )
    assert np.all(np.asarray(result.S1) <= np.asarray(result.ST) + 1e-6)
