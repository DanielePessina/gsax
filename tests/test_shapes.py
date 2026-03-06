import jax
import jax.numpy as jnp

import gsax
from gsax.problem import Problem


def test_2d_input_shape():
    """2D input (n_total, K) should squeeze time dimension."""
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, seed=42)
    K = 3
    Y = jnp.ones((sr.n_total, K))
    # Add some variation so variance isn't zero
    Y = Y + jax.random.normal(jax.random.key(1), (sr.n_total, K))
    result = gsax.analyze(sr, Y)
    assert result.S1.shape == (K, p.num_vars)
    assert result.ST.shape == (K, p.num_vars)
    assert result.S2.shape == (K, p.num_vars, p.num_vars)


def test_3d_input_shape():
    """3D input (n_total, T, K) should preserve both dimensions."""
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, seed=42)
    T, K = 2, 3
    Y = jax.random.normal(jax.random.key(1), (sr.n_total, T, K))
    result = gsax.analyze(sr, Y)
    assert result.S1.shape == (T, K, p.num_vars)
    assert result.ST.shape == (T, K, p.num_vars)
    assert result.S2.shape == (T, K, p.num_vars, p.num_vars)


def test_1d_input_shape():
    """1D input (n_total,) should squeeze both time and output dimensions."""
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, seed=42)
    Y = jax.random.normal(jax.random.key(1), (sr.n_total,))
    result = gsax.analyze(sr, Y)
    assert result.S1.shape == (p.num_vars,)
    assert result.ST.shape == (p.num_vars,)
    assert result.S2.shape == (p.num_vars, p.num_vars)


def test_single_param():
    """Single parameter should produce scalar indices, no S2."""
    p = Problem.from_dict({"x1": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, calc_second_order=False, seed=42)
    Y = jax.random.normal(jax.random.key(1), (sr.n_total,))
    result = gsax.analyze(sr, Y)
    assert result.S1.shape == (p.num_vars,)
    assert result.ST.shape == (p.num_vars,)
    assert result.S2 is None


def test_single_output():
    """Single output (K=1) should still have correct shapes."""
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sr = gsax.sample(p, n_samples=256, seed=42)
    Y = jax.random.normal(jax.random.key(1), (sr.n_total, 1))
    result = gsax.analyze(sr, Y)
    assert result.S1.shape == (1, p.num_vars)
    assert result.ST.shape == (1, p.num_vars)
