import numpy as np

from gsax.problem import Problem
from gsax.sampling import _next_power_of_2, sample


def test_next_power_of_2():
    assert _next_power_of_2(1) == 1
    assert _next_power_of_2(2) == 2
    assert _next_power_of_2(3) == 4
    assert _next_power_of_2(5) == 8
    assert _next_power_of_2(1024) == 1024
    assert _next_power_of_2(1025) == 2048


def test_sample_shape():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)})
    result = sample(p, n_samples=100, seed=42)
    D = 3
    expected_total = result.base_n * (2 * D + 2)
    assert result.samples.shape == (expected_total, D)
    assert result.n_params == D
    assert result.calc_second_order is True


def test_sample_within_bounds():
    p = Problem.from_dict({"x1": (-5.0, 5.0), "x2": (0.0, 10.0)})
    result = sample(p, n_samples=200, seed=42)
    assert np.all(result.samples[:, 0] >= -5.0)
    assert np.all(result.samples[:, 0] <= 5.0)
    assert np.all(result.samples[:, 1] >= 0.0)
    assert np.all(result.samples[:, 1] <= 10.0)


def test_power_of_2_enforcement():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=100, seed=42)
    # base_n should be power of 2
    assert result.base_n & (result.base_n - 1) == 0


def test_no_second_order():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=100, calc_second_order=False, seed=42)
    D = 2
    expected_total = result.base_n * (D + 2)
    assert result.samples.shape == (expected_total, D)
    assert result.calc_second_order is False
