import numpy as np
from scipy.stats import truncnorm

from gsax.problem import GaussianInputSpec, Problem, UniformInputSpec
from gsax.sampling import _next_power_of_2, _saltelli_step, sample


def test_next_power_of_2():
    assert _next_power_of_2(1) == 1
    assert _next_power_of_2(2) == 2
    assert _next_power_of_2(3) == 4
    assert _next_power_of_2(5) == 8
    assert _next_power_of_2(1024) == 1024
    assert _next_power_of_2(1025) == 2048


def test_sample_returns_unique_rows():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0), "x3": (0.0, 1.0)})
    result = sample(p, n_samples=100, seed=42, verbose=False)
    assert result.n_total >= 100
    assert result.samples.shape == (result.n_total, p.num_vars)
    assert np.unique(result.samples, axis=0).shape[0] == result.n_total
    assert result.sample_ids.tolist() == list(range(result.n_total))
    assert result.expanded_n_total == result.base_n * _saltelli_step(p.num_vars, True)
    assert result.expanded_to_unique.shape == (result.expanded_n_total,)
    assert result.expanded_to_unique.max() < result.n_total
    assert result.n_params == p.num_vars
    assert result.calc_second_order is True


def test_sample_within_bounds():
    p = Problem.from_dict({"x1": (-5.0, 5.0), "x2": (0.0, 10.0)})
    result = sample(p, n_samples=200, seed=42, verbose=False)
    assert np.all(result.samples[:, 0] >= -5.0)
    assert np.all(result.samples[:, 0] <= 5.0)
    assert np.all(result.samples[:, 1] >= 0.0)
    assert np.all(result.samples[:, 1] <= 10.0)


def test_power_of_2_enforcement():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=100, seed=42, verbose=False)
    assert result.base_n & (result.base_n - 1) == 0


def test_samples_df_contains_sample_id_and_named_columns():
    p = Problem.from_dict({"alpha": (0.0, 1.0), "beta": (2.0, 3.0)})
    result = sample(p, n_samples=32, seed=42, verbose=False)
    df = result.samples_df
    assert list(df.columns) == ["SampleID", "alpha", "beta"]
    assert df["SampleID"].tolist() == result.sample_ids.tolist()
    assert np.allclose(df[["alpha", "beta"]].to_numpy(), result.samples)


def test_no_second_order_expanded_count():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=100, calc_second_order=False, seed=42, verbose=False)
    assert result.calc_second_order is False
    assert result.expanded_n_total == result.base_n * _saltelli_step(p.num_vars, False)


def test_single_parameter_mapping_collapses_duplicates():
    p = Problem.from_dict({"x1": (0.0, 1.0)})

    first_only = sample(p, n_samples=16, calc_second_order=False, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, False)
    for i in range(first_only.base_n):
        group = first_only.expanded_to_unique[i * step : (i + 1) * step]
        assert group[1] == group[2]
        assert group[0] != group[1]

    second_order = sample(p, n_samples=16, calc_second_order=True, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, True)
    for i in range(second_order.base_n):
        group = second_order.expanded_to_unique[i * step : (i + 1) * step]
        assert group[0] == group[2]
        assert group[1] == group[3]
        assert group[0] != group[1]


def test_two_parameter_second_order_mapping_collapses_cross_duplicates():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=32, calc_second_order=True, seed=42, verbose=False)
    step = _saltelli_step(p.num_vars, True)

    for i in range(result.base_n):
        group = result.expanded_to_unique[i * step : (i + 1) * step]
        assert group[1] == group[4]
        assert group[2] == group[3]
        assert len(set(group.tolist())) == 4


def test_reconstructing_expanded_samples_matches_mapping():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    result = sample(p, n_samples=64, seed=42, verbose=False)
    reconstructed = result.samples[result.expanded_to_unique]
    assert reconstructed.shape == (result.expanded_n_total, p.num_vars)
    assert np.unique(reconstructed, axis=0).shape[0] == result.n_total


def test_sample_verbose_prints_summary(capsys):
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sample(p, n_samples=32, seed=42, verbose=True)
    out = capsys.readouterr().out
    assert "gsax.sample:" in out
    assert "requested_unique>=" in out
    assert "returned_unique=" in out
    assert "duplicates_removed=" in out


def test_sample_verbose_false_is_silent(capsys):
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (0.0, 1.0)})
    sample(p, n_samples=32, seed=42, verbose=False)
    out = capsys.readouterr().out
    assert out == ""


def test_mixed_distributions_preserve_sampling_metadata():
    p = Problem.from_dict(
        {
            "x1": (0.0, 1.0),
            "x2": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
            "x3": GaussianInputSpec(dist="gaussian", mean=1.0, variance=4.0, low=-2.0),
        }
    )

    result = sample(p, n_samples=128, calc_second_order=False, seed=42, verbose=False)
    assert result.n_total >= 128
    assert result.samples.shape == (result.n_total, p.num_vars)
    assert result.expanded_n_total == result.base_n * _saltelli_step(p.num_vars, False)
    assert result.expanded_to_unique.shape == (result.expanded_n_total,)
    assert result.problem.has_non_uniform_inputs is True


def test_uniform_columns_stay_within_bounds_for_mixed_problem():
    p = Problem.from_dict(
        {
            "uniform": UniformInputSpec(dist="uniform", low=-3.0, high=2.0),
            "gaussian": GaussianInputSpec(dist="gaussian", mean=0.0, variance=1.0),
        }
    )

    result = sample(p, n_samples=256, seed=1, verbose=False)
    assert np.all(result.samples[:, 0] >= -3.0)
    assert np.all(result.samples[:, 0] <= 2.0)


def test_gaussian_column_matches_target_mean_and_variance():
    p = Problem.from_dict(
        {
            "x1": GaussianInputSpec(dist="gaussian", mean=1.5, variance=2.25),
            "x2": (0.0, 1.0),
        }
    )

    result = sample(p, n_samples=4096, calc_second_order=False, seed=123, verbose=False)
    gaussian = result.samples[:, 0]

    assert abs(np.mean(gaussian) - 1.5) < 0.05
    assert abs(np.var(gaussian) - 2.25) < 0.08


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


def test_two_sided_truncated_gaussian_matches_target_variance_formula():
    p = Problem.from_dict(
        {
            "x": GaussianInputSpec(
                dist="gaussian",
                mean=0.5,
                variance=1.44,
                low=-0.5,
                high=1.5,
            )
        }
    )

    result = sample(p, n_samples=4096, calc_second_order=False, seed=99, verbose=False)
    observed = np.var(result.samples[:, 0])
    std = np.sqrt(1.44)
    a = (-0.5 - 0.5) / std
    b = (1.5 - 0.5) / std
    expected = truncnorm.var(a, b, loc=0.5, scale=std)
    assert abs(observed - expected) < 0.03
