# gsax

**Global Sensitivity Analysis in JAX**

`gsax` computes Sobol variance-based sensitivity indices entirely in JAX, giving you GPU/TPU acceleration and JIT compilation for free. It implements Saltelli's sampling scheme with Sobol quasi-random sequences and supports first-order, total-order, and second-order indices with chunked vectorization for bounded memory usage.

## Features

- Saltelli sampling via Sobol quasi-random sequences (powered by `scipy.stats.qmc`)
- First-order (S1), total-order (ST), and second-order (S2) Sobol indices
- Chunked `jit(vmap(...))` execution for bounded memory on large output grids
- Supports scalar, multi-output, and time-series model outputs
- Bootstrap confidence intervals with JAX-accelerated resampling
- Automatic data cleaning: non-finite values (NaN/Inf) are detected and dropped by group
- Diagnostic `nan_counts` on results for post-hoc quality checks
- Built-in Ishigami benchmark function with known analytical solutions
- ~458x faster than SALib on multi-output time-series analysis; ~14.5x faster on bootstrap

## Installation

```bash
pip install gsax
```

Or for development:

```bash
git clone <repo-url>
cd gsax
pip install -e ".[dev]"
```

## Quick Start

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# 1. Generate Saltelli samples
sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)
# sampling_result.samples.shape == (n_total, D)  where D = 3

# 2. Evaluate your model on the samples
Y = evaluate(sampling_result.samples)  # Y.shape == (n_total,)

# 3. Compute Sobol indices
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (D,)    — first-order indices
# result.ST.shape == (D,)    — total-order indices
# result.S2.shape == (D, D)  — second-order interaction matrix

print("First-order indices (S1):", result.S1)
print("Total-order indices (ST):", result.ST)
print("Second-order indices (S2):")
print(result.S2)
```

Expected output (Ishigami function with A=7, B=0.1):

```
First-order indices (S1): [~0.31, ~0.44, ~0.00]
Total-order indices (ST): [~0.56, ~0.44, ~0.24]
```

## Usage

### Define a problem

A `Problem` specifies the parameter names and their bounds:

```python
from gsax import Problem

# From a dictionary
problem = Problem.from_dict({
    "x1": (-3.14, 3.14),
    "x2": (-3.14, 3.14),
    "x3": (-3.14, 3.14),
})

# Or directly
problem = Problem(
    names=("x1", "x2", "x3"),
    bounds=((-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)),
)
```

### Generate samples

```python
sampling_result = gsax.sample(
    problem,
    n_samples=4096,          # minimum desired model evaluations
    calc_second_order=True,  # include second-order indices (default)
    scramble=True,           # scramble Sobol sequence (default)
    seed=42,                 # reproducibility
)

# sampling_result.samples is a NumPy array of shape (n_total, D)
# Pass it to your model
```

### Analyze results

```python
# Y can be:
#   - (n_total,)       scalar output
#   - (n_total, K)     multi-output (K outputs)
#   - (n_total, T, K)  time-series multi-output
Y = my_model(sampling_result.samples)

result = gsax.analyze(
    sampling_result,
    Y,
    chunk_size=64,  # optional: limit vmap batch size for memory control
)

# result.S1, result.ST — sensitivity indices
# result.S2            — second-order interactions (None if not computed)
```

### Multi-output models

For models with multiple outputs, pass a 2D array `(n_total, K)`. The returned indices will have shape `(K, D)`:

```python
import jax.numpy as jnp

def multi_output_model(X):
    y1 = jnp.sin(X[:, 0]) + X[:, 1] ** 2
    y2 = X[:, 0] * X[:, 2]
    return jnp.column_stack([y1, y2])

Y = multi_output_model(sampling_result.samples)  # (n_total, 2)
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (2, 3)  — 2 outputs, 3 parameters (K, D)
# result.ST.shape == (2, 3)  — (K, D)
# result.S2.shape == (2, 3, 3)  — (K, D, D)
```

For time-series multi-output models, pass a 3D array `(n_total, T, K)`:

```python
def time_series_model(X):
    # Returns shape (n_total, T, K) — e.g. 50 timesteps, 4 outputs
    ...

Y = time_series_model(sampling_result.samples)  # (n_total, 50, 4)
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (50, 4, D)  — (T, K, D)
# result.ST.shape == (50, 4, D)  — (T, K, D)
# result.S2.shape == (50, 4, D, D)  — (T, K, D, D)
```

---

## API Reference

### `Problem`

Immutable dataclass defining the parameter space.

```python
@dataclass(frozen=True)
class Problem:
    names: tuple[str, ...]                    # parameter names
    bounds: tuple[tuple[float, float], ...]   # (low, high) for each parameter
```

| Attribute / Method | Type | Description |
|---|---|---|
| `names` | `tuple[str, ...]` | Parameter name for each of the D dimensions. |
| `bounds` | `tuple[tuple[float, float], ...]` | Lower and upper bound for each parameter. Length must match `names`. |
| `from_dict(params)` | classmethod | Construct from `dict[str, tuple[float, float]]`. Keys become `names`, values become `bounds`. |
| `num_vars` | `int` (property) | D -- the number of parameters. Equivalent to `len(names)`. |

### `gsax.sample()`

Generate a Saltelli sample matrix using Sobol quasi-random sequences.

```python
def sample(
    problem: Problem,
    n_samples: int,               # minimum desired model evaluations
    *,
    calc_second_order: bool = True,
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
) -> SamplingResult
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `problem` | `Problem` | required | The parameter space definition. |
| `n_samples` | `int` | required | Minimum number of model evaluations desired. The actual `base_n` (N) will be rounded up to the next power of 2. |
| `calc_second_order` | `bool` | `True` | Whether to generate the extra sample matrices needed for second-order indices. When `True`, `n_total = N * (2D + 2)`. When `False`, `n_total = N * (D + 2)`. |
| `scramble` | `bool` | `True` | Apply Owen scrambling to the Sobol sequence for better uniformity. |
| `seed` | `int \| np.random.Generator \| None` | `None` | Seed for reproducibility of the scrambled Sobol sequence. |

**Returns:** `SamplingResult`

### `SamplingResult`

Immutable dataclass returned by `gsax.sample()`. Carries the sample matrix and all metadata needed by `gsax.analyze()`.

```python
@dataclass(frozen=True)
class SamplingResult:
    samples: np.ndarray       # (n_total, D) — scaled to parameter bounds
    base_n: int               # N, always a power of 2
    n_params: int             # D = number of parameters
    calc_second_order: bool   # whether S2 matrices were generated
    problem: Problem          # the Problem used to generate samples
```

| Field / Property | Type | Shape / Value | Description |
|---|---|---|---|
| `samples` | `np.ndarray` | `(n_total, D)` | Sample matrix with values scaled to the parameter bounds defined in `problem`. Pass this to your model. |
| `base_n` | `int` | N | The base sample count, always a power of 2 >= the requested `n_samples` divided by the step multiplier. |
| `n_params` | `int` | D | Number of parameters (same as `problem.num_vars`). |
| `calc_second_order` | `bool` | | Whether second-order cross-matrices are included in `samples`. |
| `problem` | `Problem` | | The Problem instance used during sampling. |
| `n_total` | `int` (property) | `base_n * step` | Total number of rows in `samples`. The step is `2D + 2` (second-order) or `D + 2` (first/total only). |

### `gsax.analyze()`

Compute Sobol sensitivity indices from model output.

```python
def analyze(
    sampling_result: SamplingResult,
    Y: Array,                    # model output
    *,
    num_resamples: int = 0,      # 0 = no bootstrap
    conf_level: float = 0.95,
    key: Array | None = None,    # JAX PRNG key, required when num_resamples > 0
    chunk_size: int = 2048,
) -> SAResult
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sampling_result` | `SamplingResult` | required | The result from `gsax.sample()`. |
| `Y` | `Array` | required | Model output. Shape must be `(n_total,)`, `(n_total, K)`, or `(n_total, T, K)` where `n_total` matches `sampling_result.n_total`. |
| `num_resamples` | `int` | `0` | Number of bootstrap resamples. Set to 0 to skip bootstrap (no confidence intervals). |
| `conf_level` | `float` | `0.95` | Confidence level for bootstrap intervals (e.g. 0.95 for 95% CI). |
| `key` | `Array \| None` | `None` | JAX PRNG key (e.g. `jax.random.key(0)`). **Required** when `num_resamples > 0`. |
| `chunk_size` | `int` | `2048` | Number of output columns to process per `jit(vmap(...))` call. Lower values reduce peak memory; higher values improve throughput. |

**Returns:** `SAResult`

### `SAResult`

Dataclass holding all computed sensitivity indices and optional confidence intervals.

```python
@dataclass
class SAResult:
    S1: Array                          # first-order Sobol indices
    ST: Array                          # total-order Sobol indices
    S2: Array | None                   # second-order interaction matrix
    problem: Problem                   # the Problem definition
    S1_conf: Array | None = None       # bootstrap CI bounds for S1
    ST_conf: Array | None = None       # bootstrap CI bounds for ST
    S2_conf: Array | None = None       # bootstrap CI bounds for S2
    nan_counts: dict[str, int] | None = None  # diagnostic NaN counts
```

See the next sections for detailed field documentation and shape rules.

---

## Understanding Output Shapes

The shapes of all arrays in `SAResult` are determined by the shape of `Y` passed to `gsax.analyze()`. Let **D** = number of parameters, **K** = number of outputs, **T** = number of timesteps.

### Shape mapping table

| Y shape | S1 / ST shape | S2 shape | S1_conf / ST_conf shape | S2_conf shape |
|---|---|---|---|---|
| `(n_total,)` | `(D,)` | `(D, D)` | `(2, D)` | `(2, D, D)` |
| `(n_total, K)` | `(K, D)` | `(K, D, D)` | `(2, K, D)` | `(2, K, D, D)` |
| `(n_total, T, K)` | `(T, K, D)` | `(T, K, D, D)` | `(2, T, K, D)` | `(2, T, K, D, D)` |

**Key observations:**
- The parameter dimension **D** is always the last axis of S1 and ST.
- S2 appends an extra **D** axis for the interaction pair, so it is always `(..., D, D)`.
- Confidence interval arrays always prepend a leading dimension of **2**, where index `[0]` is the lower bound and index `[1]` is the upper bound.
- S2 is `None` when `calc_second_order=False`. Similarly, all `*_conf` fields are `None` when `num_resamples=0`.

### Example: reading results

```python
# Scalar output — Y.shape == (n_total,), D == 3
result.S1[0]          # first-order index for parameter 0
result.ST[2]          # total-order index for parameter 2
result.S2[0, 1]       # second-order interaction between params 0 and 1

# Multi-output — Y.shape == (n_total, K=4), D == 3
result.S1[2, 0]       # first-order index for output 2, parameter 0
result.S2[1, 0, 2]    # second-order interaction (params 0,2) for output 1

# Time-series — Y.shape == (n_total, T=50, K=4), D == 3
result.S1[10, 2, 0]   # S1 at timestep 10, output 2, parameter 0
result.S2[10, 2, 0, 1]  # S2 interaction (params 0,1) at timestep 10, output 2
```

---

## SAResult Fields

Detailed field-by-field documentation.

### `S1` -- First-order Sobol indices

The fraction of output variance attributable to each input parameter alone (excluding interactions). Values range from 0 to 1. A higher S1 means the parameter has a stronger direct (additive) effect.

- **Shape:** `(D,)`, `(K, D)`, or `(T, K, D)` depending on Y.

### `ST` -- Total-order Sobol indices

The fraction of output variance attributable to each input parameter including all interactions with other parameters. Always `ST[i] >= S1[i]`. The gap `ST[i] - S1[i]` indicates how much of the parameter's influence comes through interactions.

- **Shape:** same as S1.

### `S2` -- Second-order interaction indices

The fraction of output variance attributable to the interaction between each pair of parameters, beyond their individual first-order effects. The matrix is **symmetric** (`S2[..., i, j] == S2[..., j, i]`) and the **diagonal is NaN** (the interaction of a parameter with itself is undefined). Only the upper triangle is estimated directly; the lower triangle mirrors it.

- **Shape:** `(D, D)`, `(K, D, D)`, or `(T, K, D, D)`.
- **Value:** `None` if `calc_second_order=False` was used in `gsax.sample()`.

### `S1_conf`, `ST_conf`, `S2_conf` -- Bootstrap confidence intervals

Each confidence array has the same shape as the corresponding index array, but with an extra leading dimension of 2:
- `[0, ...]` = lower bound of the confidence interval
- `[1, ...]` = upper bound of the confidence interval

For example, if `S1.shape == (K, D)`, then `S1_conf.shape == (2, K, D)`.

- **Value:** `None` when `num_resamples=0` (no bootstrap requested).

### `problem` -- Problem definition

The `Problem` instance that was used, carried through for convenience.

### `nan_counts` -- Diagnostic NaN counts

A dictionary reporting how many NaN values appear in each index array after computation. Useful for detecting edge cases such as constant outputs or degenerate sample groups.

```python
result.nan_counts
# e.g. {"S1": 0, "ST": 0, "S2": 1}
```

- **Value:** `None` only in unusual circumstances; normally always populated.

---

## Bootstrap Confidence Intervals

To quantify uncertainty in the estimated indices, use bootstrap resampling:

```python
import jax

result = gsax.analyze(
    sampling_result,
    Y,
    num_resamples=200,           # number of bootstrap resamples (R)
    conf_level=0.95,             # 95% confidence interval
    key=jax.random.key(0),      # JAX PRNG key (required for bootstrap)
)

# Access confidence intervals
lower_S1 = result.S1_conf[0]    # lower bound, same shape as result.S1
upper_S1 = result.S1_conf[1]    # upper bound, same shape as result.S1

lower_ST = result.ST_conf[0]
upper_ST = result.ST_conf[1]

# For S2 (if calc_second_order=True):
lower_S2 = result.S2_conf[0]    # lower bound, same shape as result.S2
upper_S2 = result.S2_conf[1]    # upper bound, same shape as result.S2
```

The bootstrap is fully vectorized in JAX, making it significantly faster than sequential resampling approaches. With `R=200` resamples on a multi-output time-series problem (D=5, T=100, K=4), gsax is ~14.5x faster than SALib.

**When to use bootstrap:** Bootstrap is recommended when you need to report uncertainty bounds or assess convergence. For exploratory analysis where you only need point estimates, set `num_resamples=0` (the default) to skip it entirely.

---

## Data Cleaning

`gsax.analyze()` automatically handles non-finite values (NaN, Inf, -Inf) in the model output `Y`:

1. **Group-based removal:** Saltelli's method structures `Y` into groups of `step` rows (where `step = 2D + 2` or `D + 2`). If *any* element within a group is non-finite, the **entire group** is dropped. This preserves the mathematical structure required by the estimator.

2. **Informational message:** When groups are dropped, gsax prints a message indicating how many groups were removed out of the total.

3. **Error on total failure:** If *all* groups contain non-finite values, a `ValueError` is raised.

4. **Post-analysis diagnostics:** The `nan_counts` field on `SAResult` reports how many NaN values remain in the computed indices. This can happen legitimately (e.g., S2 diagonal is always NaN, or a constant output produces undefined indices).

```python
# Example: model that occasionally returns NaN
Y = my_flaky_model(sampling_result.samples)
# Y may contain some NaN values

result = gsax.analyze(sampling_result, Y)
# gsax: dropped 3 of 1024 sample groups containing non-finite values

# Check diagnostics
print(result.nan_counts)  # {"S1": 0, "ST": 0, "S2": 3}  (3 = diagonal NaNs for D=3)
```

---

## Dependencies

- `jax >= 0.4`
- `jaxlib >= 0.4`
- `scipy >= 1.10`

## License

See [LICENSE](LICENSE) for details.

## Benchmark Results

The benchmark script (`benchmark_salib.py`) validates correctness against SALib and measures performance on a coupled-oscillator model with D=5 parameters, T=100 timesteps, and K=4 outputs. Key findings:

- **Correctness:** S1, ST, and S2 match SALib to `atol=1e-6` on the Ishigami function.
- **Analysis speed:** gsax is **~458x faster** than SALib for multi-output time-series analysis. SALib must loop `analyze()` over every (T, K) slice; gsax vectorizes across all slices in a single JIT-compiled pass.
- **Bootstrap speed:** gsax bootstrap with R=200 resamples is **~14.5x faster** than SALib's bootstrap.

```bash
(base) danielepessina@MacBookPro gsax % uv run benchmark_salib.py
======================================================================
CORRECTNESS CHECK  (Ishigami, shared samples)
======================================================================
  S1     match (atol=1e-06): PASS
  ST     match (atol=1e-06): PASS
  S2     match (atol=1e-06): PASS

======================================================================
TIMING BENCHMARK — coupled oscillators
  D=5, T=100, K=4, base_n=4096, n_total=49152
  SALib must call analyze() 100x4 = 400 times
  n_repeats=1
======================================================================

Warming up gsax JIT ... done.
/Users/danielepessina/Documents/Local Uni/gsax/.venv/lib/python3.12/site-packages/SALib/analyze/sobol.py:141: RuntimeWarning: invalid value encountered in divide
  Y = (Y - Y.mean()) / Y.std()

Phase               gsax (ms)     SALib (ms)    speedup
------------------------------------------------------
  sample               26.4           39.4        1.5x
  evaluate             67.6          237.3        3.5x
  analyze             169.0        77455.1      458.4x
------------------------------------------------------
  total               262.9        77731.8      295.7x

======================================================================
BOOTSTRAP BENCHMARK — coupled oscillators
  D=5, T=100, K=4, base_n=4096, R=200
  n_repeats=1
======================================================================

Warming up gsax bootstrap JIT ... done.

Method                            Time (ms)   vs gsax-noboot
------------------------------------------------------------
  gsax (no bootstrap)               164.7             1.0x
  gsax (bootstrap R=200)           6861.6            41.7x
  SALib (bootstrap R=200)         99252.8           602.5x

  gsax bootstrap speedup vs SALib: 14.5x

======================================================================
ALL CORRECTNESS CHECKS PASSED
(base) danielepessina@MacBookPro gsax %
```
