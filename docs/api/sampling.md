# Sampling

## `gsax.sample()`

Generate a unique Sobol/Saltelli sample matrix using Sobol quasi-random sequences.

```python
def sample(
    problem: Problem,
    n_samples: int,
    *,
    calc_second_order: bool = True,
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    verbose: bool = True,
) -> SamplingResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem` | `Problem` | required | The parameter space definition. |
| `n_samples` | `int` | required | Minimum number of unique model evaluations desired. The actual `base_n` (N) is increased until the deduplicated sample matrix has at least this many rows. |
| `calc_second_order` | `bool` | `True` | Whether to generate the extra Saltelli cross-matrices needed for second-order indices. When `True`, `expanded_n_total = N * (2D + 2)`. When `False`, `expanded_n_total = N * (D + 2)`. |
| `scramble` | `bool` | `True` | Apply Owen scrambling to the Sobol sequence for better uniformity. |
| `seed` | `int \| Generator \| None` | `None` | Seed for reproducibility. |
| `verbose` | `bool` | `True` | Print a compact summary showing unique vs expanded row counts. |

**Returns:** [`SamplingResult`](#samplingresult)

---

## `SamplingResult`

Immutable dataclass returned by `gsax.sample()`. Carries the unique sample matrix and all metadata needed by `gsax.analyze()` to reconstruct the expanded Saltelli layout internally.

```python
@dataclass(frozen=True)
class SamplingResult:
    samples: np.ndarray
    sample_ids: np.ndarray
    expanded_n_total: int
    expanded_to_unique: np.ndarray
    base_n: int
    n_params: int
    calc_second_order: bool
    problem: Problem
```

### Fields

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `samples` | `np.ndarray` | `(n_total, D)` | Unique sample matrix scaled to parameter bounds. Pass this to your model. |
| `sample_ids` | `np.ndarray` | `(n_total,)` | Stable integer IDs `0..n_total-1` aligned with `samples`. |
| `expanded_n_total` | `int` | `N * step` | Total Saltelli row count used internally by `analyze()`. |
| `expanded_to_unique` | `np.ndarray` | `(expanded_n_total,)` | Index map from each expanded Saltelli row to the corresponding unique row in `samples`. |
| `base_n` | `int` | N | Base Sobol sample count, always a power of 2. |
| `n_params` | `int` | D | Number of parameters. |
| `calc_second_order` | `bool` | | Whether second-order cross-matrices are included. |
| `problem` | `Problem` | | The Problem instance used during sampling. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_total` | `int` | Number of unique rows: `samples.shape[0]`. |
| `samples_df` | `pd.DataFrame` | Tabular view with `SampleID` followed by parameter columns. |
