# Analyze (Sobol)

## `gsax.analyze()`

Compute Sobol sensitivity indices from model output.

```python
def analyze(
    sampling_result: SamplingResult,
    Y: Array,
    *,
    num_resamples: int = 0,
    conf_level: float = 0.95,
    key: Array | None = None,
    chunk_size: int = 2048,
) -> SAResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampling_result` | `SamplingResult` | required | Result from `gsax.sample()`. |
| `Y` | `Array` | required | Model output evaluated on the unique rows in `sampling_result.samples`. Shape: `(n_total,)`, `(n_total, K)`, or `(n_total, T, K)`. |
| `num_resamples` | `int` | `0` | Number of bootstrap resamples. 0 = no bootstrap. |
| `conf_level` | `float` | `0.95` | Confidence level for bootstrap intervals. |
| `key` | `Array \| None` | `None` | JAX PRNG key. Required when `num_resamples > 0`. |
| `chunk_size` | `int` | `2048` | Output columns per `jit(vmap(...))` call. Lower = less memory, higher = more throughput. |

**Returns:** [`SAResult`](#saresult)

---

## `SAResult`

Dataclass holding all computed sensitivity indices and optional confidence intervals.

```python
@dataclass
class SAResult:
    S1: Array
    ST: Array
    S2: Array | None
    problem: Problem
    S1_conf: Array | None = None
    ST_conf: Array | None = None
    S2_conf: Array | None = None
    nan_counts: dict[str, int] | None = None
```

### Fields

#### `S1` -- First-order Sobol indices

Fraction of output variance due to each parameter alone (excluding interactions). Values in [0, 1].

- **Shape:** `(D,)`, `(K, D)`, or `(T, K, D)`

#### `ST` -- Total-order Sobol indices

Fraction of output variance due to each parameter including all interactions. Always `ST[i] >= S1[i]`.

- **Shape:** same as S1

#### `S2` -- Second-order interaction indices

Variance fraction from pairwise interactions beyond individual first-order effects. Symmetric matrix; diagonal is NaN.

- **Shape:** `(D, D)`, `(K, D, D)`, or `(T, K, D, D)`
- `None` if `calc_second_order=False`

#### `S1_conf`, `ST_conf`, `S2_conf` -- Bootstrap confidence intervals

Same shape as the corresponding index array with a leading dimension of 2: `[0]` = lower, `[1]` = upper.

- `None` when `num_resamples=0`

#### `nan_counts`

Dictionary reporting NaN counts in each index array. Useful for diagnosing constant outputs or degenerate groups.

### `to_dataset(time_coords=None)`

Convert to a labeled `xarray.Dataset`. See the [xarray example](/examples/xarray) for full usage.

```python
ds = result.to_dataset()
ds.S1.sel(param="x1")
ds.S1.sel(param="x1", output="temperature")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_coords` | `list \| np.ndarray \| None` | `None` | Coordinate values for the time dimension (3-D results only). Defaults to integer indices. |

**Dimensions:** `param`, `output` (multi-output), `time` (3-D), `param_i`/`param_j` (S2). Output names come from `problem.output_names` or default to `y0, y1, ...`. Confidence intervals are split into `S1_lower`/`S1_upper`, `ST_lower`/`ST_upper`, `S2_lower`/`S2_upper`.

### Output Shape Reference

| Y shape | S1 / ST | S2 | S1_conf / ST_conf | S2_conf |
|---------|---------|-----|-------------------|---------|
| `(N,)` | `(D,)` | `(D, D)` | `(2, D)` | `(2, D, D)` |
| `(N, K)` | `(K, D)` | `(K, D, D)` | `(2, K, D)` | `(2, K, D, D)` |
| `(N, T, K)` | `(T, K, D)` | `(T, K, D, D)` | `(2, T, K, D)` | `(2, T, K, D, D)` |
