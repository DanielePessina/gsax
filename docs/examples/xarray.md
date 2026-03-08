# xarray Labeled Output

gsax results can be converted to labeled `xarray.Dataset` objects for intuitive access by parameter name, output name, and time coordinate.

## Basic Usage

```python
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)
Y = evaluate(sampling_result.samples)
result = gsax.analyze(sampling_result, Y)

# Convert to xarray Dataset
ds = result.to_dataset()
print(ds)
# <xarray.Dataset>
# Dimensions:  (param: 3, param_i: 3, param_j: 3)
# Coordinates:
#   * param     (param) <U2 'x1' 'x2' 'x3'
#   * param_i   (param_i) <U2 'x1' 'x2' 'x3'
#   * param_j   (param_j) <U2 'x1' 'x2' 'x3'
# Data variables:
#     S1        (param) float32 ...
#     ST        (param) float32 ...
#     S2        (param_i, param_j) float32 ...

# Access by name
ds.S1.sel(param="x1")       # first-order index for x1
ds.S2.sel(param_i="x1", param_j="x2")  # second-order interaction
```

## Multi-Output with Named Outputs

Label your outputs using `output_names` on the `Problem`:

```python
problem = gsax.Problem.from_dict(
    {
        "amplitude": (0.5, 2.0),
        "frequency": (1.0, 5.0),
        "damping": (0.01, 0.5),
    },
    output_names=("temperature", "pressure"),
)

# ... run analysis with (N, 2) output ...

ds = result.to_dataset()
# Dimensions: (output: 2, param: 3)

ds.S1.sel(param="amplitude", output="temperature")
ds.ST.sel(output="pressure")
```

If `output_names` is not set, outputs are labeled `y0, y1, ...` automatically.

## Time-Series with Custom Coordinates

For 3-D results `(T, K, D)`, pass time coordinate values:

```python
import numpy as np

t = np.linspace(0, 5, 50)

# ... run analysis with (N, T, K) output ...

ds = result.to_dataset(time_coords=t)
# Dimensions: (time: 50, output: K, param: D)

# Select a specific time and output
ds.S1.sel(time=2.5, method="nearest")
ds.ST.sel(param="frequency", output="temperature")
```

Without `time_coords`, integer indices `0, 1, 2, ...` are used.

## Confidence Intervals

Bootstrap confidence intervals are split into separate `_lower` and `_upper` variables:

```python
import jax

result = gsax.analyze(
    sampling_result, Y,
    num_resamples=200,
    key=jax.random.key(0),
)

ds = result.to_dataset()

# ds contains: S1, ST, S2, S1_lower, S1_upper, ST_lower, ST_upper, S2_lower, S2_upper
print(ds.S1_lower.sel(param="x1"))
print(ds.S1_upper.sel(param="x1"))
```

## HDMR Results

`HDMRResult.to_dataset()` works the same way. Term-indexed variables (`Sa`, `Sb`, `S`) use a `term` dimension, while `ST` uses `param`:

```python
hdmr = gsax.analyze_hdmr(problem, X, Y, maxorder=2)
ds = hdmr.to_dataset()

# Access by term label
ds.Sa.sel(term="amplitude/frequency")
ds.ST.sel(param="amplitude")

# select and rmse are included when present
ds.select.sel(term="amplitude")
```

## API Summary

### `SAResult.to_dataset(time_coords=None)`

| Dimension | Coordinates | Used by |
|-----------|-------------|---------|
| `param` | `problem.names` | S1, ST |
| `output` | `problem.output_names` or `y0, y1, ...` | multi-output results |
| `time` | `time_coords` or `0, 1, ...` | 3-D results |
| `param_i` | `problem.names` | S2 |
| `param_j` | `problem.names` | S2 |

### `HDMRResult.to_dataset(time_coords=None)`

| Dimension | Coordinates | Used by |
|-----------|-------------|---------|
| `term` | `self.terms` | Sa, Sb, S, select |
| `param` | `problem.names` | ST |
| `output` | `problem.output_names` or `y0, y1, ...` | multi-output results |
| `time` | `time_coords` or `0, 1, ...` | 3-D results |
