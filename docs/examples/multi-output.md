# Multi-Output & Time-Series

gsax handles multi-output and time-series models natively. All indices are computed in a single vectorized pass.

## Multi-output (N, K)

Pass a 2D array where K is the number of outputs:

```python
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM

sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)
X = sampling_result.samples

def multi_output_model(X):
    y1 = jnp.sin(X[:, 0]) + X[:, 1] ** 2
    y2 = X[:, 0] * X[:, 2]
    return jnp.column_stack([y1, y2])

Y = multi_output_model(X)  # (n_total, 2)
result = gsax.analyze(sampling_result, Y)

# result.S1.shape == (2, 3)  -- (K, D)
# result.ST.shape == (2, 3)  -- (K, D)
# result.S2.shape == (2, 3, 3)  -- (K, D, D)

# Access per-output indices
print("S1 for output 0:", result.S1[0])
print("S1 for output 1:", result.S1[1])
```

## Time-series (N, T, K)

Pass a 3D array where T is the number of timesteps and K the number of outputs:

```python
def time_series_model(X):
    # Returns (n_total, T, K) -- e.g. 50 timesteps, 4 outputs
    ...

Y = time_series_model(X)  # (n_total, 50, 4)
result = gsax.analyze(sampling_result, Y)

# result.S1.shape == (50, 4, 3)  -- (T, K, D)
# result.ST.shape == (50, 4, 3)  -- (T, K, D)
# result.S2.shape == (50, 4, 3, 3)  -- (T, K, D, D)

# S1 at timestep 10, output 2, parameter 0
print(result.S1[10, 2, 0])
```

## Edge Cases: Single Output or Single Timestep

A 2D array is **always** interpreted as `(N, K)` — multiple outputs, no time dimension. This matters when your model has only one output or only one timestep:

```python
# Single output, no time dimension — pass a 1D array
Y = my_model(X)          # shape (n_total,)
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (D,)

# Single output WITH time dimension — reshape to (N, T, 1)
Y = my_model(X)          # shape (n_total, T) — e.g. 50 timesteps
Y = Y[:, :, None]        # reshape to (n_total, 50, 1)
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (50, 1, D)  — (T, K=1, D)

# Multiple outputs, single timestep — just pass (N, K)
Y = my_model(X)          # shape (n_total, 4) — 4 outputs
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (4, D)  — (K, D)
# No need for a time dimension; (N, 1, 4) also works but is unnecessary.
```

The same rules apply to `gsax.analyze_hdmr()`.

## Shape Reference

| Y shape | S1 / ST | S2 |
|---------|---------|-----|
| `(N,)` | `(D,)` | `(D, D)` |
| `(N, K)` | `(K, D)` | `(K, D, D)` |
| `(N, T, K)` | `(T, K, D)` | `(T, K, D, D)` |

D is always the last axis. When using bootstrap, confidence arrays prepend a leading dimension of 2: `S1_conf.shape == (2, ..., D)`.
