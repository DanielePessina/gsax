# gsax — Global Sensitivity Analysis in JAX

Trigger: user wants Sobol sensitivity analysis, Sobol indices, global SA, or variance-based sensitivity analysis.

## Workflow

```python
import jax.numpy as jnp
import gsax

# 1. Define the problem: parameter names and (lower, upper) bounds
problem = gsax.Problem.from_dict({
    "x1": (-3.14, 3.14),
    "x2": (-3.14, 3.14),
    "x3": (-3.14, 3.14),
})

# 2. Generate Saltelli samples
sr = gsax.sample(problem, n_samples=4096, seed=0)
# sr.samples is a NumPy array of shape (n_total, D)

# 3. Evaluate the model on every sample row
X = sr.samples  # shape (n_total, D)
Y = my_model(X)  # must return shape (n_total,), (n_total, K), or (n_total, T, K)

# 4. Compute Sobol indices
result = gsax.analyze(sr, Y)
print(result.S1)  # first-order indices
print(result.ST)  # total-order indices
print(result.S2)  # second-order indices (if calc_second_order=True)
```

## API reference

### `gsax.Problem.from_dict(params: dict[str, tuple[float, float]]) -> Problem`
Create a problem from a dict mapping parameter names to `(lower, upper)` bounds.

### `gsax.sample(problem, n_samples, *, calc_second_order=True, scramble=True, seed=None) -> SamplingResult`
Generate a Saltelli sample matrix using Sobol sequences.
- `n_samples`: minimum desired number of model evaluations (rounded up to power of 2 internally).
- `calc_second_order`: set `False` to skip second-order indices (fewer samples needed).
- `seed`: int or `np.random.Generator` for reproducibility.
- Returns `SamplingResult` with `.samples` shape `(n_total, D)` scaled to bounds.

### `gsax.analyze(sampling_result, Y, *, chunk_size=2048) -> SAResult`
Compute Sobol indices from model outputs.
- `Y`: JAX or NumPy array of model outputs. Supported shapes:
  - `(n_total,)` — scalar output
  - `(n_total, K)` — K outputs
  - `(n_total, T, K)` — K outputs over T time steps
- `chunk_size`: number of output combinations per vmap batch (controls memory usage).
- Returns `SAResult`.

### `SAResult` fields
- `S1`: first-order indices. Shape `(D,)`, `(K, D)`, or `(T, K, D)`.
- `ST`: total-order indices. Same shape as `S1`.
- `S2`: second-order indices. Shape `(D, D)`, `(K, D, D)`, or `(T, K, D, D)`. `None` if `calc_second_order=False`.
- `problem`: the `Problem` used.

## Common patterns

### Scalar output
```python
Y = model(sr.samples)  # shape (n_total,)
result = gsax.analyze(sr, Y)
# result.S1 shape: (D,)
```

### Multi-output
```python
Y = model(sr.samples)  # shape (n_total, K)
result = gsax.analyze(sr, Y)
# result.S1 shape: (K, D) — one set of indices per output
```

### Time-series output
```python
Y = model(sr.samples)  # shape (n_total, T, K)
result = gsax.analyze(sr, Y)
# result.S1 shape: (T, K, D) — indices evolve over time
```

### Skip second-order indices (halves sample count)
```python
sr = gsax.sample(problem, n_samples=4096, calc_second_order=False)
Y = model(sr.samples)
result = gsax.analyze(sr, Y)
# result.S2 is None
```

### Large outputs — reduce memory with chunk_size
```python
result = gsax.analyze(sr, Y, chunk_size=512)
```
