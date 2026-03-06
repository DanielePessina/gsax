# gsax

**Global Sensitivity Analysis in JAX**

`gsax` computes Sobol variance-based sensitivity indices entirely in JAX, giving you GPU/TPU acceleration and JIT compilation for free. It implements Saltelli's sampling scheme with Sobol quasi-random sequences and supports first-order, total-order, and second-order indices with chunked vectorization for bounded memory usage.

## Features

- Saltelli sampling via Sobol quasi-random sequences (powered by `scipy.stats.qmc`)
- First-order (S1), total-order (ST), and second-order (S2) Sobol indices
- Chunked `jit(vmap(...))` execution for bounded memory on large output grids
- Supports scalar, multi-output, and time-series model outputs
- Built-in Ishigami benchmark function with known analytical solutions

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

# 2. Evaluate your model on the samples
Y = evaluate(sampling_result.samples)

# 3. Compute Sobol indices
result = gsax.analyze(sampling_result, Y)

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

Y = multi_output_model(sampling_result.samples)
result = gsax.analyze(sampling_result, Y)
# result.S1.shape == (2, 3)  — 2 outputs, 3 parameters
```

## Dependencies

- `jax >= 0.4`
- `jaxlib >= 0.4`
- `scipy >= 1.10`

## License

See [LICENSE](LICENSE) for details.
