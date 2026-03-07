# RS-HDMR Example

RS-HDMR works with any set of (X, Y) pairs -- no structured sampling design required.

## Sensitivity Analysis

```python
import jax
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

# Generate random input samples (any sampling method works)
key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(key, (2000, 3), minval=bounds[:, 0], maxval=bounds[:, 1])

# Evaluate the model
Y = evaluate(X)

# Compute HDMR indices
result = gsax.analyze_hdmr(
    PROBLEM, X, Y,
    maxorder=2,
    chunk_size=64,
)

# Sobol-compatible indices
print("S1:", result.S1)
print("ST:", result.ST)

# HDMR-specific per-term decomposition
print("Terms:", result.terms)  # ('x1', 'x2', 'x3', 'x1/x2', 'x1/x3', 'x2/x3')
print("Sa:", result.Sa)        # structural (uncorrelated) contribution
print("Sb:", result.Sb)        # correlative contribution
```

## Using the Emulator

The fitted HDMR surrogate can predict at new input points:

```python
# Predict at the original inputs (sanity check)
Y_pred = gsax.emulate_hdmr(result, X)

# Predict at new inputs
key2 = jax.random.PRNGKey(99)
X_new = jax.random.uniform(key2, (500, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
Y_new = gsax.emulate_hdmr(result, X_new)
```

## When to Use HDMR

- Model evaluations are expensive and you want to reuse existing runs
- Inputs may be correlated
- You need a fast surrogate for prediction at new inputs
- You want to understand both structural and correlative contributions to variance
