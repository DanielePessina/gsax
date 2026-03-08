# RS-HDMR Example

Use HDMR when you already have arbitrary `(X, Y)` pairs or when you want a
surrogate that can predict at new inputs.

## Sensitivity analysis from random samples

```python
import jax
import jax.numpy as jnp
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(key, (2000, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
Y = evaluate(X)

result = gsax.analyze_hdmr(
    PROBLEM,
    X,
    Y,
    maxorder=2,
    chunk_size=256,
)

print("S1:", result.S1)
print("ST:", result.ST)
print("Terms:", result.terms)
print("Sa:", result.Sa)
print("Sb:", result.Sb)
print("RMSE:", result.rmse)
```

## Use the emulator

```python
Y_pred = gsax.emulate_hdmr(result, X[:5])
print("Prediction shape:", Y_pred.shape)
print("Absolute residuals:", jnp.abs(Y[:5] - Y_pred))
```

## What to look at

- `result.S1` is the structural first-order contribution extracted from
  `result.Sa`.
- `result.ST` is the total contribution per parameter after summing all terms
  that involve that parameter.
- `result.terms` tells you which columns in `Sa`, `Sb`, and `S` correspond to
  first-order and interaction terms.
- `result.rmse` helps you decide whether the fitted surrogate is accurate enough
  for downstream interpretation.

## Practical caveats

- `analyze_hdmr()` accepts `(N,)`, `(N, K)`, and `(N, T, K)` outputs, so the
  same shape rules from [Multi-Output & Time-Series](/examples/multi-output)
  still apply.
- HDMR does not use a structured Saltelli design; if you want exact Sobol
  estimators on independent inputs, start from [Basic Example](/examples/basic)
  instead.
- If you want labeled `term`, `param`, `time`, and `output` coordinates, call
  `result.to_dataset()` and continue with
  [xarray Labeled Output](/examples/xarray).

## See also

- [Methods](/guide/methods) for the conceptual difference between Sobol and
  HDMR.
- [xarray Labeled Output](/examples/xarray) for exporting `Sa`, `Sb`, `S`, and
  `ST` to a labeled dataset.
- [Advanced Workflow](/examples/advanced-workflow) for a custom time-series
  model that runs Sobol and HDMR side by side.
