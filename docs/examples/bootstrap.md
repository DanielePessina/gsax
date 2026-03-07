# Bootstrap Confidence Intervals

Use bootstrap resampling to quantify uncertainty in the estimated indices.

```python
import jax
import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate

sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)
Y = evaluate(sampling_result.samples)

result = gsax.analyze(
    sampling_result,
    Y,
    num_resamples=200,
    conf_level=0.95,
    key=jax.random.key(0),
)

# Point estimates
print("S1:", result.S1)
print("ST:", result.ST)

# Confidence intervals -- shape (2, D) where [0]=lower, [1]=upper
print("S1 lower:", result.S1_conf[0])
print("S1 upper:", result.S1_conf[1])

print("ST lower:", result.ST_conf[0])
print("ST upper:", result.ST_conf[1])

# S2 confidence (if calc_second_order=True)
print("S2 lower:", result.S2_conf[0])
print("S2 upper:", result.S2_conf[1])
```

## Notes

- The bootstrap is fully vectorized in JAX and runs ~14.5x faster than SALib's sequential approach.
- Set `num_resamples=0` (the default) to skip bootstrap entirely when you only need point estimates.
- A `jax.random.key` is required when `num_resamples > 0`.
- For multi-output results, the confidence arrays follow the same shape rules with a leading dimension of 2. For example, if `S1.shape == (K, D)`, then `S1_conf.shape == (2, K, D)`.
