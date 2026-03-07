import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate
import jax
import jax.numpy as jnp

# =============================================================================
# 1. Sobol analysis (sample + analyze)
# =============================================================================

# Generate Saltelli samples
sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42)
# sampling_result.samples.shape == (n_total, 3)

# Evaluate the Ishigami function on unique samples
Y = evaluate(jnp.asarray(sampling_result.samples))  # (n_total,)

# Compute Sobol indices
result = gsax.analyze(sampling_result, Y)
print("=== Sobol indices ===")
print("S1:", result.S1)  # (3,)
print("ST:", result.ST)  # (3,)
print("S2:")
print(result.S2)  # (3, 3)

# =============================================================================
# 2. Sobol analysis with bootstrap confidence intervals
# =============================================================================

result_ci = gsax.analyze(
    sampling_result,
    Y,
    num_resamples=200,
    conf_level=0.95,
    key=jax.random.key(0),
)
print("\n=== Sobol with bootstrap ===")
print("S1:", result_ci.S1)
assert result_ci.S1_conf is not None
print("S1 CI low: ", result_ci.S1_conf[0])
print("S1 CI high:", result_ci.S1_conf[1])

# =============================================================================
# 3. RS-HDMR analysis (analyze_hdmr)
# =============================================================================

# Generate random samples (no structured sampling needed)
key = jax.random.PRNGKey(42)
bounds = jnp.array(PROBLEM.bounds)
X = jax.random.uniform(key, (2000, 3), minval=bounds[:, 0], maxval=bounds[:, 1])

# Evaluate the model
Y_hdmr = evaluate(X)  # (2000,)

# Compute HDMR sensitivity indices
hdmr_result = gsax.analyze_hdmr(PROBLEM, X, Y_hdmr, maxorder=2)

print("\n=== HDMR indices ===")
print("S1:", hdmr_result.S1)   # (3,) first-order
print("ST:", hdmr_result.ST)   # (3,) total-order
print("Sa:", hdmr_result.Sa)   # per-term structural contribution
print("Sb:", hdmr_result.Sb)   # per-term correlative contribution
print("Terms:", hdmr_result.terms)

# Use the fitted surrogate as an emulator
Y_pred = gsax.emulate_hdmr(hdmr_result, X)
print(f"\nEmulator RMSE: {hdmr_result.rmse}")
