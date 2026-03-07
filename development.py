import gsax
from gsax.benchmarks.ishigami import PROBLEM, evaluate
import jax
import jax.numpy as jnp

# =============================================================================
# 1. Sobol analysis (sample + analyze)
# =============================================================================

# Generate Saltelli samples for first- and total-order indices only.
# Set calc_second_order=True if you also want S2.
sampling_result = gsax.sample(PROBLEM, n_samples=4096, seed=42, calc_second_order=False)
# sampling_result.samples.shape == (n_total, 3)

# Evaluate the Ishigami function on unique samples
Y = evaluate(jnp.asarray(sampling_result.samples))  # (n_total,)

# Compute Sobol indices
result = gsax.analyze(sampling_result, Y)
print("=== Sobol indices ===")
print("S1:", result.S1)  # (3,)
print("ST:", result.ST)  # (3,)
print("S2:", result.S2)  # None when calc_second_order=False

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

# =============================================================================
# 4. Rough timing demo (different workloads, not an apples-to-apples benchmark)
# =============================================================================
import time

n_samples_list = [512, 1024, 2048, 4096]
n_outputs_list = [1, 3, 10]
n_repeats = 5

print("\n=== Rough Timing Demo: analyze vs analyze_hdmr ===")
print("Sobol uses Saltelli-expanded samples; HDMR uses an independent random design.")
print("Treat these numbers as separate workload timings, not a direct speed comparison.")
print(f"{'n_samples':>10} {'n_outputs':>10} {'sobol analyze (s)':>18} {'hdmr analyze (s)':>18}")
print("-" * 62)

for n_samples in n_samples_list:
    for n_out in n_outputs_list:
        # --- Sobol analyze setup ---
        sr = gsax.sample(PROBLEM, n_samples=n_samples, seed=0, calc_second_order=False)
        X_sobol = jnp.asarray(sr.samples)
        base_Y_sobol = evaluate(X_sobol)
        if n_out == 1:
            Y_sobol = base_Y_sobol
        else:
            Y_sobol = jnp.column_stack([base_Y_sobol] * n_out)

        # --- HDMR setup ---
        key_bench = jax.random.PRNGKey(0)
        bounds = jnp.array(PROBLEM.bounds)
        X_hdmr = jax.random.uniform(key_bench, (n_samples, 3), minval=bounds[:, 0], maxval=bounds[:, 1])
        base_Y_hdmr = evaluate(X_hdmr)
        if n_out == 1:
            Y_hdmr_bench = base_Y_hdmr
        else:
            Y_hdmr_bench = jnp.column_stack([base_Y_hdmr] * n_out)

        # --- Warmup (compile) ---
        sobol_warmup = gsax.analyze(sr, Y_sobol)
        hdmr_warmup = gsax.analyze_hdmr(PROBLEM, X_hdmr, Y_hdmr_bench, maxorder=2)
        jax.block_until_ready(sobol_warmup.S1)
        jax.block_until_ready(hdmr_warmup.S1)

        # --- Time Sobol analyze ---
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            sobol_result = gsax.analyze(sr, Y_sobol)
            jax.block_until_ready(sobol_result.S1)
        t_analyze = (time.perf_counter() - t0) / n_repeats

        # --- Time HDMR ---
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            hdmr_result = gsax.analyze_hdmr(PROBLEM, X_hdmr, Y_hdmr_bench, maxorder=2)
            jax.block_until_ready(hdmr_result.S1)
        t_hdmr = (time.perf_counter() - t0) / n_repeats

        print(f"{n_samples:>10} {n_out:>10} {t_analyze:>18.4f} {t_hdmr:>18.4f}")
