"""Damped harmonic oscillator — gsax sensitivity analysis example.

D = 5 parameters, T = 50 timepoints, K = 5 outputs.
Demonstrates both Sobol (gsax.analyze) and RS-HDMR (gsax.analyze_hdmr),
including xarray Dataset conversion via to_dataset().
"""

import jax
import jax.numpy as jnp
import numpy as np

import gsax

# =============================================================================
# 1. Problem definition
# =============================================================================

problem = gsax.Problem.from_dict(
    {
        "amplitude": (0.5, 2.0),
        "frequency": (1.0, 5.0),
        "damping": (0.01, 0.5),
        "phase": (0.0, 2 * np.pi),
        "offset": (-1.0, 1.0),
    },
    output_names=("displacement", "velocity", "energy", "envelope", "modulated"),
)

T = 50
t = jnp.linspace(0, 5, T)  # (T,)


# =============================================================================
# 2. Model function  (N,) params -> (N, T, K=5)
# =============================================================================


def model(X: jax.Array) -> jax.Array:
    """Evaluate the damped harmonic oscillator for N parameter samples.

    Args:
        X: (N, 5) array — [amplitude, frequency, damping, phase, offset].

    Returns:
        Y: (N, T, 5) array of outputs.
    """
    amp = X[:, 0, None]  # (N, 1)
    freq = X[:, 1, None]
    damp = X[:, 2, None]
    phase = X[:, 3, None]
    off = X[:, 4, None]

    # t broadcast: (1, T)
    tt = t[None, :]

    # y0: displacement
    y0 = amp * jnp.sin(2 * jnp.pi * freq * tt + phase) * jnp.exp(-damp * tt) + off

    # y1: velocity (analytical derivative w.r.t. t)
    y1 = (
        amp
        * jnp.exp(-damp * tt)
        * (
            2 * jnp.pi * freq * jnp.cos(2 * jnp.pi * freq * tt + phase)
            - damp * jnp.sin(2 * jnp.pi * freq * tt + phase)
        )
    )

    # y2: energy proxy
    y2 = y0**2 + y1**2

    # y3: envelope
    y3 = amp * jnp.exp(-damp * tt)

    # y4: offset-modulated displacement
    y4 = y0 * (1 + off * tt)

    return jnp.stack([y0, y1, y2, y3, y4], axis=-1)  # (N, T, 5)


# =============================================================================
# 3. Sobol analysis (S1, ST, S2)
# =============================================================================

print("=== Sobol Analysis ===")
sampling_result = gsax.sample(problem, n_samples=2048, seed=42, calc_second_order=True)

X_sobol = jnp.asarray(sampling_result.samples)
Y_sobol = model(X_sobol)  # (N, T, K)
print(f"Y shape: {Y_sobol.shape}")

sobol = gsax.analyze(sampling_result, Y_sobol)

print(f"S1 shape: {sobol.S1.shape}")  # (T, K, D)
print(f"ST shape: {sobol.ST.shape}")
assert sobol.S2 is not None
print(f"S2 shape: {sobol.S2.shape}")  # (T, K, D, D)

print("\nS1 (first timepoint, all outputs):")
print(sobol.S1[0, :, :])

print("\nST (first timepoint, all outputs):")
print(sobol.ST[0, :, :])

print("\nS2 (first timepoint, first output) — upper triangle:")
print(sobol.S2[0, 0, :, :])

# =============================================================================
# 4. RS-HDMR analysis
# =============================================================================

print("\n=== HDMR Analysis ===")
key = jax.random.PRNGKey(42)
bounds = jnp.array(problem.bounds)
X_hdmr = jax.random.uniform(key, (2000, 5), minval=bounds[:, 0], maxval=bounds[:, 1])

Y_hdmr = model(X_hdmr)  # (2000, T, K)
print(f"Y shape: {Y_hdmr.shape}")

hdmr = gsax.analyze_hdmr(problem, X_hdmr, Y_hdmr, maxorder=2)

print(f"S1 shape: {hdmr.S1.shape}")  # (T, K, D)
print(f"ST shape: {hdmr.ST.shape}")

print("\nS1 (first timepoint, all outputs):")
print(hdmr.S1[0, :, :])

print("\nST (first timepoint, all outputs):")
print(hdmr.ST[0, :, :])

print(f"\nEmulator RMSE: {hdmr.rmse}")

# =============================================================================
# 5. Emulator prediction
# =============================================================================

print("\n=== HDMR Emulator ===")
Y_pred = gsax.emulate_hdmr(hdmr, X_hdmr[:5])
print(f"Prediction shape: {Y_pred.shape}")

residual = jnp.abs(Y_hdmr[:5] - Y_pred)
print(f"Max absolute error (first 5 samples): {residual.max():.6f}")

# =============================================================================
# 6. xarray Dataset conversion
# =============================================================================

print("\n=== xarray Datasets ===")

# Convert Sobol result to labeled xarray Dataset
time_values = np.asarray(t)  # use actual time coordinates
ds_sobol = sobol.to_dataset(time_coords=time_values)
print(f"\nSobol Dataset:\n{ds_sobol}")

# Labeled access: S1 for "amplitude" parameter at all times and outputs
print("\nS1 for 'amplitude' (all times, all outputs):")
print(ds_sobol.S1.sel(param="amplitude"))

# S1 for a specific output at a specific time
print("\nS1 for 'displacement' output at time 0:")
print(ds_sobol.S1.sel(output="displacement", time=time_values[0]))

# S2 interaction matrix uses param_i / param_j dimensions
print("\nS2 dims:", ds_sobol.S2.dims)
print("S2 amplitude x frequency (first time, first output):")
print(
    ds_sobol.S2.sel(
        param_i="amplitude", param_j="frequency", time=time_values[0], output="displacement"
    ).values
)

# Convert HDMR result to labeled xarray Dataset
ds_hdmr = hdmr.to_dataset(time_coords=time_values)
print(f"\nHDMR Dataset:\n{ds_hdmr}")

# Term-indexed access
print("\nST for 'amplitude' parameter:")
print(ds_hdmr.ST.sel(param="amplitude"))

print("\nSa for 'amplitude/frequency' interaction term:")
print(ds_hdmr.Sa.sel(term="amplitude/frequency"))
