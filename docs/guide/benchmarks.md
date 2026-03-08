# Benchmarks

gsax is benchmarked against [SALib](https://salib.readthedocs.io/) on a coupled-oscillator model with varying output shapes. All timings are **post-JIT** (steady-state), best of 5 iterations, on the same hardware and data.

## Results

The benchmark evaluates three methods — `analyze` (Sobol, first/total order only), `analyze` (Sobol with second-order), and `analyze_hdmr` — across four output-shape scenarios, with and without bootstrap confidence intervals.

**Machine:** Apple M3 Pro, CPU only (no GPU), JAX 0.5.x, Python 3.12.

### Sobol — no bootstrap

| Scenario (T×K) | Method | gsax (ms) | SALib (ms) | Speedup |
|---|---|---:|---:|---:|
| 1×1 | analyze (no S2) | 0.8 | 0.2 | **0.3×** |
| 1×1 | analyze (S2) | 0.9 | 0.9 | **1.0×** |
| 1×6 | analyze (no S2) | 1.0 | 1.4 | **1.4×** |
| 1×6 | analyze (S2) | 1.5 | 5.3 | **3.5×** |
| 50×1 | analyze (no S2) | 2.8 | 12.3 | **4.4×** |
| 50×1 | analyze (S2) | 4.9 | 45.8 | **9.4×** |
| 50×6 | analyze (no S2) | 8.8 | 74.3 | **8.5×** |
| 50×6 | analyze (S2) | 14.8 | 276.6 | **18.7×** |

### Sobol — 300 bootstrap resamples

| Scenario (T×K) | Method | gsax (ms) | SALib (ms) | Speedup |
|---|---|---:|---:|---:|
| 1×1 | analyze (no S2) | 5.2 | 28.5 | **5.5×** |
| 1×1 | analyze (S2) | 8.5 | 80.0 | **9.4×** |
| 1×6 | analyze (no S2) | 17.0 | 162.8 | **9.6×** |
| 1×6 | analyze (S2) | 36.4 | 490.5 | **13.5×** |
| 50×1 | analyze (no S2) | 121.7 | 1434.9 | **11.8×** |
| 50×1 | analyze (S2) | 280.6 | 4142.0 | **14.8×** |
| 50×6 | analyze (no S2) | 726.1 | 9384.2 | **12.9×** |
| 50×6 | analyze (S2) | 1666.7 | 26596.8 | **16.0×** |

### HDMR

| Scenario (T×K) | Method | gsax (ms) | SALib (ms) | Speedup |
|---|---|---:|---:|---:|
| 1×1 | analyze_hdmr | 17.6 | 89.5 | **5.1×** |
| 1×6 | analyze_hdmr | 19.0 | 508.6 | **26.7×** |
| 50×1 | analyze_hdmr | 22.3 | 3990.5 | **178.7×** |
| 50×6 | analyze_hdmr | 37.0 | 28345.8 | **766.3×** |

## Why gsax is faster

**SALib** processes each `(t, k)` output slice in a Python loop. For a 50-timestep × 6-output model, that's 300 sequential calls to the Sobol analyzer.

**gsax** uses:

- **Fused kernels** that compute the pooled variance once and derive all S1, ST, and S2 indices from it (instead of recomputing it D×2 times per output).
- **Vectorized execution** via `jax.vmap` over all T×K output combinations in a single compiled pass.
- **Scalar fast-path** for T×K=1 that bypasses vmap overhead entirely.
- **JIT compilation** so repeated calls (e.g. bootstrap resamples or parameter sweeps) run at native speed.

The speedup grows with T×K because SALib's per-slice overhead is linear while gsax's vectorized cost is nearly flat. With bootstrap enabled, JIT compilation pays off even more — resampled analyses reuse the same compiled kernel, while SALib re-runs pure Python each time.

## Benchmark setup

- **Model:** Coupled damped oscillators (D=5 parameters, T timepoints, K outputs).
- **Samples:** N=1024 base Sobol points (7,168 expanded rows for first/total; 12,288 for second-order).
- **Bootstrap:** 300 resamples for the bootstrap tables; no bootstrap for the base tables.
- **HDMR:** maxorder=2, m=2, same N=1024 random samples.
- **Correctness:** Validated against analytical Ishigami solutions (D=3, N=16384) and SALib on the same data.

## Reproducing

The full benchmark script is at [`benchmark_salib.py`](https://github.com/DanielePessina/gsax/blob/dev/benchmark_salib.py) in the repository root. Run it locally:

```bash
uv run python benchmark_salib.py
```

It first runs correctness checks (Ishigami function, exact match with SALib), then prints the timing table above. Your numbers will vary by hardware.
