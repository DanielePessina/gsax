# Benchmarks

gsax is benchmarked against [SALib](https://salib.readthedocs.io/) on a coupled-oscillator model with varying output shapes. All timings are **post-JIT** (steady-state), best of 5 iterations, on the same hardware and data.

## Results

The benchmark evaluates three methods — `analyze` (Sobol, first/total order only), `analyze` (Sobol with second-order), and `analyze_hdmr` — across four output-shape scenarios.

**Machine:** Apple M3 Pro, CPU only (no GPU), JAX 0.5.x, Python 3.12.

| Scenario (T×K) | Method | gsax (ms) | SALib (ms) | Speedup |
|---|---|---:|---:|---:|
| 1×1 | analyze (no S2) | 0.6 | 13.2 | **20.7×** |
| 1×1 | analyze (S2) | 0.9 | 36.7 | **43.0×** |
| 1×1 | analyze_hdmr | 17.8 | 83.1 | **4.7×** |
| 1×6 | analyze (no S2) | 0.9 | 80.9 | **88.2×** |
| 1×6 | analyze (S2) | 1.3 | 280.4 | **216.6×** |
| 1×6 | analyze_hdmr | 19.3 | 501.4 | **26.0×** |
| 50×1 | analyze (no S2) | 2.2 | 661.8 | **300.4×** |
| 50×1 | analyze (S2) | 3.8 | 2092.2 | **554.7×** |
| 50×1 | analyze_hdmr | 23.2 | 4024.6 | **173.5×** |
| 50×6 | analyze (no S2) | 7.6 | 4442.1 | **582.6×** |
| 50×6 | analyze (S2) | 14.3 | 13289.2 | **929.2×** |
| 50×6 | analyze_hdmr | 38.5 | 29115.3 | **757.1×** |

## Why gsax is faster

**SALib** processes each `(t, k)` output slice in a Python loop. For a 50-timestep × 6-output model, that's 300 sequential calls to the Sobol analyzer.

**gsax** uses:

- **Fused kernels** that compute the pooled variance once and derive all S1, ST, and S2 indices from it (instead of recomputing it D×2 times per output).
- **Vectorized execution** via `jax.vmap` over all T×K output combinations in a single compiled pass.
- **Scalar fast-path** for T×K=1 that bypasses vmap overhead entirely.
- **JIT compilation** so repeated calls (e.g. bootstrap resamples or parameter sweeps) run at native speed.

The speedup grows with T×K because SALib's per-slice overhead is linear while gsax's vectorized cost is nearly flat.

## Benchmark setup

- **Model:** Coupled damped oscillators (D=5 parameters, T timepoints, K outputs).
- **Samples:** N=1024 base Sobol points (7,168 expanded rows for first/total; 12,288 for second-order).
- **HDMR:** maxorder=2, m=2, same N=1024 random samples.
- **Correctness:** Validated against analytical Ishigami solutions and SALib on the same data.

## Reproducing

The full benchmark script is at [`benchmark_salib.py`](https://github.com/DanielePessina/gsax/blob/dev/benchmark_salib.py) in the repository root. Run it locally:

```bash
uv run python benchmark_salib.py
```

It first runs correctness checks (Ishigami function, exact match with SALib), then prints the timing table above. Your numbers will vary by hardware.
