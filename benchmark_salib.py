"""Benchmark: gsax vs SALib — timing on a multi-output, time-indexed model.

Model: 5 parameters, 300 time steps, 4 outputs (coupled oscillators).
Also validates correctness on the Ishigami function with shared samples.
"""

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from SALib.analyze import sobol as salib_sobol
from SALib.sample import sobol as salib_sample_sobol

import gsax
from gsax.benchmarks.ishigami import PROBLEM as ISHIGAMI_PROBLEM
from gsax.benchmarks.ishigami import evaluate as ishigami_evaluate

# ---------------------------------------------------------------------------
# Benchmark model: coupled damped oscillators
# ---------------------------------------------------------------------------
# 5 parameters on [0, 1]:
#   x0 = amplitude,  x1 = frequency,  x2 = damping,
#   x3 = coupling,   x4 = drift
#
# 4 outputs over 300 time points in [0, 5]:
#   y0(t) = x0 * sin(2π x1 t) * exp(-x2 t)
#   y1(t) = x1 * cos(2π x0 t) + x4 * t²
#   y2(t) = x3 * sin(x4 * 10 t) + x0 * x2
#   y3(t) = (x3 + x4) * exp(-x0 t) * sin(2π x1 t)

T_POINTS = 100
K_OUTPUTS = 4
D_PARAMS = 5

BENCH_PROBLEM = gsax.Problem.from_dict(
    {
        "amplitude": (0.1, 2.0),
        "frequency": (0.5, 5.0),
        "damping": (0.01, 1.0),
        "coupling": (0.1, 3.0),
        "drift": (0.0, 1.0),
    }
)

_T_GRID = jnp.linspace(0.0, 5.0, T_POINTS)  # (T,)


def coupled_oscillators(X: jax.Array) -> jax.Array:
    """Evaluate the benchmark model.

    Args:
        X: (n_total, 5) parameter samples scaled to bounds.

    Returns:
        Y: (n_total, T, K) model outputs.
    """
    x0 = X[:, 0, None]  # (N, 1)
    x1 = X[:, 1, None]
    x2 = X[:, 2, None]
    x3 = X[:, 3, None]
    x4 = X[:, 4, None]

    t = _T_GRID[None, :]  # (1, T)

    y0 = x0 * jnp.sin(2 * jnp.pi * x1 * t) * jnp.exp(-x2 * t)
    y1 = x1 * jnp.cos(2 * jnp.pi * x0 * t) + x4 * t**2
    y2 = x3 * jnp.sin(x4 * 10 * t) + x0 * x2
    y3 = (x3 + x4) * jnp.exp(-x0 * t) * jnp.sin(2 * jnp.pi * x1 * t)

    return jnp.stack([y0, y1, y2, y3], axis=-1)  # (N, T, K)


def coupled_oscillators_numpy(X: np.ndarray) -> np.ndarray:
    """Same model, pure numpy for SALib."""
    x0 = X[:, 0, None]
    x1 = X[:, 1, None]
    x2 = X[:, 2, None]
    x3 = X[:, 3, None]
    x4 = X[:, 4, None]

    t = np.linspace(0.0, 5.0, T_POINTS)[None, :]

    y0 = x0 * np.sin(2 * np.pi * x1 * t) * np.exp(-x2 * t)
    y1 = x1 * np.cos(2 * np.pi * x0 * t) + x4 * t**2
    y2 = x3 * np.sin(x4 * 10 * t) + x0 * x2
    y3 = (x3 + x4) * np.exp(-x0 * t) * np.sin(2 * np.pi * x1 * t)

    return np.stack([y0, y1, y2, y3], axis=-1)  # (N, T, K)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gsax_problem_to_salib(problem: gsax.Problem) -> dict:
    return {
        "num_vars": problem.num_vars,
        "names": list(problem.names),
        "bounds": [list(b) for b in problem.bounds],
    }


# ---------------------------------------------------------------------------
# Correctness check (Ishigami, shared samples — quick sanity test)
# ---------------------------------------------------------------------------


def benchmark_correctness() -> bool:
    n_samples = 2**14 * 8
    sr = gsax.sample(ISHIGAMI_PROBLEM, n_samples, seed=42, calc_second_order=True)
    Y_jax = ishigami_evaluate(jnp.asarray(sr.samples))
    Y_np = np.asarray(Y_jax)

    gsax_result = gsax.analyze(sr, Y_jax)

    salib_problem = gsax_problem_to_salib(ISHIGAMI_PROBLEM)
    salib_result = salib_sobol.analyze(
        salib_problem,
        Y_np,
        calc_second_order=True,
        print_to_console=False,
    )

    D = ISHIGAMI_PROBLEM.num_vars
    mask = np.triu(np.ones((D, D), dtype=bool), k=1)
    all_pass = True

    print("=" * 70)
    print("CORRECTNESS CHECK  (Ishigami, shared samples)")
    print("=" * 70)
    for label, g, s, atol in [
        ("S1", np.asarray(gsax_result.S1), salib_result["S1"], 1e-6),
        ("ST", np.asarray(gsax_result.ST), salib_result["ST"], 1e-6),
        ("S2", np.asarray(gsax_result.S2)[mask], salib_result["S2"][mask], 1e-6),
    ]:
        ok = np.allclose(g, s, atol=atol)
        print(f"  {label:<6} match (atol={atol}): {'PASS' if ok else 'FAIL'}")
        all_pass &= ok

    return all_pass


# ---------------------------------------------------------------------------
# Timing benchmark — multi-output model
# ---------------------------------------------------------------------------


def benchmark_timing(base_n: int = 256, n_repeats: int = 3) -> None:
    D = D_PARAMS
    step = 2 * D + 2
    n_total = base_n * step
    salib_problem = gsax_problem_to_salib(BENCH_PROBLEM)

    print(f"\n{'=' * 70}")
    print("TIMING BENCHMARK — coupled oscillators")
    print(f"  D={D}, T={T_POINTS}, K={K_OUTPUTS}, base_n={base_n}, n_total={n_total}")
    print(f"  SALib must call analyze() {T_POINTS}x{K_OUTPUTS} = {T_POINTS * K_OUTPUTS} times")
    print(f"  n_repeats={n_repeats}")
    print("=" * 70)

    # --- Warmup gsax JIT ---
    print("\nWarming up gsax JIT ...", end=" ", flush=True)
    sr_w = gsax.sample(BENCH_PROBLEM, n_total, seed=0, calc_second_order=True)
    Y_w = coupled_oscillators(jnp.asarray(sr_w.samples))
    r_w = gsax.analyze(sr_w, Y_w)
    jax.block_until_ready(r_w.S1)
    jax.block_until_ready(r_w.ST)
    jax.block_until_ready(r_w.S2)
    print("done.")

    # ---- gsax: sample + evaluate + analyze (all in JAX) ----
    gsax_sample_times = []
    gsax_eval_times = []
    gsax_analyze_times = []

    for i in range(n_repeats):
        t0 = time.perf_counter()
        sr = gsax.sample(BENCH_PROBLEM, n_total, seed=i, calc_second_order=True)
        t1 = time.perf_counter()
        Y = coupled_oscillators(jnp.asarray(sr.samples))
        jax.block_until_ready(Y)
        t2 = time.perf_counter()
        result = gsax.analyze(sr, Y)
        jax.block_until_ready(result.S1)
        jax.block_until_ready(result.ST)
        jax.block_until_ready(result.S2)
        t3 = time.perf_counter()

        gsax_sample_times.append(t1 - t0)
        gsax_eval_times.append(t2 - t1)
        gsax_analyze_times.append(t3 - t2)

    # ---- SALib: sample + evaluate + analyze (loop over T*K) ----
    salib_sample_times = []
    salib_eval_times = []
    salib_analyze_times = []

    for i in range(n_repeats):
        t0 = time.perf_counter()
        X = salib_sample_sobol.sample(salib_problem, base_n, calc_second_order=True)
        t1 = time.perf_counter()
        Y_np = coupled_oscillators_numpy(X)  # (n_total, T, K)
        t2 = time.perf_counter()

        # SALib can only analyze 1D output — loop over (T, K)
        for t_idx in range(T_POINTS):
            for k_idx in range(K_OUTPUTS):
                salib_sobol.analyze(
                    salib_problem,
                    Y_np[:, t_idx, k_idx],
                    calc_second_order=True,
                    print_to_console=False,
                )
        t3 = time.perf_counter()

        salib_sample_times.append(t1 - t0)
        salib_eval_times.append(t2 - t1)
        salib_analyze_times.append(t3 - t2)

    # ---- Print results ----
    def _stats(times: list[float]) -> tuple[float, float, float]:
        a = np.array(times) * 1e3
        return a.mean(), a.std(), a.min()

    print(f"\n{'Phase':<14} {'gsax (ms)':>14} {'SALib (ms)':>14} {'speedup':>10}")
    print("-" * 54)

    for label, g_t, s_t in [
        ("sample", gsax_sample_times, salib_sample_times),
        ("evaluate", gsax_eval_times, salib_eval_times),
        ("analyze", gsax_analyze_times, salib_analyze_times),
    ]:
        g_mean, _, _ = _stats(g_t)
        s_mean, _, _ = _stats(s_t)
        sp = s_mean / g_mean if g_mean > 0 else float("inf")
        print(f"  {label:<12} {g_mean:>12.1f}   {s_mean:>12.1f}   {sp:>8.1f}x")

    g_total = (
        np.array(gsax_sample_times) + np.array(gsax_eval_times) + np.array(gsax_analyze_times)
    )
    s_total = (
        np.array(salib_sample_times) + np.array(salib_eval_times) + np.array(salib_analyze_times)
    )
    g_mean = g_total.mean() * 1e3
    s_mean = s_total.mean() * 1e3
    print("-" * 54)
    print(f"  {'total':<12} {g_mean:>12.1f}   {s_mean:>12.1f}   {s_mean / g_mean:>8.1f}x")


# ---------------------------------------------------------------------------
# Bootstrap benchmark
# ---------------------------------------------------------------------------


def benchmark_bootstrap_timing(
    base_n: int = 256, num_resamples: int = 100, n_repeats: int = 3
) -> None:
    D = D_PARAMS
    step = 2 * D + 2
    n_total = base_n * step
    salib_problem = gsax_problem_to_salib(BENCH_PROBLEM)

    print(f"\n{'=' * 70}")
    print("BOOTSTRAP BENCHMARK — coupled oscillators")
    print(
        f"  D={D}, T={T_POINTS}, K={K_OUTPUTS}, base_n={base_n}, R={num_resamples}"
    )
    print(f"  n_repeats={n_repeats}")
    print("=" * 70)

    # Shared sample + evaluation (computed once)
    sr = gsax.sample(BENCH_PROBLEM, n_total, seed=0, calc_second_order=True)
    Y_jax = coupled_oscillators(jnp.asarray(sr.samples))
    jax.block_until_ready(Y_jax)
    Y_np = np.asarray(Y_jax)

    # Warmup gsax bootstrap JIT
    print("\nWarming up gsax bootstrap JIT ...", end=" ", flush=True)
    _ = gsax.analyze(sr, Y_jax, num_resamples=num_resamples, key=jax.random.key(99))
    print("done.")

    # --- gsax no-bootstrap ---
    gsax_no_boot = []
    for i in range(n_repeats):
        t0 = time.perf_counter()
        r = gsax.analyze(sr, Y_jax)
        jax.block_until_ready(r.S1)
        jax.block_until_ready(r.ST)
        jax.block_until_ready(r.S2)
        gsax_no_boot.append(time.perf_counter() - t0)

    # --- gsax with bootstrap ---
    gsax_boot = []
    for i in range(n_repeats):
        t0 = time.perf_counter()
        r = gsax.analyze(
            sr, Y_jax, num_resamples=num_resamples, key=jax.random.key(i)
        )
        jax.block_until_ready(r.S1)
        jax.block_until_ready(r.S1_conf)
        gsax_boot.append(time.perf_counter() - t0)

    # --- SALib with bootstrap ---
    salib_boot = []
    for i in range(n_repeats):
        t0 = time.perf_counter()
        for t_idx in range(T_POINTS):
            for k_idx in range(K_OUTPUTS):
                salib_sobol.analyze(
                    salib_problem,
                    Y_np[:, t_idx, k_idx],
                    calc_second_order=True,
                    num_resamples=num_resamples,
                    print_to_console=False,
                )
        salib_boot.append(time.perf_counter() - t0)

    # --- Print results ---
    def _ms(times):
        return np.mean(times) * 1e3

    g_nb = _ms(gsax_no_boot)
    g_b = _ms(gsax_boot)
    s_b = _ms(salib_boot)

    print(f"\n{'Method':<30} {'Time (ms)':>12} {'vs gsax-noboot':>16}")
    print("-" * 60)
    print(f"  {'gsax (no bootstrap)':<28} {g_nb:>10.1f}   {'1.0x':>14}")
    print(f"  {'gsax (bootstrap R=' + str(num_resamples) + ')':<28} {g_b:>10.1f}   {g_b / g_nb:>13.1f}x")
    print(f"  {'SALib (bootstrap R=' + str(num_resamples) + ')':<28} {s_b:>10.1f}   {s_b / g_nb:>13.1f}x")
    print(f"\n  gsax bootstrap speedup vs SALib: {s_b / g_b:.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    correct = benchmark_correctness()
    benchmark_timing(base_n=4096, n_repeats=1)
    benchmark_bootstrap_timing(base_n=4096, num_resamples=200, n_repeats=1)

    print("\n" + "=" * 70)
    if correct:
        print("ALL CORRECTNESS CHECKS PASSED")
        return 0
    else:
        print("SOME CORRECTNESS CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
