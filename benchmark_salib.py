"""Benchmark: gsax vs SALib — correctness + timing across output shapes.

Correctness: Ishigami function (D=3) with known analytical solutions.
Timing: coupled oscillators model with varying T (timepoints) and K (outputs).
"""

from __future__ import annotations

import sys
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from SALib.analyze import hdmr as salib_hdmr
from SALib.analyze import sobol as salib_sobol
from SALib.sample import sobol as salib_sample_sobol

import gsax
from gsax.benchmarks.ishigami import ANALYTICAL_S1 as ISHIGAMI_ANALYTICAL_S1
from gsax.benchmarks.ishigami import ANALYTICAL_ST as ISHIGAMI_ANALYTICAL_ST
from gsax.benchmarks.ishigami import PROBLEM as ISHIGAMI_PROBLEM
from gsax.benchmarks.ishigami import evaluate as ishigami_evaluate

# ---------------------------------------------------------------------------
# Benchmark model: coupled damped oscillators (D=5)
# ---------------------------------------------------------------------------

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


def coupled_oscillators(X: jax.Array, T: int, K: int) -> jax.Array:
    """Evaluate benchmark model -> shape depends on (T, K).

    Returns:
        (N,)       if T=1, K=1
        (N, K)     if T=1, K>1
        (N, T, K)  if T>1
    """
    x0 = X[:, 0, None]
    x1 = X[:, 1, None]
    x2 = X[:, 2, None]
    x3 = X[:, 3, None]
    x4 = X[:, 4, None]

    t = jnp.linspace(0.0, 5.0, T)[None, :]  # (1, T)

    y0 = x0 * jnp.sin(2 * jnp.pi * x1 * t) * jnp.exp(-x2 * t)
    y1 = x1 * jnp.cos(2 * jnp.pi * x0 * t) + x4 * t**2
    y2 = x3 * jnp.sin(x4 * 10 * t) + x0 * x2
    y3 = (x3 + x4) * jnp.exp(-x0 * t) * jnp.sin(2 * jnp.pi * x1 * t)
    y4 = x0 * x3 * jnp.cos(x1 * t) + x2 * t
    y5 = x4 * jnp.sin(x0 * t) * jnp.exp(-x3 * t)

    all_outputs = [y0, y1, y2, y3, y4, y5]
    Y = jnp.stack(all_outputs[:K], axis=-1)  # (N, T, K)

    if T == 1 and K == 1:
        return Y[:, 0, 0]  # (N,)
    if T == 1:
        return Y[:, 0, :]  # (N, K)
    return Y  # (N, T, K)


def coupled_oscillators_numpy(X: np.ndarray, T: int, K: int) -> np.ndarray:
    """Same model, pure numpy for SALib."""
    x0 = X[:, 0, None]
    x1 = X[:, 1, None]
    x2 = X[:, 2, None]
    x3 = X[:, 3, None]
    x4 = X[:, 4, None]

    t = np.linspace(0.0, 5.0, T)[None, :]

    y0 = x0 * np.sin(2 * np.pi * x1 * t) * np.exp(-x2 * t)
    y1 = x1 * np.cos(2 * np.pi * x0 * t) + x4 * t**2
    y2 = x3 * np.sin(x4 * 10 * t) + x0 * x2
    y3 = (x3 + x4) * np.exp(-x0 * t) * np.sin(2 * np.pi * x1 * t)
    y4 = x0 * x3 * np.cos(x1 * t) + x2 * t
    y5 = x4 * np.sin(x0 * t) * np.exp(-x3 * t)

    all_outputs = [y0, y1, y2, y3, y4, y5]
    Y = np.stack(all_outputs[:K], axis=-1)  # (N, T, K)

    if T == 1 and K == 1:
        return Y[:, 0, 0]
    if T == 1:
        return Y[:, 0, :]
    return Y


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
# Correctness (Ishigami)
# ---------------------------------------------------------------------------


def benchmark_correctness() -> bool:
    """Validate gsax analyze and analyze_hdmr against SALib and analytical values."""
    base_n = 1024
    D = ISHIGAMI_PROBLEM.num_vars
    salib_problem = gsax_problem_to_salib(ISHIGAMI_PROBLEM)
    analytical_S1 = np.array(ISHIGAMI_ANALYTICAL_S1)
    analytical_ST = np.array(ISHIGAMI_ANALYTICAL_ST)

    rows: list[tuple[str, str, float, float, float, bool]] = []

    # --- Sobol: gsax vs SALib (shared Saltelli samples) ---
    n_samples = 2**14 * 8
    sr = gsax.sample(ISHIGAMI_PROBLEM, n_samples, seed=42, calc_second_order=True)
    Y_jax = ishigami_evaluate(jnp.asarray(sr.samples))
    Y_np = np.asarray(Y_jax)

    gsax_sobol = gsax.analyze(sr, Y_jax)
    salib_sobol_result = salib_sobol.analyze(
        salib_problem, Y_np, calc_second_order=True, print_to_console=False,
    )

    g_S1 = np.asarray(gsax_sobol.S1)
    g_ST = np.asarray(gsax_sobol.ST)
    s_S1 = salib_sobol_result["S1"]
    s_ST = salib_sobol_result["ST"]

    for i in range(D):
        match = bool(np.abs(g_S1[i] - s_S1[i]) < 1e-6)
        rows.append(("analyze (S2)", f"S1[{i}]", g_S1[i], s_S1[i], analytical_S1[i], match))
    for i in range(D):
        match = bool(np.abs(g_ST[i] - s_ST[i]) < 1e-6)
        rows.append(("analyze (S2)", f"ST[{i}]", g_ST[i], s_ST[i], analytical_ST[i], match))

    # S2 check (upper triangle)
    mask = np.triu(np.ones((D, D), dtype=bool), k=1)
    g_S2 = np.asarray(gsax_sobol.S2)[mask]
    s_S2 = salib_sobol_result["S2"][mask]
    for idx, (g, s) in enumerate(zip(g_S2, s_S2)):
        match = bool(np.abs(g - s) < 1e-6)
        rows.append(("analyze (S2)", f"S2[{idx}]", float(g), float(s), float("nan"), match))

    # --- Sobol gsax vs analytical ---
    for i in range(D):
        if analytical_S1[i] == 0.0:
            ok = abs(g_S1[i]) < 0.05
        else:
            ok = abs(g_S1[i] - analytical_S1[i]) / abs(analytical_S1[i]) < 0.15
        rows.append(("analyze vs analyt.", f"S1[{i}]", g_S1[i], float("nan"), analytical_S1[i], bool(ok)))
    for i in range(D):
        ok = abs(g_ST[i] - analytical_ST[i]) / max(abs(analytical_ST[i]), 0.01) < 0.15
        rows.append(("analyze vs analyt.", f"ST[{i}]", g_ST[i], float("nan"), analytical_ST[i], bool(ok)))

    # --- HDMR: gsax vs SALib ---
    rng = np.random.default_rng(42)
    bounds = np.array(ISHIGAMI_PROBLEM.bounds)
    X_np = rng.uniform(bounds[:, 0], bounds[:, 1], size=(base_n, D))
    Y_hdmr_np = np.asarray(ishigami_evaluate(jnp.asarray(X_np)))
    X_jax_hdmr = jnp.asarray(X_np)
    Y_hdmr_jax = jnp.asarray(Y_hdmr_np)

    gsax_hdmr = gsax.analyze_hdmr(ISHIGAMI_PROBLEM, X_jax_hdmr, Y_hdmr_jax, maxorder=2, m=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        salib_hdmr_result = salib_hdmr.analyze(
            salib_problem, X_np, Y_hdmr_np, maxorder=2, maxiter=100, print_to_console=False,
        )

    g_hdmr_S1 = np.asarray(gsax_hdmr.S1)
    g_hdmr_ST = np.asarray(gsax_hdmr.ST)
    s_hdmr_S1 = np.array(salib_hdmr_result["Sa"][:D])
    s_hdmr_ST = np.array(salib_hdmr_result["ST"][:D])

    for i in range(D):
        match = bool(np.abs(g_hdmr_S1[i] - s_hdmr_S1[i]) < 0.05)
        rows.append(("analyze_hdmr", f"S1[{i}]", g_hdmr_S1[i], s_hdmr_S1[i], analytical_S1[i], match))
    for i in range(D):
        match = bool(np.abs(g_hdmr_ST[i] - s_hdmr_ST[i]) < 0.05)
        rows.append(("analyze_hdmr", f"ST[{i}]", g_hdmr_ST[i], s_hdmr_ST[i], analytical_ST[i], match))

    # --- HDMR gsax vs analytical ---
    for i in range(D):
        if analytical_S1[i] == 0.0:
            ok = abs(g_hdmr_S1[i]) < 0.1
        else:
            ok = abs(g_hdmr_S1[i] - analytical_S1[i]) / abs(analytical_S1[i]) < 0.30
        rows.append(("hdmr vs analyt.", f"S1[{i}]", g_hdmr_S1[i], float("nan"), analytical_S1[i], bool(ok)))
    for i in range(D):
        ok = abs(g_hdmr_ST[i] - analytical_ST[i]) / max(abs(analytical_ST[i]), 0.01) < 0.25
        rows.append(("hdmr vs analyt.", f"ST[{i}]", g_hdmr_ST[i], float("nan"), analytical_ST[i], bool(ok)))

    # --- Print correctness table ---
    all_pass = True
    print("=" * 78)
    print("CORRECTNESS CHECK  (Ishigami)")
    print("=" * 78)
    print(f"{'Method':<22} {'Index':<8} {'gsax':>10} {'SALib':>10} {'analytical':>10} {'match':>6}")
    print("-" * 78)
    for method, index, g_val, s_val, a_val, ok in rows:
        all_pass &= ok
        g_str = f"{g_val:>10.4f}"
        s_str = f"{s_val:>10.4f}" if not np.isnan(s_val) else f"{'---':>10}"
        a_str = f"{a_val:>10.4f}" if not np.isnan(a_val) else f"{'---':>10}"
        tag = "PASS" if ok else "FAIL"
        print(f"{method:<22} {index:<8} {g_str} {s_str} {a_str} {tag:>6}")

    return all_pass


# ---------------------------------------------------------------------------
# Timing benchmark
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("1x1", 1, 1),
    ("1x6", 1, 6),
    ("50x1", 50, 1),
    ("50x6", 50, 6),
]


def _time_gsax_sobol(sr, Y_jax, calc_second_order: bool) -> float:
    """Time gsax.analyze (already JIT-warmed)."""
    t0 = time.perf_counter()
    r = gsax.analyze(sr, Y_jax)
    jax.block_until_ready(r.S1)
    jax.block_until_ready(r.ST)
    if calc_second_order and r.S2 is not None:
        jax.block_until_ready(r.S2)
    return time.perf_counter() - t0


def _time_salib_sobol(salib_problem, Y_np, calc_second_order: bool, T: int, K: int) -> float:
    """Time SALib sobol.analyze, looping over T*K slices."""
    t0 = time.perf_counter()
    if T == 1 and K == 1:
        salib_sobol.analyze(salib_problem, Y_np, calc_second_order=calc_second_order, print_to_console=False)
    elif T == 1:
        for k in range(K):
            salib_sobol.analyze(salib_problem, Y_np[:, k], calc_second_order=calc_second_order, print_to_console=False)
    else:
        for t_idx in range(T):
            for k in range(K):
                salib_sobol.analyze(salib_problem, Y_np[:, t_idx, k], calc_second_order=calc_second_order, print_to_console=False)
    return time.perf_counter() - t0


def _time_gsax_hdmr(problem, X_jax, Y_jax) -> float:
    """Time gsax.analyze_hdmr."""
    t0 = time.perf_counter()
    r = gsax.analyze_hdmr(problem, X_jax, Y_jax, maxorder=2, m=2)
    jax.block_until_ready(r.S1)
    jax.block_until_ready(r.ST)
    return time.perf_counter() - t0


def _time_salib_hdmr(salib_problem, X_np, Y_np, T: int, K: int) -> float:
    """Time SALib hdmr.analyze, looping over T*K slices."""
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if T == 1 and K == 1:
            salib_hdmr.analyze(salib_problem, X_np, Y_np, maxorder=2, maxiter=100, print_to_console=False)
        elif T == 1:
            for k in range(K):
                salib_hdmr.analyze(salib_problem, X_np, Y_np[:, k], maxorder=2, maxiter=100, print_to_console=False)
        else:
            for t_idx in range(T):
                for k_idx in range(K):
                    salib_hdmr.analyze(salib_problem, X_np, Y_np[:, t_idx, k_idx], maxorder=2, maxiter=100, print_to_console=False)
    return time.perf_counter() - t0


def benchmark_timing(base_n: int = 1024) -> None:
    """Run all 4 scenarios x 3 methods, print timing table."""
    D = D_PARAMS
    salib_problem = gsax_problem_to_salib(BENCH_PROBLEM)
    bounds = np.array(BENCH_PROBLEM.bounds)

    print(f"\n{'=' * 78}")
    print("TIMING BENCHMARK — coupled oscillators")
    print(f"  D={D}, base_n={base_n}")
    print("=" * 78)

    # --- JIT warmup for gsax (both analyze and analyze_hdmr) ---
    print("\nWarming up gsax JIT ...", end=" ", flush=True)
    sr_w = gsax.sample(BENCH_PROBLEM, base_n * (2 * D + 2), seed=0, calc_second_order=True)
    Y_w = coupled_oscillators(jnp.asarray(sr_w.samples), T=2, K=2)
    r_w = gsax.analyze(sr_w, Y_w)
    jax.block_until_ready(r_w.S1)

    sr_w_no_s2 = gsax.sample(BENCH_PROBLEM, base_n * (D + 2), seed=0, calc_second_order=False)
    Y_w_no_s2 = coupled_oscillators(jnp.asarray(sr_w_no_s2.samples), T=2, K=2)
    r_w2 = gsax.analyze(sr_w_no_s2, Y_w_no_s2)
    jax.block_until_ready(r_w2.S1)

    rng = np.random.default_rng(0)
    X_w = rng.uniform(bounds[:, 0], bounds[:, 1], size=(base_n, D))
    X_w_jax = jnp.asarray(X_w)
    Y_w_hdmr = coupled_oscillators(X_w_jax, T=2, K=2)
    r_w3 = gsax.analyze_hdmr(BENCH_PROBLEM, X_w_jax, Y_w_hdmr, maxorder=2, m=2)
    jax.block_until_ready(r_w3.S1)
    print("done.")

    # --- Run all scenarios ---
    timing_rows: list[tuple[str, str, float, float, float]] = []

    for scenario_label, T, K in SCENARIOS:
        print(f"\nScenario {scenario_label} (T={T}, K={K}) ...", flush=True)

        for calc_s2, method_label in [(False, "analyze (no S2)"), (True, "analyze (S2)")]:
            sr = gsax.sample(
                BENCH_PROBLEM,
                base_n * ((2 * D + 2) if calc_s2 else (D + 2)),
                seed=1,
                calc_second_order=calc_s2,
            )
            Y_jax = coupled_oscillators(jnp.asarray(sr.samples), T=T, K=K)
            jax.block_until_ready(Y_jax)

            X_salib = salib_sample_sobol.sample(salib_problem, base_n, calc_second_order=calc_s2)
            Y_salib = coupled_oscillators_numpy(X_salib, T=T, K=K)

            g_time = _time_gsax_sobol(sr, Y_jax, calc_s2)
            s_time = _time_salib_sobol(salib_problem, Y_salib, calc_s2, T, K)
            speedup = s_time / g_time if g_time > 0 else float("inf")
            timing_rows.append((scenario_label, method_label, g_time * 1e3, s_time * 1e3, speedup))

        # -- HDMR: shared (X, Y) --
        rng = np.random.default_rng(42)
        X_hdmr_np = rng.uniform(bounds[:, 0], bounds[:, 1], size=(base_n, D))
        X_hdmr_jax = jnp.asarray(X_hdmr_np)
        Y_hdmr_jax = coupled_oscillators(X_hdmr_jax, T=T, K=K)
        jax.block_until_ready(Y_hdmr_jax)
        Y_hdmr_np = np.asarray(Y_hdmr_jax)

        g_time = _time_gsax_hdmr(BENCH_PROBLEM, X_hdmr_jax, Y_hdmr_jax)
        s_time = _time_salib_hdmr(salib_problem, X_hdmr_np, Y_hdmr_np, T, K)
        speedup = s_time / g_time if g_time > 0 else float("inf")
        timing_rows.append((scenario_label, "analyze_hdmr", g_time * 1e3, s_time * 1e3, speedup))

    # --- Print timing table ---
    print(f"\n{'Scenario (TxK)':<16} {'Method':<20} {'gsax (ms)':>12} {'SALib (ms)':>12} {'speedup':>10}")
    print("-" * 72)
    for scenario, method, g_ms, s_ms, sp in timing_rows:
        print(f"{scenario:<16} {method:<20} {g_ms:>10.1f}   {s_ms:>10.1f}   {sp:>8.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    correct = benchmark_correctness()
    benchmark_timing(base_n=1024)

    print("\n" + "=" * 78)
    if correct:
        print("ALL CORRECTNESS CHECKS PASSED")
        return 0
    else:
        print("SOME CORRECTNESS CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
