"""RS-HDMR (Random Sampling High-Dimensional Model Representation) analysis.

Provides ``analyze_hdmr`` for computing ANCOVA-based sensitivity indices from
arbitrary (X, Y) pairs using B-spline surrogate modelling, and ``emulate_hdmr``
for prediction with the fitted surrogate.
"""

import itertools

import jax.numpy as jnp
from jax import Array

from gsax._hdmr import (
    _build_B1,
    _build_B2,
    _build_B3,
    _compute_f_crits,
    _make_hdmr_kernel,
)
from gsax.problem import Problem
from gsax.results_hdmr import HDMRResult


def _normalize_X(X: Array, problem: Problem) -> Array:
    """Normalize X to [0, 1] using problem bounds."""
    bounds = jnp.array(problem.bounds)  # (D, 2)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return (X - lo) / (hi - lo)


def _build_terms(D: int, maxorder: int) -> tuple:
    """Build parameter combination arrays and term labels."""
    c1 = list(range(D))
    n1 = D
    c2, c3 = [], []
    n2, n3 = 0, 0
    if maxorder >= 2:
        c2 = list(itertools.combinations(range(D), 2))
        n2 = len(c2)
    if maxorder >= 3:
        c3 = list(itertools.combinations(range(D), 3))
        n3 = len(c3)
    n = n1 + n2 + n3
    return c1, c2, c3, n1, n2, n3, n


def _build_term_labels(
    problem: Problem, c1: list, c2: list, c3: list,
) -> tuple[str, ...]:
    """Build human-readable term labels."""
    names = problem.names
    labels = [names[i] for i in c1]
    labels += ["/".join(names[i] for i in combo) for combo in c2]
    labels += ["/".join(names[i] for i in combo) for combo in c3]
    return tuple(labels)


def _compute_ST(
    S: Array, c1: list, c2: list, c3: list, D: int,
) -> Array:
    """Compute total-order indices by summing S over terms involving each param."""
    n1 = len(c1)
    ST = S[:n1]  # First order: term r = parameter r

    offset = n1
    for j, (a, b) in enumerate(c2):
        ST = ST.at[a].add(S[offset + j])
        ST = ST.at[b].add(S[offset + j])

    offset = n1 + len(c2)
    for j, combo in enumerate(c3):
        for p in combo:
            ST = ST.at[p].add(S[offset + j])

    return ST


def _prepare_Y(Y: Array) -> tuple[Array, bool, bool]:
    """Promote Y to 3-D (N, T, K). Same pattern as analyze._prepare_Y."""
    squeeze_time = False
    squeeze_output = False
    if Y.ndim == 1:
        Y = Y[:, None, None]
        squeeze_time = True
        squeeze_output = True
    elif Y.ndim == 2:
        Y = Y[:, None, :]
        squeeze_time = True
    return Y, squeeze_time, squeeze_output


def _squeeze_hdmr(
    Sa: Array, Sb: Array, S: Array, ST: Array,
    squeeze_time: bool, squeeze_output: bool,
) -> tuple:
    """Remove singleton T/K dims from HDMR result arrays."""
    arrays = [Sa, Sb, S, ST]

    if squeeze_time and squeeze_output:
        arrays = [a[0, 0] for a in arrays]
    elif squeeze_time:
        arrays = [a[0] for a in arrays]

    return tuple(arrays)


def analyze_hdmr(
    problem: Problem,
    X: Array,
    Y: Array,
    *,
    maxorder: int = 2,
    maxiter: int = 100,
    m: int = 2,
    subsample_size: int | None = None,
    lambdax: float = 0.01,
) -> HDMRResult:
    """Compute sensitivity indices via RS-HDMR with B-spline surrogate modelling.

    Works with **any** set of (X, Y) pairs -- no structured sampling required.
    Decomposes the input-output relationship into hierarchical component
    functions using B-spline regression, then derives ANCOVA-based sensitivity
    indices.

    Args:
        problem: Parameter names and bounds.
        X: (N, D) input samples.
        Y: (N,), (N, K), or (N, T, K) model outputs.
        maxorder: Maximum HDMR expansion order (1, 2, or 3).
        maxiter: Maximum backfitting iterations for first-order terms.
        m: Number of B-spline intervals (basis size = m + 3 per dimension).
        subsample_size: Subsample size R (default: use all N samples).
        lambdax: Tikhonov regularization parameter.

    Returns:
        HDMRResult with Sa, Sb, S, ST, emulator, etc.
    """
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)

    N, D = X.shape
    if D != problem.num_vars:
        raise ValueError(
            f"X has {D} columns but problem defines {problem.num_vars} parameters"
        )
    if N < 300:
        raise ValueError(f"Need at least 300 samples, got {N}")
    if maxorder not in (1, 2, 3):
        raise ValueError(f"maxorder must be 1, 2, or 3, got {maxorder}")
    if D == 2 and maxorder > 2:
        raise ValueError("maxorder must be <= 2 when D = 2")

    R = subsample_size if subsample_size is not None else N

    # Build terms
    c1, c2, c3, n1, n2, n3, n = _build_terms(D, maxorder)
    term_labels = _build_term_labels(problem, c1, c2, c3)
    m1 = m + 3
    m2 = m1 ** 2
    m3 = m1 ** 3

    # Normalize X
    X_n = _normalize_X(X, problem)

    # Build B-spline bases
    B1 = _build_B1(X_n, m)  # (N, m1, D)
    B2 = _build_B2(B1, jnp.array(c2, dtype=int), m1) if n2 > 0 else jnp.zeros((N, 1, 1))
    B3 = _build_B3(B1, jnp.array(c3, dtype=int), m1) if n3 > 0 else jnp.zeros((N, 1, 1))

    # Precompute F critical values
    f_crits = _compute_f_crits(0.95, m1, m2, m3, R)

    # Use all samples
    indices = jnp.arange(N)

    # Promote Y to 3D
    Y_3d, squeeze_time, squeeze_output = _prepare_Y(Y)
    _, T, K_out = Y_3d.shape

    # Create kernel (JIT-compiled, all sizes captured in closure)
    kernel = _make_hdmr_kernel(
        maxorder, m1, n1, maxiter, m2, m3, n2, n3, n, lambdax, R,
    )

    # Accumulators for all (t, k) combos
    Sa_all, Sb_all, S_all, ST_all = [], [], [], []
    select_sum = jnp.zeros(n)
    rmse_all = []
    C1_sum = jnp.zeros((m1, n1))
    C2_sum = jnp.zeros((m2, n2)) if n2 > 0 else jnp.zeros((1, 1))
    C3_sum = jnp.zeros((m3, n3)) if n3 > 0 else jnp.zeros((1, 1))
    f0_sum = 0.0
    n_fits = 0

    for t in range(T):
        for k in range(K_out):
            Y_tk = Y_3d[:, t, k]  # (N,)

            B1_sub = B1[indices]
            B2_sub = B2[indices] if n2 > 0 else jnp.zeros((R, 1, 1))
            B3_sub = B3[indices] if n3 > 0 else jnp.zeros((R, 1, 1))
            Y_sub = Y_tk[indices]

            sa, sb, s, sel, rmse_val, c1_coef, c2_coef, c3_coef, f0_val = kernel(
                B1_sub, B2_sub, B3_sub, Y_sub, f_crits,
            )

            Sa_all.append(sa)
            Sb_all.append(sb)
            S_all.append(s)
            ST_all.append(_compute_ST(s, c1, c2, c3, D))

            select_sum = select_sum + sel
            rmse_all.append(rmse_val)

            C1_sum = C1_sum + c1_coef
            if n2 > 0:
                C2_sum = C2_sum + c2_coef
            if n3 > 0:
                C3_sum = C3_sum + c3_coef
            f0_sum = f0_sum + float(f0_val)
            n_fits += 1

    # Reshape to (T, K, n_terms) / (T, K, D)
    Sa_out = jnp.stack(Sa_all).reshape(T, K_out, n)
    Sb_out = jnp.stack(Sb_all).reshape(T, K_out, n)
    S_out = jnp.stack(S_all).reshape(T, K_out, n)
    ST_out = jnp.stack(ST_all).reshape(T, K_out, D)

    # Squeeze
    Sa_out, Sb_out, S_out, ST_out = _squeeze_hdmr(
        Sa_out, Sb_out, S_out, ST_out,
        squeeze_time, squeeze_output,
    )

    # Build emulator dict
    emulator = {
        "C1": C1_sum / n_fits,
        "C2": C2_sum / n_fits if n2 > 0 else None,
        "C3": C3_sum / n_fits if n3 > 0 else None,
        "f0": f0_sum / n_fits,
        "m": m,
        "maxorder": maxorder,
        "c2": c2,
        "c3": c3,
    }

    return HDMRResult(
        Sa=Sa_out,
        Sb=Sb_out,
        S=S_out,
        ST=ST_out,
        problem=problem,
        terms=term_labels,
        emulator=emulator,
        select=select_sum,
        rmse=jnp.stack(rmse_all) if rmse_all else None,
    )


def emulate_hdmr(result: HDMRResult, X_new: Array) -> Array:
    """Predict at new input points using the fitted HDMR surrogate.

    Args:
        result: HDMRResult from ``analyze_hdmr`` (must have ``emulator`` set).
        X_new: (N_new, D) new input points within the problem bounds.

    Returns:
        Y_pred: (N_new,) predicted outputs.
    """
    em = result.emulator
    if em is None:
        raise ValueError("HDMRResult has no emulator (emulator is None)")

    X_new = jnp.asarray(X_new)
    maxorder = em["maxorder"]
    C1 = em["C1"]
    f0 = em["f0"]
    m1 = em["m"] + 3

    # Normalize
    X_n = _normalize_X(X_new, result.problem)

    # Build bases and compute predictions
    B1 = _build_B1(X_n, em["m"])  # (N_new, m1, D)
    Y_total = jnp.sum(jnp.einsum('rmj,mj->rj', B1, C1), axis=1)

    if maxorder >= 2 and em["C2"] is not None:
        B2 = _build_B2(B1, jnp.array(em["c2"], dtype=int), m1)
        Y_total = Y_total + jnp.sum(jnp.einsum('rmj,mj->rj', B2, em["C2"]), axis=1)

    if maxorder >= 3 and em["C3"] is not None:
        B3 = _build_B3(B1, jnp.array(em["c3"], dtype=int), m1)
        Y_total = Y_total + jnp.sum(jnp.einsum('rmj,mj->rj', B3, em["C3"]), axis=1)

    return Y_total + f0
