"""Main Sobol sensitivity analysis computation using JAX."""

import jax
import jax.numpy as jnp
from jax import Array, vmap

from gsax._indices import first_order, second_order, total_order
from gsax.results import SAResult
from gsax.sampling import SamplingResult


def _separate_output_values(
    Y: Array, D: int, calc_second_order: bool
) -> tuple[Array, Array, Array, Array | None]:
    """Separate model outputs into A, B, AB, BA matrices."""
    step = 2 * D + 2 if calc_second_order else D + 2

    A = Y[0::step]  # (base_n, ...)
    B = Y[(step - 1) :: step]  # (base_n, ...)
    AB = jnp.stack([Y[(j + 1) :: step] for j in range(D)], axis=1)  # (base_n, D, ...)

    BA = None
    if calc_second_order:
        BA = jnp.stack([Y[(j + 1 + D) :: step] for j in range(D)], axis=1)  # (base_n, D, ...)

    return A, B, AB, BA


def _kernel_first_total(A: Array, AB: Array, B: Array) -> tuple[Array, Array]:
    """Jittable kernel for first-order and total-order indices.

    Args:
        A: (N,) model outputs from matrix A.
        AB: (N, D) model outputs from AB matrices.
        B: (N,) model outputs from matrix B.

    Returns:
        S1, ST — both (D,).
    """
    D = AB.shape[1]
    S1 = vmap(lambda j: first_order(A, AB[:, j], B))(jnp.arange(D))
    ST = vmap(lambda j: total_order(A, AB[:, j], B))(jnp.arange(D))
    return S1, ST


def _kernel_second_order(
    A: Array, AB: Array, BA: Array, B: Array
) -> tuple[Array, Array, Array]:
    """Jittable kernel for first-, total-, and second-order indices.

    Args:
        A: (N,) model outputs from matrix A.
        AB: (N, D) model outputs from AB matrices.
        BA: (N, D) model outputs from BA matrices.
        B: (N,) model outputs from matrix B.

    Returns:
        S1, ST — both (D,).
        S2 — (D, D).
    """
    D = AB.shape[1]
    S1 = vmap(lambda j: first_order(A, AB[:, j], B))(jnp.arange(D))
    ST = vmap(lambda j: total_order(A, AB[:, j], B))(jnp.arange(D))

    def s2_row(j):
        def s2_elem(k):
            return second_order(A, AB[:, j], AB[:, k], BA[:, j], B)

        return vmap(s2_elem)(jnp.arange(D))

    S2 = vmap(s2_row)(jnp.arange(D))  # (D, D)
    return S1, ST, S2


def _prepare_Y(
    Y: Array,
) -> tuple[Array, bool, bool]:
    """Ensure Y is 3D (n_total, T, K) and track which dims to squeeze back."""
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


def _squeeze_results(
    S1: Array, ST: Array, S2: Array | None,
    squeeze_time: bool, squeeze_output: bool,
    S1_conf: Array | None = None,
    ST_conf: Array | None = None,
    S2_conf: Array | None = None,
) -> tuple[Array, Array, Array | None, Array | None, Array | None, Array | None]:
    """Squeeze singleton T and K dimensions from all result arrays."""
    if squeeze_time and squeeze_output:
        S1 = S1[0, 0]
        ST = ST[0, 0]
        if S2 is not None:
            S2 = S2[0, 0]
        if S1_conf is not None:
            S1_conf = S1_conf[:, 0, 0]
            ST_conf = ST_conf[:, 0, 0]
            if S2_conf is not None:
                S2_conf = S2_conf[:, 0, 0]
    elif squeeze_time:
        S1 = S1[0]
        ST = ST[0]
        if S2 is not None:
            S2 = S2[0]
        if S1_conf is not None:
            S1_conf = S1_conf[:, 0]
            ST_conf = ST_conf[:, 0]
            if S2_conf is not None:
                S2_conf = S2_conf[:, 0]
    return S1, ST, S2, S1_conf, ST_conf, S2_conf


def _analyze_no_bootstrap(
    sampling_result: SamplingResult, Y: Array, *, chunk_size: int
) -> SAResult:
    """Fast path: vmap over T*K, no resampling."""
    Y, squeeze_time, squeeze_output = _prepare_Y(Y)
    D = sampling_result.n_params
    base_n = sampling_result.base_n
    calc_second_order = sampling_result.calc_second_order

    _, T, K = Y.shape
    A, B, AB, BA = _separate_output_values(Y, D, calc_second_order)

    A_flat = A.transpose(1, 2, 0).reshape(T * K, base_n)
    B_flat = B.transpose(1, 2, 0).reshape(T * K, base_n)
    AB_flat = AB.transpose(2, 3, 0, 1).reshape(T * K, base_n, D)

    total = T * K
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    cs = min(chunk_size, total)

    if calc_second_order:
        assert BA is not None
        BA_flat = BA.transpose(2, 3, 0, 1).reshape(T * K, base_n, D)
        batched = jax.jit(jax.vmap(_kernel_second_order, in_axes=(0, 0, 0, 0)))

        s1_parts, st_parts, s2_parts = [], [], []
        for start in range(0, total, cs):
            end = min(start + cs, total)
            s1, st, s2 = batched(
                A_flat[start:end], AB_flat[start:end],
                BA_flat[start:end], B_flat[start:end],
            )
            s1_parts.append(s1)
            st_parts.append(st)
            s2_parts.append(s2)

        S1_out = jnp.concatenate(s1_parts).reshape(T, K, D)
        ST_out = jnp.concatenate(st_parts).reshape(T, K, D)
        S2_out = jnp.concatenate(s2_parts).reshape(T, K, D, D)
    else:
        batched = jax.jit(jax.vmap(_kernel_first_total, in_axes=(0, 0, 0)))

        s1_parts, st_parts = [], []
        for start in range(0, total, cs):
            end = min(start + cs, total)
            s1, st = batched(
                A_flat[start:end], AB_flat[start:end], B_flat[start:end],
            )
            s1_parts.append(s1)
            st_parts.append(st)

        S1_out = jnp.concatenate(s1_parts).reshape(T, K, D)
        ST_out = jnp.concatenate(st_parts).reshape(T, K, D)
        S2_out = None

    S1_out, ST_out, S2_out, _, _, _ = _squeeze_results(
        S1_out, ST_out, S2_out, squeeze_time, squeeze_output
    )
    return SAResult(S1=S1_out, ST=ST_out, S2=S2_out, problem=sampling_result.problem)


def _analyze_bootstrap(
    sampling_result: SamplingResult,
    Y: Array,
    *,
    num_resamples: int,
    conf_level: float,
    key: Array,
    chunk_size: int,
) -> SAResult:
    """Bootstrap path: loop over T*K, vmap over R resamples (chunked)."""
    from gsax._bootstrap import _bootstrap_first_total, _bootstrap_second_order

    Y, squeeze_time, squeeze_output = _prepare_Y(Y)
    D = sampling_result.n_params
    base_n = sampling_result.base_n
    calc_second_order = sampling_result.calc_second_order

    _, T, K = Y.shape
    A, B, AB, BA = _separate_output_values(Y, D, calc_second_order)
    # A: (base_n, T, K), AB: (base_n, D, T, K)

    # Generate shared bootstrap indices once: (R, base_n)
    indices = jax.random.randint(key, shape=(num_resamples, base_n), minval=0, maxval=base_n)

    alpha = (1.0 - conf_level) / 2.0
    percentiles = jnp.array([alpha * 100, (1.0 - alpha) * 100])

    # JIT the point-estimate kernels (same shapes → compiled once)
    jit_ft = jax.jit(_kernel_first_total)
    jit_so = jax.jit(_kernel_second_order) if calc_second_order else None

    # Accumulators for (T, K, ...) results
    S1_list, ST_list = [], []
    S1_lo_list, S1_hi_list = [], []
    ST_lo_list, ST_hi_list = [], []
    S2_list, S2_lo_list, S2_hi_list = [], [], []

    for t in range(T):
        for k in range(K):
            # Extract 1D slices for this (t, k)
            a = A[:, t, k]       # (base_n,)
            b = B[:, t, k]       # (base_n,)
            ab = AB[:, :, t, k]  # (base_n, D)

            if calc_second_order:
                assert BA is not None
                ba = BA[:, :, t, k]  # (base_n, D)

                s1, st, s2 = jit_so(a, ab, ba, b)
                S2_list.append(s2)

                s1_boot, st_boot, s2_boot = _bootstrap_second_order(
                    indices, a, ab, ba, b, chunk_size
                )
                s2_ci = jnp.percentile(s2_boot, percentiles, axis=0)  # (2, D, D)
                S2_lo_list.append(s2_ci[0])
                S2_hi_list.append(s2_ci[1])
            else:
                s1, st = jit_ft(a, ab, b)
                s1_boot, st_boot = _bootstrap_first_total(
                    indices, a, ab, b, chunk_size
                )

            S1_list.append(s1)
            ST_list.append(st)

            s1_ci = jnp.percentile(s1_boot, percentiles, axis=0)  # (2, D)
            st_ci = jnp.percentile(st_boot, percentiles, axis=0)  # (2, D)
            S1_lo_list.append(s1_ci[0])
            S1_hi_list.append(s1_ci[1])
            ST_lo_list.append(st_ci[0])
            ST_hi_list.append(st_ci[1])

    S1_out = jnp.stack(S1_list).reshape(T, K, D)
    ST_out = jnp.stack(ST_list).reshape(T, K, D)
    S1_conf = jnp.stack([
        jnp.stack(S1_lo_list).reshape(T, K, D),
        jnp.stack(S1_hi_list).reshape(T, K, D),
    ])  # (2, T, K, D)
    ST_conf = jnp.stack([
        jnp.stack(ST_lo_list).reshape(T, K, D),
        jnp.stack(ST_hi_list).reshape(T, K, D),
    ])

    if calc_second_order:
        S2_out = jnp.stack(S2_list).reshape(T, K, D, D)
        S2_conf = jnp.stack([
            jnp.stack(S2_lo_list).reshape(T, K, D, D),
            jnp.stack(S2_hi_list).reshape(T, K, D, D),
        ])
    else:
        S2_out = None
        S2_conf = None

    S1_out, ST_out, S2_out, S1_conf, ST_conf, S2_conf = _squeeze_results(
        S1_out, ST_out, S2_out, squeeze_time, squeeze_output,
        S1_conf, ST_conf, S2_conf,
    )
    return SAResult(
        S1=S1_out, ST=ST_out, S2=S2_out, problem=sampling_result.problem,
        S1_conf=S1_conf, ST_conf=ST_conf, S2_conf=S2_conf,
    )


def analyze(
    sampling_result: SamplingResult,
    Y: Array,
    *,
    num_resamples: int = 0,
    conf_level: float = 0.95,
    key: Array | None = None,
    chunk_size: int = 2048,
) -> SAResult:
    """Compute Sobol sensitivity indices using JAX.

    Args:
        sampling_result: Result from gsax.sample().
        Y: Model outputs. Shape (n_total,), (n_total, K), or (n_total, T, K).
        num_resamples: Number of bootstrap resamples. 0 disables bootstrap.
        conf_level: Confidence level for bootstrap CIs (default 0.95).
        key: JAX PRNG key. Required when ``num_resamples > 0``.
        chunk_size: Number of (T, K) combos (no bootstrap) or resamples
            (bootstrap) to process per vmap batch. Defaults to 2048.

    Returns:
        SAResult with S1, ST, and optionally S2 and confidence intervals.
    """
    Y = jnp.asarray(Y)

    if num_resamples > 0:
        if key is None:
            raise ValueError("key is required when num_resamples > 0")
        return _analyze_bootstrap(
            sampling_result, Y,
            num_resamples=num_resamples,
            conf_level=conf_level,
            key=key,
            chunk_size=chunk_size,
        )

    return _analyze_no_bootstrap(sampling_result, Y, chunk_size=chunk_size)
