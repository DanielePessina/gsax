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


def analyze(
    sampling_result: SamplingResult,
    Y: Array,
    *,
    chunk_size: int | None = None,
) -> SAResult:
    """Compute Sobol sensitivity indices using JAX.

    Args:
        sampling_result: Result from gsax.sample().
        Y: Model outputs. Shape (n_total,), (n_total, K), or (n_total, T, K).
        chunk_size: Number of (T, K) output combinations to process per
            vmap batch.  ``None`` (default) processes all at once.

    Returns:
        SAResult with S1, ST, and optionally S2.
    """
    Y = jnp.asarray(Y)
    D = sampling_result.n_params
    base_n = sampling_result.base_n
    calc_second_order = sampling_result.calc_second_order

    # Handle dimensionality: ensure Y is 3D (n_total, T, K)
    squeeze_time = False
    if Y.ndim == 1:
        Y = Y[:, None, None]  # (n_total, 1, 1)
        squeeze_time = True
        squeeze_output = True
    elif Y.ndim == 2:
        Y = Y[:, None, :]  # (n_total, 1, K)
        squeeze_time = True
        squeeze_output = False
    else:
        squeeze_output = False

    _, T, K = Y.shape

    # Separate into A, B, AB, BA
    A, B, AB, BA = _separate_output_values(Y, D, calc_second_order)
    # A: (base_n, T, K), AB: (base_n, D, T, K), etc.

    # Flatten (T, K) into a single batch dimension for vmap
    # A: (base_n, T, K) -> (T*K, base_n)
    A_flat = A.transpose(1, 2, 0).reshape(T * K, base_n)
    B_flat = B.transpose(1, 2, 0).reshape(T * K, base_n)
    # AB: (base_n, D, T, K) -> (T*K, base_n, D)
    AB_flat = AB.transpose(2, 3, 0, 1).reshape(T * K, base_n, D)

    total = T * K
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    cs = total if chunk_size is None else chunk_size

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

    # Squeeze dimensions as needed
    if squeeze_time and squeeze_output:
        S1_out = S1_out[0, 0]  # (D,)
        ST_out = ST_out[0, 0]
        if S2_out is not None:
            S2_out = S2_out[0, 0]  # (D, D)
    elif squeeze_time:
        S1_out = S1_out[0]  # (K, D)
        ST_out = ST_out[0]
        if S2_out is not None:
            S2_out = S2_out[0]  # (K, D, D)

    return SAResult(
        S1=S1_out,
        ST=ST_out,
        S2=S2_out,
        problem=sampling_result.problem,
    )
