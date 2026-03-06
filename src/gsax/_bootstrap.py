"""Bootstrap resampling kernels for confidence intervals on Sobol indices."""

import jax
import jax.numpy as jnp
from jax import Array

from gsax.analyze import _kernel_first_total, _kernel_second_order


@jax.jit
def _resample_ft(idx_chunk: Array, A: Array, AB: Array, B: Array):
    """vmap _kernel_first_total over a chunk of bootstrap index sets."""
    def single(idx):
        return _kernel_first_total(A[idx], AB[idx], B[idx])
    return jax.vmap(single)(idx_chunk)


@jax.jit
def _resample_so(idx_chunk: Array, A: Array, AB: Array, BA: Array, B: Array):
    """vmap _kernel_second_order over a chunk of bootstrap index sets."""
    def single(idx):
        return _kernel_second_order(A[idx], AB[idx], BA[idx], B[idx])
    return jax.vmap(single)(idx_chunk)


def _bootstrap_first_total(
    indices: Array, A: Array, AB: Array, B: Array, chunk_size: int
) -> tuple[Array, Array]:
    """Bootstrap first-order and total-order indices over R resamples.

    Args:
        indices: (R, base_n) resampling indices.
        A: (base_n,) model outputs from matrix A.
        AB: (base_n, D) model outputs from AB matrices.
        B: (base_n,) model outputs from matrix B.
        chunk_size: Number of resamples to vmap per chunk.

    Returns:
        S1_boot: (R, D), ST_boot: (R, D).
    """
    R = indices.shape[0]
    s1_parts, st_parts = [], []
    cs = min(chunk_size, R)
    for start in range(0, R, cs):
        end = min(start + cs, R)
        s1, st = _resample_ft(indices[start:end], A, AB, B)
        s1_parts.append(s1)
        st_parts.append(st)

    return jnp.concatenate(s1_parts), jnp.concatenate(st_parts)


def _bootstrap_second_order(
    indices: Array, A: Array, AB: Array, BA: Array, B: Array, chunk_size: int
) -> tuple[Array, Array, Array]:
    """Bootstrap first-, total-, and second-order indices over R resamples.

    Args:
        indices: (R, base_n) resampling indices.
        A: (base_n,) model outputs from matrix A.
        AB: (base_n, D) model outputs from AB matrices.
        BA: (base_n, D) model outputs from BA matrices.
        B: (base_n,) model outputs from matrix B.
        chunk_size: Number of resamples to vmap per chunk.

    Returns:
        S1_boot: (R, D), ST_boot: (R, D), S2_boot: (R, D, D).
    """
    R = indices.shape[0]
    s1_parts, st_parts, s2_parts = [], [], []
    cs = min(chunk_size, R)
    for start in range(0, R, cs):
        end = min(start + cs, R)
        s1, st, s2 = _resample_so(indices[start:end], A, AB, BA, B)
        s1_parts.append(s1)
        st_parts.append(st)
        s2_parts.append(s2)

    return (
        jnp.concatenate(s1_parts),
        jnp.concatenate(st_parts),
        jnp.concatenate(s2_parts),
    )
