"""Sobol sensitivity index formulas (first-order, total-order, second-order)."""

import jax.numpy as jnp
from jax import Array


def first_order(A: Array, AB_j: Array, B: Array) -> Array:
    """Compute the first-order Sobol sensitivity index for parameter j.

    S1_j = mean(B * (AB_j - A)) / var([A, B])

    Args:
        A: (N,) model outputs from matrix A.
        AB_j: (N,) model outputs from the AB_j matrix.
        B: (N,) model outputs from matrix B.

    Returns:
        Scalar Array with the first-order index.
    """
    y = jnp.concatenate([A, B])
    var = jnp.var(y)
    numerator = jnp.mean(B * (AB_j - A))
    return jnp.where(var == 0, jnp.nan, numerator / var)


def total_order(A: Array, AB_j: Array, B: Array) -> Array:
    """Compute the total-order Sobol sensitivity index for parameter j.

    ST_j = 0.5 * mean((A - AB_j)^2) / var([A, B])

    Args:
        A: (N,) model outputs from matrix A.
        AB_j: (N,) model outputs from the AB_j matrix.
        B: (N,) model outputs from matrix B.

    Returns:
        Scalar Array with the total-order index.
    """
    y = jnp.concatenate([A, B])
    var = jnp.var(y)
    numerator = 0.5 * jnp.mean((A - AB_j) ** 2)
    return jnp.where(var == 0, jnp.nan, numerator / var)


def second_order(A: Array, AB_j: Array, AB_k: Array, BA_j: Array, B: Array) -> Array:
    """Compute the second-order Sobol interaction index for parameters j and k.

    S2_jk = mean(BA_j * AB_k - A * B) / var([A,B]) - S1_j - S1_k

    Args:
        A: (N,) model outputs from matrix A.
        AB_j: (N,) model outputs from the AB_j matrix.
        AB_k: (N,) model outputs from the AB_k matrix.
        BA_j: (N,) model outputs from the BA_j matrix.
        B: (N,) model outputs from matrix B.

    Returns:
        Scalar Array with the second-order interaction index.
    """
    y = jnp.concatenate([A, B])
    var = jnp.var(y)
    Vjk = jnp.where(var == 0, jnp.nan, jnp.mean(BA_j * AB_k - A * B) / var)
    Sj = first_order(A, AB_j, B)
    Sk = first_order(A, AB_k, B)
    return Vjk - Sj - Sk
