import jax.numpy as jnp
from jax import Array


def first_order(A: Array, AB_j: Array, B: Array) -> Array:
    """S1_j = mean(B * (AB_j - A)) / var([A, B])"""
    y = jnp.concatenate([A, B])
    return jnp.mean(B * (AB_j - A)) / jnp.var(y)


def total_order(A: Array, AB_j: Array, B: Array) -> Array:
    """ST_j = 0.5 * mean((A - AB_j)^2) / var([A, B])"""
    y = jnp.concatenate([A, B])
    return 0.5 * jnp.mean((A - AB_j) ** 2) / jnp.var(y)


def second_order(A: Array, AB_j: Array, AB_k: Array, BA_j: Array, B: Array) -> Array:
    """S2_jk = mean(BA_j * AB_k - A * B) / var([A,B]) - S1_j - S1_k"""
    y = jnp.concatenate([A, B])
    Vjk = jnp.mean(BA_j * AB_k - A * B) / jnp.var(y)
    Sj = first_order(A, AB_j, B)
    Sk = first_order(A, AB_k, B)
    return Vjk - Sj - Sk
