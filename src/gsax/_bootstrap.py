"""Bootstrap confidence intervals for Sobol sensitivity indices."""

import jax
import jax.numpy as jnp
from jax import Array, vmap

from gsax._indices import first_order, second_order, total_order


def bootstrap_indices(
    key: Array,
    A: Array,
    AB: Array,
    BA: Array | None,
    B: Array,
    num_resamples: int = 100,
    conf_level: float = 0.95,
) -> tuple[Array, Array, Array | None]:
    """Compute bootstrap confidence intervals for S1, ST, and optionally S2.

    Args:
        key: JAX PRNG key
        A: (N,) model outputs for A matrix
        AB: (N, D) model outputs for AB matrices
        BA: (N, D) model outputs for BA matrices, or None if no second order
        B: (N,) model outputs for B matrix
        num_resamples: number of bootstrap resamples
        conf_level: confidence level (e.g. 0.95)

    Returns:
        S1_conf: (D,) confidence half-widths for first-order indices
        ST_conf: (D,) confidence half-widths for total-order indices
        S2_conf: (D, D) confidence half-widths for second-order indices, or None
    """
    N = A.shape[0]
    D = AB.shape[1]
    Z = jax.scipy.stats.norm.ppf(0.5 + conf_level / 2)

    keys = jax.random.split(key, num_resamples)
    r = vmap(lambda k: jax.random.randint(k, (N,), 0, N))(keys)  # (num_resamples, N)

    def compute_one_resample(idx):
        A_r = A[idx]
        B_r = B[idx]
        AB_r = AB[idx]  # (N, D)
        s1 = vmap(lambda j: first_order(A_r, AB_r[:, j], B_r))(jnp.arange(D))
        st = vmap(lambda j: total_order(A_r, AB_r[:, j], B_r))(jnp.arange(D))
        return s1, st

    all_s1, all_st = vmap(compute_one_resample)(r)  # (num_resamples, D)
    S1_conf = Z * jnp.std(all_s1, axis=0, ddof=1)
    ST_conf = Z * jnp.std(all_st, axis=0, ddof=1)

    if BA is None:
        return S1_conf, ST_conf, None

    def compute_s2_resample(idx):
        A_r = A[idx]
        B_r = B[idx]
        AB_r = AB[idx]  # (N, D)
        BA_r = BA[idx]  # (N, D)

        def s2_row(j):
            def s2_elem(k):
                return second_order(A_r, AB_r[:, j], AB_r[:, k], BA_r[:, j], B_r)

            return vmap(s2_elem)(jnp.arange(D))

        return vmap(s2_row)(jnp.arange(D))  # (D, D)

    all_s2 = vmap(compute_s2_resample)(r)  # (num_resamples, D, D)
    S2_conf = Z * jnp.std(all_s2, axis=0, ddof=1)

    return S1_conf, ST_conf, S2_conf
