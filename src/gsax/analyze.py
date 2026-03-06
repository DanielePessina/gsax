"""Main Sobol sensitivity analysis computation using JAX."""

import jax
import jax.numpy as jnp
from jax import Array, vmap

from gsax._bootstrap import bootstrap_indices
from gsax._indices import first_order, second_order, total_order
from gsax.results import SAResult
from gsax.sampling import SamplingResult


def _separate_output_values(
    Y: Array, D: int, base_n: int, calc_second_order: bool
) -> tuple[Array, Array, Array, Array | None]:
    """Separate model outputs into A, B, AB, BA matrices.

    Args:
        Y: (n_total, ...) model outputs
        D: number of parameters
        base_n: base sample count
        calc_second_order: whether second-order indices are computed

    Returns:
        A: (base_n, ...)
        B: (base_n, ...)
        AB: (base_n, D, ...)
        BA: (base_n, D, ...) or None
    """
    step = 2 * D + 2 if calc_second_order else D + 2

    A = Y[0::step]  # (base_n, ...)
    B = Y[(step - 1) :: step]  # (base_n, ...)
    AB = jnp.stack([Y[(j + 1) :: step] for j in range(D)], axis=1)  # (base_n, D, ...)

    BA = None
    if calc_second_order:
        BA = jnp.stack([Y[(j + 1 + D) :: step] for j in range(D)], axis=1)  # (base_n, D, ...)

    return A, B, AB, BA


def _compute_for_single_output(
    A: Array,
    AB: Array,
    BA: Array | None,
    B: Array,
    key: Array,
    num_resamples: int,
    conf_level: float,
    calc_second_order: bool,
) -> tuple[Array, Array, Array, Array, Array | None, Array | None]:
    """Compute indices for a single output at a single timestep.

    Args:
        A: (N,) model outputs from matrix A.
        AB: (N, D) model outputs from AB matrices.
        BA: (N, D) model outputs from BA matrices, or None.
        B: (N,) model outputs from matrix B.
        key: JAX PRNG key for bootstrap resampling.
        num_resamples: Number of bootstrap resamples.
        conf_level: Confidence level for bootstrap CIs.
        calc_second_order: Whether to compute second-order indices.

    Returns:
        S1, ST: (D,)
        S1_conf, ST_conf: (D,)
        S2: (D, D) or None
        S2_conf: (D, D) or None
    """
    D = AB.shape[1]

    S1 = vmap(lambda j: first_order(A, AB[:, j], B))(jnp.arange(D))
    ST = vmap(lambda j: total_order(A, AB[:, j], B))(jnp.arange(D))

    S1_conf, ST_conf, S2_conf = bootstrap_indices(
        key,
        A,
        AB,
        BA if calc_second_order else None,
        B,
        num_resamples=num_resamples,
        conf_level=conf_level,
    )

    S2 = None
    if calc_second_order and BA is not None:

        def s2_row(j):
            def s2_elem(k):
                return second_order(A, AB[:, j], AB[:, k], BA[:, j], B)

            return vmap(s2_elem)(jnp.arange(D))

        S2 = vmap(s2_row)(jnp.arange(D))  # (D, D)

    return S1, ST, S1_conf, ST_conf, S2, S2_conf


def analyze(
    sampling_result: SamplingResult,
    Y: Array,
    *,
    key: Array,
    num_resamples: int = 100,
    conf_level: float = 0.95,
) -> SAResult:
    """Compute Sobol sensitivity indices using JAX.

    Args:
        sampling_result: Result from gsax.sample()
        Y: Model outputs. Shape (n_total,), (n_total, K), or (n_total, T, K)
        key: JAX PRNG key for bootstrap
        num_resamples: Number of bootstrap resamples
        conf_level: Confidence level for bootstrap CIs

    Returns:
        SAResult with S1, ST, S2 and their confidence intervals
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

    n_total, T, K = Y.shape

    # Separate into A, B, AB, BA
    A, B, AB, BA = _separate_output_values(Y, D, base_n, calc_second_order)
    # A: (base_n, T, K), AB: (base_n, D, T, K), etc.

    # Generate keys for each (t, k) combination
    all_keys = jax.random.split(key, T * K)  # (T*K,) key array

    # We need to iterate over T and K dimensions
    def compute_for_time_output(t, k):
        a = A[:, t, k]  # (base_n,)
        b = B[:, t, k]  # (base_n,)
        ab = AB[:, :, t, k]  # (base_n, D)
        ba = BA[:, :, t, k] if BA is not None else None
        subkey = all_keys[t * K + k]

        return _compute_for_single_output(
            a,
            ab,
            ba,
            b,
            subkey,
            num_resamples=num_resamples,
            conf_level=conf_level,
            calc_second_order=calc_second_order,
        )

    # Collect results with loops (vmap over indices is tricky with optional BA)
    S1_all = []
    ST_all = []
    S1_conf_all = []
    ST_conf_all = []
    S2_all = []
    S2_conf_all = []

    for t in range(T):
        s1_k, st_k, s1c_k, stc_k, s2_k, s2c_k = [], [], [], [], [], []
        for k_idx in range(K):
            s1, st, s1c, stc, s2, s2c = compute_for_time_output(t, k_idx)
            s1_k.append(s1)
            st_k.append(st)
            s1c_k.append(s1c)
            stc_k.append(stc)
            if s2 is not None:
                s2_k.append(s2)
            if s2c is not None:
                s2c_k.append(s2c)

        S1_all.append(jnp.stack(s1_k))  # (K, D)
        ST_all.append(jnp.stack(st_k))  # (K, D)
        S1_conf_all.append(jnp.stack(s1c_k))  # (K, D)
        ST_conf_all.append(jnp.stack(stc_k))  # (K, D)
        if s2_k:
            S2_all.append(jnp.stack(s2_k))  # (K, D, D)
            S2_conf_all.append(jnp.stack(s2c_k))  # (K, D, D)

    S1_out = jnp.stack(S1_all)  # (T, K, D)
    ST_out = jnp.stack(ST_all)  # (T, K, D)
    S1_conf_out = jnp.stack(S1_conf_all)
    ST_conf_out = jnp.stack(ST_conf_all)
    S2_out = jnp.stack(S2_all) if S2_all else None  # (T, K, D, D)
    S2_conf_out = jnp.stack(S2_conf_all) if S2_conf_all else None

    # Squeeze dimensions as needed
    if squeeze_time and squeeze_output:
        S1_out = S1_out[0, 0]  # (D,)
        ST_out = ST_out[0, 0]
        S1_conf_out = S1_conf_out[0, 0]
        ST_conf_out = ST_conf_out[0, 0]
        if S2_out is not None:
            S2_out = S2_out[0, 0]  # (D, D)
            S2_conf_out = S2_conf_out[0, 0]
    elif squeeze_time:
        S1_out = S1_out[0]  # (K, D)
        ST_out = ST_out[0]
        S1_conf_out = S1_conf_out[0]
        ST_conf_out = ST_conf_out[0]
        if S2_out is not None:
            S2_out = S2_out[0]  # (K, D, D)
            S2_conf_out = S2_conf_out[0]

    return SAResult(
        S1=S1_out,
        S1_conf=S1_conf_out,
        ST=ST_out,
        ST_conf=ST_conf_out,
        S2=S2_out,
        S2_conf=S2_conf_out,
        problem=sampling_result.problem,
    )
