"""Main Sobol sensitivity analysis computation using JAX.

This module implements the Saltelli sampling-based Sobol variance decomposition.
Model outputs Y are split into base matrices A and B and their cross-matrices
AB (and BA for second-order), then first-order (S1), total-order (ST), and
optionally second-order (S2) Sobol indices are computed.

Array shape conventions used throughout:
    N  — number of base Sobol samples (base_n after cleaning)
    D  — number of input parameters
    T  — number of time steps (singleton-squeezed when absent)
    K  — number of output variables (singleton-squeezed when absent)
    R  — number of bootstrap resamples
    step — rows per Saltelli group: 2D+2 (second order) or D+2 (first only)
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import Array, vmap

from gsax._indices import first_order, second_order, total_order
from gsax.results import SAResult
from gsax.sampling import SamplingResult, _saltelli_step


@lru_cache(maxsize=None)
def _get_no_bootstrap_batched_kernel(calc_second_order: bool):
    """Cache final batched Sobol wrappers keyed only on order flag."""
    if calc_second_order:
        return jax.jit(jax.vmap(_kernel_second_order, in_axes=(0, 0, 0, 0)))
    return jax.jit(jax.vmap(_kernel_first_total, in_axes=(0, 0, 0)))


@lru_cache(maxsize=None)
def _get_bootstrap_point_kernel(calc_second_order: bool):
    """Cache bootstrap point-estimate kernels separately from no-bootstrap wrappers."""
    if calc_second_order:
        return jax.jit(_kernel_second_order)
    return jax.jit(_kernel_first_total)


def _drop_nonfinite(Y: Array, step: int) -> tuple[Array, int]:
    """Drop entire Saltelli groups that contain any non-finite (NaN/Inf) value.

    Each Saltelli group is a contiguous block of ``step`` rows in Y that
    corresponds to one base sample and its D (or 2D) cross-matrix evaluations.
    If any single element in the group is non-finite the whole group is removed,
    because a partial group would corrupt the A/B/AB/BA split.

    Args:
        Y: Model output array, shape (n_total, ...) where n_total = N * step.
            Trailing dimensions are typically (T, K) or absent.
        step: Number of rows per Saltelli group (D+2 or 2D+2).

    Returns:
        (Y_clean, n_dropped) — cleaned output array with shape
        (N_good * step, ...) and the number of groups that were removed.
    """
    n_total = Y.shape[0]
    base_n = n_total // step
    trailing = Y.shape[1:]

    # Reshape from flat rows into groups: (n_total, ...) -> (N, step, ...)
    grouped = Y[: base_n * step].reshape(base_n, step, *trailing)

    # Flatten each group to a single vector for finiteness check:
    # (N, step, ...) -> (N, step * prod(...)), then reduce along axis=1
    finite_mask = jnp.all(jnp.isfinite(grouped.reshape(base_n, -1)), axis=1)
    # finite_mask: (N,) boolean — True if all elements in that group are finite

    n_good = int(jnp.sum(finite_mask))
    n_dropped = base_n - n_good
    if n_dropped > 0:
        # Use NumPy for boolean indexing (not supported by JAX tracing)
        import numpy as np

        mask_np = np.asarray(finite_mask)
        grouped_clean = jnp.asarray(np.asarray(grouped)[mask_np])
        # grouped_clean: (N_good, step, ...) -> flatten back to row-major
        Y_clean = grouped_clean.reshape(n_good * step, *trailing)
        # Y_clean: (N_good * step, ...)
        return Y_clean, n_dropped
    return Y, 0


def _count_and_report_nans(
    S1: Array,
    ST: Array,
    S2: Array | None,
) -> dict[str, int]:
    """Count NaN values in computed Sobol index arrays and warn if any exist.

    Diagonal entries of S2 are always NaN by convention (undefined self-
    interaction), so they are excluded from the count via an off-diagonal mask.

    Args:
        S1: First-order indices, shape (D,) or (T, K, D) before squeezing.
        ST: Total-order indices, same shape as S1.
        S2: Second-order indices, shape (D, D) or (T, K, D, D), or None.

    Returns:
        Dictionary mapping index name to the count of unexpected NaN entries.
    """
    counts: dict[str, int] = {
        "S1": int(jnp.sum(jnp.isnan(S1))),
        "ST": int(jnp.sum(jnp.isnan(ST))),
    }
    if S2 is not None:
        # off_diag_mask: (D, D) — True everywhere except the diagonal
        off_diag_mask = ~jnp.eye(S2.shape[-1], dtype=bool)
        # Only count NaNs in off-diagonal positions (diagonal is NaN by design)
        counts["S2"] = int(jnp.sum(jnp.isnan(S2) & off_diag_mask))
    total = sum(counts.values())
    if total > 0:
        parts = ", ".join(f"{k}: {v}" for k, v in counts.items())
        print(f"gsax: {total} NaN indices in results ({parts})")
    return counts


def _separate_output_values(
    Y: Array, D: int, calc_second_order: bool
) -> tuple[Array, Array, Array, Array | None]:
    """De-interleave flat Saltelli output rows into the A, B, AB, BA matrices.

    The Saltelli layout repeats for each of the N base samples:
        [A_i, AB_i_0, ..., AB_i_{D-1}, (BA_i_0, ..., BA_i_{D-1}), B_i]
    where the BA block is present only when ``calc_second_order`` is True.
    This function uses stride-based slicing to extract each component.

    Args:
        Y: Flat model outputs, shape (N * step, ...) where ``...`` is
            typically (T, K) after _prepare_Y.
        D: Number of input parameters.
        calc_second_order: Whether second-order indices are being computed.

    Returns:
        A:  shape (N, ...) — outputs evaluated at the A base matrix.
        B:  shape (N, ...) — outputs evaluated at the B base matrix.
        AB: shape (N, D, ...) — outputs where column j of A is replaced by B.
        BA: shape (N, D, ...) or None — outputs where column j of B is
            replaced by A (only when calc_second_order is True).
    """
    step = 2 * D + 2 if calc_second_order else D + 2

    # Stride-slice: take every `step`-th row starting at offset 0 (A) or step-1 (B)
    A = Y[0::step]  # (N, ...)
    B = Y[(step - 1) :: step]  # (N, ...)

    # For each parameter j, AB_j starts at row offset j+1 with stride `step`.
    # Stack D slices along a new axis=1 to form the parameter dimension.
    AB = jnp.stack([Y[(j + 1) :: step] for j in range(D)], axis=1)  # (N, D, ...)

    BA = None
    if calc_second_order:
        # BA_j starts at row offset j+1+D (after the AB block)
        BA = jnp.stack([Y[(j + 1 + D) :: step] for j in range(D)], axis=1)  # (N, D, ...)

    return A, B, AB, BA


def _kernel_first_total(A: Array, AB: Array, B: Array) -> tuple[Array, Array]:
    """Jittable kernel that computes first-order and total-order Sobol indices.

    Uses vmap over the parameter dimension D so that all D indices are
    computed in a single vectorised call rather than a Python loop.

    Args:
        A:  (N,) model outputs evaluated at the A base matrix.
        AB: (N, D) model outputs from each cross-matrix AB_j.
        B:  (N,) model outputs evaluated at the B base matrix.

    Returns:
        S1: (D,) first-order Sobol indices.
        ST: (D,) total-order Sobol indices.
    """
    D = AB.shape[1]
    # vmap over parameter index j: each call receives A (N,), AB[:,j] (N,), B (N,)
    S1 = vmap(lambda j: first_order(A, AB[:, j], B))(jnp.arange(D))  # (D,)
    ST = vmap(lambda j: total_order(A, AB[:, j], B))(jnp.arange(D))  # (D,)
    return S1, ST


def _kernel_second_order(A: Array, AB: Array, BA: Array, B: Array) -> tuple[Array, Array, Array]:
    """Jittable kernel for first-, total-, and second-order Sobol indices.

    Second-order indices S2[j, k] measure the interaction effect between
    parameters j and k. Only the upper triangle is meaningful; the matrix
    is later normalised by ``_normalize_s2_matrix``.

    Uses nested vmap: the outer vmap iterates rows (j), the inner iterates
    columns (k), producing the full (D, D) interaction matrix in one pass.

    Args:
        A:  (N,) model outputs from the A base matrix.
        AB: (N, D) model outputs from each cross-matrix AB_j.
        BA: (N, D) model outputs from each cross-matrix BA_j.
        B:  (N,) model outputs from the B base matrix.

    Returns:
        S1: (D,) first-order Sobol indices.
        ST: (D,) total-order Sobol indices.
        S2: (D, D) second-order interaction indices (upper triangle valid).
    """
    D = AB.shape[1]
    # vmap over parameter index j: scalar estimators -> (D,) vectors
    S1 = vmap(lambda j: first_order(A, AB[:, j], B))(jnp.arange(D))  # (D,)
    ST = vmap(lambda j: total_order(A, AB[:, j], B))(jnp.arange(D))  # (D,)

    def s2_row(j):
        """Compute one row of the S2 matrix: S2[j, :] for all k."""

        def s2_elem(k):
            # Each call uses AB[:,j] (N,), AB[:,k] (N,), BA[:,j] (N,)
            return second_order(A, AB[:, j], AB[:, k], BA[:, j], B)

        return vmap(s2_elem)(jnp.arange(D))  # (D,)

    S2 = vmap(s2_row)(jnp.arange(D))  # (D, D)
    return S1, ST, S2


def _normalize_s2_matrix(S2: Array) -> Array:
    """Symmetrise the S2 matrix and set diagonal entries to NaN.

    The Saltelli estimator only populates the upper triangle of S2 with
    meaningful values (j < k). This function copies the upper triangle to
    the lower triangle so that S2[j, k] == S2[k, j], and fills the
    diagonal with NaN since self-interaction S2[j, j] is undefined.

    Args:
        S2: Second-order index matrix, shape (..., D, D). The leading
            dimensions can be (T, K) or (2, T, K) for confidence bounds.

    Returns:
        Symmetrised S2 with shape (..., D, D), diagonal entries set to NaN.
    """
    D = S2.shape[-1]
    # Extract upper triangle (k > j), zeroing everything else: (..., D, D)
    upper = jnp.triu(S2, k=1)
    # Transpose last two axes to mirror upper -> lower: (..., D, D)
    mirrored = upper + jnp.swapaxes(upper, -1, -2)
    # diag_mask: (D, D) — broadcasts over leading dims
    diag_mask = jnp.eye(D, dtype=bool)
    return jnp.where(diag_mask, jnp.nan, mirrored)


def _prepare_Y(
    Y: Array,
) -> tuple[Array, bool, bool]:
    """Promote Y to a canonical 3-D shape (n_total, T, K).

    All downstream computation assumes Y has exactly three dimensions.
    Singleton dimensions are inserted where needed and boolean flags record
    which ones were added so ``_squeeze_results`` can remove them later.

    Promotion rules:
        1-D (n_total,)     -> (n_total, 1, 1)  squeeze_time=True, squeeze_output=True
        2-D (n_total, K)   -> (n_total, 1, K)  squeeze_time=True
        3-D (n_total, T, K) -> unchanged

    Args:
        Y: Model output array with 1, 2, or 3 dimensions.

    Returns:
        (Y_3d, squeeze_time, squeeze_output) where Y_3d has shape
        (n_total, T, K) and the booleans indicate which singleton axes
        were inserted.
    """
    squeeze_time = False
    squeeze_output = False
    if Y.ndim == 1:
        Y = Y[:, None, None]  # (n_total,) -> (n_total, 1, 1)
        squeeze_time = True
        squeeze_output = True
    elif Y.ndim == 2:
        Y = Y[:, None, :]  # (n_total, K) -> (n_total, 1, K)
        squeeze_time = True
    return Y, squeeze_time, squeeze_output


def _expand_unique_outputs(sampling_result: SamplingResult, Y: Array) -> Array:
    """Rebuild expanded Saltelli outputs from unique user-evaluated outputs."""
    if Y.shape[0] != sampling_result.n_total:
        raise ValueError(
            f"Y.shape[0] must match sampling_result.n_total ({sampling_result.n_total}), "
            f"got {Y.shape[0]}"
        )
    expanded_to_unique = jnp.asarray(sampling_result.expanded_to_unique)
    return jnp.take(Y, expanded_to_unique, axis=0)


def _squeeze_results(
    S1: Array,
    ST: Array,
    S2: Array | None,
    squeeze_time: bool,
    squeeze_output: bool,
    S1_conf: Array | None = None,
    ST_conf: Array | None = None,
    S2_conf: Array | None = None,
) -> tuple[Array, Array, Array | None, Array | None, Array | None, Array | None]:
    """Remove singleton T and/or K dimensions that _prepare_Y inserted.

    This is the inverse of _prepare_Y's promotion: it strips the leading
    (T, K) axes from index arrays so the output shapes match what the
    user originally provided.

    Shape transitions for index arrays (S1, ST):
        squeeze both:  (T=1, K=1, D)       -> (D,)
        squeeze time:  (T=1, K, D)          -> (K, D)
        neither:       (T, K, D)            -> (T, K, D)

    For confidence arrays (S1_conf, etc.) the leading axis is the bound
    dimension (2 = [lo, hi]):
        squeeze both:  (2, T=1, K=1, D)    -> (2, D)
        squeeze time:  (2, T=1, K, D)      -> (2, K, D)

    For S2 / S2_conf the last two axes are (D, D) instead of (D,).

    Args:
        S1: First-order indices, shape (T, K, D).
        ST: Total-order indices, shape (T, K, D).
        S2: Second-order indices, shape (T, K, D, D) or None.
        squeeze_time: Whether the T axis is a singleton to remove.
        squeeze_output: Whether the K axis is a singleton to remove.
        S1_conf: Bootstrap CI bounds for S1, shape (2, T, K, D) or None.
        ST_conf: Bootstrap CI bounds for ST, shape (2, T, K, D) or None.
        S2_conf: Bootstrap CI bounds for S2, shape (2, T, K, D, D) or None.

    Returns:
        Tuple of (S1, ST, S2, S1_conf, ST_conf, S2_conf) with singleton
        dimensions removed as indicated by the squeeze flags.
    """
    if squeeze_time and squeeze_output:
        # (T=1, K=1, D) -> (D,) ; (T=1, K=1, D, D) -> (D, D)
        S1 = S1[0, 0]
        ST = ST[0, 0]
        if S2 is not None:
            S2 = S2[0, 0]
        if S1_conf is not None:
            # (2, T=1, K=1, D) -> (2, D)
            S1_conf = S1_conf[:, 0, 0]
            ST_conf = ST_conf[:, 0, 0]
            if S2_conf is not None:
                # (2, T=1, K=1, D, D) -> (2, D, D)
                S2_conf = S2_conf[:, 0, 0]
    elif squeeze_time:
        # (T=1, K, D) -> (K, D) ; (T=1, K, D, D) -> (K, D, D)
        S1 = S1[0]
        ST = ST[0]
        if S2 is not None:
            S2 = S2[0]
        if S1_conf is not None:
            # (2, T=1, K, D) -> (2, K, D)
            S1_conf = S1_conf[:, 0]
            ST_conf = ST_conf[:, 0]
            if S2_conf is not None:
                # (2, T=1, K, D, D) -> (2, K, D, D)
                S2_conf = S2_conf[:, 0]
    return S1, ST, S2, S1_conf, ST_conf, S2_conf


def _analyze_no_bootstrap(
    sampling_result: SamplingResult, Y: Array, *, chunk_size: int
) -> SAResult:
    """Fast path: compute Sobol indices by vmapping over all (T, K) combos.

    The T*K output combinations are flattened into a single batch dimension
    and processed via ``jax.vmap`` in chunks of ``chunk_size`` to control
    memory usage. No bootstrap resampling is performed.

    Args:
        sampling_result: Metadata from the sampling step (N, D, etc.).
        Y: Model outputs, shape (N * step,) or (N * step, K) or
            (N * step, T, K). Promoted to 3-D internally.
        chunk_size: Maximum number of (T, K) combos per vmap batch.

    Returns:
        SAResult containing S1 (D,), ST (D,), and optionally S2 (D, D),
        with T and K dimensions restored or squeezed as appropriate.
    """
    Y, squeeze_time, squeeze_output = _prepare_Y(Y)
    # Y: (N * step, T, K) after promotion
    D = sampling_result.n_params
    calc_second_order = sampling_result.calc_second_order

    _, T, K = Y.shape  # Y is now guaranteed 3-D
    A, B, AB, BA = _separate_output_values(Y, D, calc_second_order)
    base_n = A.shape[0]
    # A: (N, T, K), B: (N, T, K), AB: (N, D, T, K), BA: (N, D, T, K) or None

    # Reshape so the batch dim is T*K (all output combos) and the sample dim
    # is N, matching the kernel signatures which expect (N,) and (N, D).
    # transpose: (N, T, K) -> (T, K, N) then flatten -> (T*K, N)
    A_flat = A.transpose(1, 2, 0).reshape(T * K, base_n)  # (T*K, N)
    B_flat = B.transpose(1, 2, 0).reshape(T * K, base_n)  # (T*K, N)
    # transpose: (N, D, T, K) -> (T, K, N, D) then flatten -> (T*K, N, D)
    AB_flat = AB.transpose(2, 3, 0, 1).reshape(T * K, base_n, D)  # (T*K, N, D)

    total = T * K
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    cs = min(chunk_size, total)

    if calc_second_order:
        assert BA is not None
        # Same reshape as AB: (N, D, T, K) -> (T*K, N, D)
        BA_flat = BA.transpose(2, 3, 0, 1).reshape(T * K, base_n, D)  # (T*K, N, D)

        batched = _get_no_bootstrap_batched_kernel(True)
        s1_parts, st_parts, s2_parts = [], [], []
        for start in range(0, total, cs):
            end = min(start + cs, total)
            # Each chunk slice: A_flat[start:end] is (chunk, N), etc.
            s1, st, s2 = batched(
                A_flat[start:end],
                AB_flat[start:end],
                BA_flat[start:end],
                B_flat[start:end],
            )
            # s1, st: (chunk, D); s2: (chunk, D, D)
            s1_parts.append(s1)
            st_parts.append(st)
            s2_parts.append(s2)

        # Concatenate chunks and reshape back to (T, K, ...)
        S1_out = jnp.concatenate(s1_parts).reshape(T, K, D)  # (T, K, D)
        ST_out = jnp.concatenate(st_parts).reshape(T, K, D)  # (T, K, D)
        S2_out = _normalize_s2_matrix(
            jnp.concatenate(s2_parts).reshape(T, K, D, D)  # (T, K, D, D)
        )
    else:
        batched = _get_no_bootstrap_batched_kernel(False)
        s1_parts, st_parts = [], []
        for start in range(0, total, cs):
            end = min(start + cs, total)
            s1, st = batched(
                A_flat[start:end],
                AB_flat[start:end],
                B_flat[start:end],
            )
            # s1, st: (chunk, D)
            s1_parts.append(s1)
            st_parts.append(st)

        S1_out = jnp.concatenate(s1_parts).reshape(T, K, D)  # (T, K, D)
        ST_out = jnp.concatenate(st_parts).reshape(T, K, D)  # (T, K, D)
        S2_out = None

    # Remove singleton T and/or K dimensions to match original Y shape
    S1_out, ST_out, S2_out, _, _, _ = _squeeze_results(
        S1_out, ST_out, S2_out, squeeze_time, squeeze_output
    )
    nan_counts = _count_and_report_nans(S1_out, ST_out, S2_out)
    return SAResult(
        S1=S1_out,
        ST=ST_out,
        S2=S2_out,
        problem=sampling_result.problem,
        nan_counts=nan_counts,
    )


def _analyze_bootstrap(
    sampling_result: SamplingResult,
    Y: Array,
    *,
    num_resamples: int,
    conf_level: float,
    key: Array,
    chunk_size: int,
) -> SAResult:
    """Bootstrap path: loop over (T, K) combos, vmap over R resamples.

    Unlike the no-bootstrap path which vmaps over T*K, here we loop over
    each (t, k) output combination individually and vmap over the R bootstrap
    resamples (in chunks of ``chunk_size``). This trades some speed for
    lower peak memory since each iteration only holds R copies of the
    1-D sample vectors.

    Args:
        sampling_result: Metadata from the sampling step.
        Y: Model outputs, promoted to (N * step, T, K) internally.
        num_resamples: R, the number of bootstrap resamples.
        conf_level: Confidence level (e.g. 0.95 for 95% CI).
        key: JAX PRNG key for generating bootstrap indices.
        chunk_size: Number of resamples per vmap batch in bootstrap kernels.

    Returns:
        SAResult with point estimates, confidence intervals, and NaN counts.
    """
    from gsax._bootstrap import _bootstrap_first_total, _bootstrap_second_order

    Y, squeeze_time, squeeze_output = _prepare_Y(Y)
    # Y: (N * step, T, K)
    D = sampling_result.n_params
    calc_second_order = sampling_result.calc_second_order

    _, T, K = Y.shape
    A, B, AB, BA = _separate_output_values(Y, D, calc_second_order)
    base_n = A.shape[0]
    # A: (N, T, K), B: (N, T, K), AB: (N, D, T, K), BA: (N, D, T, K) or None

    # Generate bootstrap resample indices once, shared across all (t, k) combos.
    # Each row is a set of N indices drawn with replacement from [0, N).
    indices = jax.random.randint(key, shape=(num_resamples, base_n), minval=0, maxval=base_n)
    # indices: (R, N)

    # Compute percentile boundaries for the confidence interval
    alpha = (1.0 - conf_level) / 2.0
    percentiles = jnp.array([alpha * 100, (1.0 - alpha) * 100])  # (2,)

    # Cache point-estimate kernels separately from the no-bootstrap batched wrappers.
    jit_ft = _get_bootstrap_point_kernel(False)
    jit_so = _get_bootstrap_point_kernel(True) if calc_second_order else None

    # Accumulators — each list collects T*K items (one per output combo)
    S1_list, ST_list = [], []
    S1_lo_list, S1_hi_list = [], []
    ST_lo_list, ST_hi_list = [], []
    S2_list, S2_lo_list, S2_hi_list = [], [], []

    for t in range(T):
        for k in range(K):
            # Extract 1-D sample vectors for this specific (t, k) output
            a = A[:, t, k]  # (N,)
            b = B[:, t, k]  # (N,)
            ab = AB[:, :, t, k]  # (N, D)

            if calc_second_order:
                assert BA is not None
                ba = BA[:, :, t, k]  # (N, D)

                # Point estimates: s1 (D,), st (D,), s2 (D, D)
                s1, st, s2 = jit_so(a, ab, ba, b)
                S2_list.append(s2)

                # Bootstrap: s1_boot (R, D), st_boot (R, D), s2_boot (R, D, D)
                s1_boot, st_boot, s2_boot = _bootstrap_second_order(
                    indices, a, ab, ba, b, chunk_size
                )
                # Percentiles along resample axis: (2, D, D)
                s2_ci = jnp.percentile(s2_boot, percentiles, axis=0)  # (2, D, D)
                S2_lo_list.append(s2_ci[0])  # (D, D)
                S2_hi_list.append(s2_ci[1])  # (D, D)
            else:
                # Point estimates: s1 (D,), st (D,)
                s1, st = jit_ft(a, ab, b)
                # Bootstrap: s1_boot (R, D), st_boot (R, D)
                s1_boot, st_boot = _bootstrap_first_total(indices, a, ab, b, chunk_size)

            S1_list.append(s1)  # (D,)
            ST_list.append(st)  # (D,)

            # Percentiles along resample axis: (2, D)
            s1_ci = jnp.percentile(s1_boot, percentiles, axis=0)  # (2, D)
            st_ci = jnp.percentile(st_boot, percentiles, axis=0)  # (2, D)
            S1_lo_list.append(s1_ci[0])  # (D,)
            S1_hi_list.append(s1_ci[1])  # (D,)
            ST_lo_list.append(st_ci[0])  # (D,)
            ST_hi_list.append(st_ci[1])  # (D,)

    # Stack T*K items and reshape back to (T, K, D) grid
    S1_out = jnp.stack(S1_list).reshape(T, K, D)  # (T*K, D) -> (T, K, D)
    ST_out = jnp.stack(ST_list).reshape(T, K, D)  # (T*K, D) -> (T, K, D)

    # Stack lo/hi bounds into a (2, T, K, D) confidence array
    S1_conf = jnp.stack(
        [
            jnp.stack(S1_lo_list).reshape(T, K, D),
            jnp.stack(S1_hi_list).reshape(T, K, D),
        ]
    )  # (2, T, K, D)
    ST_conf = jnp.stack(
        [
            jnp.stack(ST_lo_list).reshape(T, K, D),
            jnp.stack(ST_hi_list).reshape(T, K, D),
        ]
    )  # (2, T, K, D)

    if calc_second_order:
        # (T*K, D, D) -> (T, K, D, D), then normalise
        S2_out = _normalize_s2_matrix(jnp.stack(S2_list).reshape(T, K, D, D))
        # Stack lo/hi: each list has T*K items of shape (D, D)
        # -> (2, T, K, D, D), then normalise
        S2_conf = _normalize_s2_matrix(
            jnp.stack(
                [
                    jnp.stack(S2_lo_list).reshape(T, K, D, D),
                    jnp.stack(S2_hi_list).reshape(T, K, D, D),
                ]
            )  # (2, T, K, D, D)
        )
    else:
        S2_out = None
        S2_conf = None

    # Remove singleton T and/or K dimensions to match original Y shape
    S1_out, ST_out, S2_out, S1_conf, ST_conf, S2_conf = _squeeze_results(
        S1_out,
        ST_out,
        S2_out,
        squeeze_time,
        squeeze_output,
        S1_conf,
        ST_conf,
        S2_conf,
    )
    nan_counts = _count_and_report_nans(S1_out, ST_out, S2_out)
    return SAResult(
        S1=S1_out,
        ST=ST_out,
        S2=S2_out,
        problem=sampling_result.problem,
        S1_conf=S1_conf,
        ST_conf=ST_conf,
        S2_conf=S2_conf,
        nan_counts=nan_counts,
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
    """Compute Sobol sensitivity indices from model outputs using JAX.

    This is the main entry point. It accepts model outputs Y evaluated at the
    unique rows returned by ``gsax.sample()``, reconstructs the expanded
    Saltelli ordering internally, cleans non-finite values, and dispatches to
    either the fast no-bootstrap path or the bootstrap path depending on
    ``num_resamples``.

    Args:
        sampling_result: Result from ``gsax.sample()`` containing the unique
            sample matrix plus expansion metadata.
        Y: Model outputs evaluated at each unique row of
            ``sampling_result.samples``. Accepted shapes:
                (n_total,)       — scalar output, single time step
                (n_total, K)     — K outputs, single time step
                (n_total, T, K)  — K outputs over T time steps
            where ``n_total`` is the unique row count.
        num_resamples: R, the number of bootstrap resamples for confidence
            intervals. Set to 0 (default) to skip bootstrap.
        conf_level: Confidence level for bootstrap CIs (default 0.95).
        key: JAX PRNG key for bootstrap randomness. Required when
            ``num_resamples > 0``.
        chunk_size: Controls vmap batch size. In the no-bootstrap path this
            is the number of (T, K) combos per batch; in the bootstrap path
            it is the number of resamples per batch. Defaults to 2048.

    Returns:
        SAResult containing:
            S1 — first-order indices, shape (D,) / (K, D) / (T, K, D)
            ST — total-order indices, same shape as S1
            S2 — second-order indices (D, D) / ... or None
            S1_conf, ST_conf, S2_conf — (2, ...) CI bounds or None
    """
    Y = jnp.asarray(Y)
    Y = _expand_unique_outputs(sampling_result, Y)

    D = sampling_result.n_params
    step = _saltelli_step(D, sampling_result.calc_second_order)

    # Clean non-finite outputs by dropping entire Saltelli groups
    Y, n_dropped = _drop_nonfinite(Y, step)
    if n_dropped > 0:
        total_groups = Y.shape[0] // step + n_dropped
        print(
            f"gsax: dropped {n_dropped} of {total_groups} sample groups containing non-finite values"
        )
        if Y.shape[0] == 0:
            raise ValueError("All samples contain non-finite values")

    if num_resamples > 0:
        if key is None:
            raise ValueError("key is required when num_resamples > 0")
        return _analyze_bootstrap(
            sampling_result,
            Y,
            num_resamples=num_resamples,
            conf_level=conf_level,
            key=key,
            chunk_size=chunk_size,
        )

    return _analyze_no_bootstrap(sampling_result, Y, chunk_size=chunk_size)
