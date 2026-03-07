"""Sobol/Saltelli sampling with unique user-facing rows.

The public contract of this module is intentionally split in two layers:

1. ``sample()`` returns only the unique rows that a user should evaluate.
2. ``SamplingResult`` also carries enough metadata to reconstruct the full
   expanded Saltelli layout later inside :func:`gsax.analyze`.

This avoids wasted model evaluations in low-dimensional cases where the
expanded Saltelli design contains exact duplicate rows.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from gsax.problem import Problem


@dataclass(frozen=True)
class SamplingResult:
    """Unique Sobol samples plus metadata for Saltelli reconstruction.

    Attributes:
        samples: Unique rows to evaluate with the user's model. Shape
            ``(n_total, D)`` after scaling to the problem bounds.
        sample_ids: Stable integer identifiers aligned 1:1 with ``samples``.
            Useful for joining model outputs back onto the sampling table.
        expanded_n_total: Row count of the full expanded Saltelli layout before
            deduplication. This is the number of rows analyzed internally.
        expanded_to_unique: Integer index map of shape ``(expanded_n_total,)``.
            For each expanded Saltelli row, gives the corresponding row index in
            ``samples``.
        base_n: Number of base Sobol points used to construct the Saltelli
            design. Always a power of 2.
        n_params: Number of problem dimensions ``D``.
        calc_second_order: Whether the expanded design includes BA blocks for
            second-order Sobol indices.
        problem: Problem definition used to scale the samples.
    """

    samples: np.ndarray  # shape (n_unique, D), scaled to bounds
    sample_ids: np.ndarray
    expanded_n_total: int
    expanded_to_unique: np.ndarray
    base_n: int
    n_params: int
    calc_second_order: bool
    problem: Problem

    @property
    def n_total(self) -> int:
        """Number of unique rows in ``samples``."""
        return self.samples.shape[0]

    @property
    def samples_df(self) -> pd.DataFrame:
        """Return the unique sample matrix as a DataFrame with ``SampleID``.

        The DataFrame is intended as a convenience view for export, inspection,
        or joining with model outputs. The underlying canonical representation
        remains the NumPy array in ``samples``.
        """
        data = {"SampleID": self.sample_ids}
        for idx, name in enumerate(self.problem.names):
            data[name] = self.samples[:, idx]
        return pd.DataFrame(data, copy=False)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= *n*."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _saltelli_step(n_params: int, calc_second_order: bool) -> int:
    """Return the number of expanded Saltelli rows per base Sobol point."""
    return 2 * n_params + 2 if calc_second_order else n_params + 2


def _build_expanded_samples(
    problem: Problem,
    base_n: int,
    *,
    calc_second_order: bool,
    scramble: bool,
    seed: int | np.random.Generator | None,
) -> np.ndarray:
    """Generate the full expanded Saltelli matrix for a fixed ``base_n``.

    The returned matrix still includes exact duplicate rows when the Saltelli
    construction collapses in low dimensions. Deduplication happens later.
    """
    D = problem.num_vars
    sampler = Sobol(d=2 * D, scramble=scramble, seed=seed)
    base = sampler.random(base_n)

    # Split the 2D-dimensional Sobol draw into the standard Saltelli base
    # matrices A and B, each with shape (base_n, D).
    A = base[:, :D]
    B = base[:, D:]
    rows = []
    for i in range(base_n):
        # Emit the Saltelli group for base point i:
        # [A_i, AB_i_0, ..., AB_i_{D-1}, (BA_i_0, ..., BA_i_{D-1}), B_i]
        rows.append(A[i])
        for j in range(D):
            AB_j = A[i].copy()
            AB_j[j] = B[i, j]
            rows.append(AB_j)
        if calc_second_order:
            for j in range(D):
                BA_j = B[i].copy()
                BA_j[j] = A[i, j]
                rows.append(BA_j)
        rows.append(B[i])

    samples_unit = np.array(rows)
    bounds = np.array(problem.bounds)
    low, high = bounds[:, 0], bounds[:, 1]
    # Scale from the unit hypercube into the user-specified parameter bounds.
    return samples_unit * (high - low) + low


def _stable_unique_rows(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate rows while preserving first-occurrence order.

    Returns:
        ``(unique_samples, expanded_to_unique)`` where ``expanded_to_unique``
        maps each original row position in ``samples`` back to the retained
        unique row index.
    """
    samples = np.ascontiguousarray(samples)
    unique_rows: list[np.ndarray] = []
    expanded_to_unique = np.empty(samples.shape[0], dtype=np.int64)
    seen: dict[bytes, int] = {}

    for idx, row in enumerate(samples):
        # ``row.tobytes()`` gives a stable exact-match key for the already
        # scaled floating-point row. Exact deduplication is what we want here:
        # if two rows are bitwise equal, evaluating the model twice is wasteful.
        key = row.tobytes()
        unique_idx = seen.get(key)
        if unique_idx is None:
            unique_idx = len(unique_rows)
            seen[key] = unique_idx
            unique_rows.append(row.copy())
        expanded_to_unique[idx] = unique_idx

    if unique_rows:
        unique_samples = np.vstack(unique_rows)
    else:
        unique_samples = np.empty((0, samples.shape[1]), dtype=samples.dtype)
    return unique_samples, expanded_to_unique


def _print_sampling_summary(
    *,
    n_params: int,
    target_n: int,
    unique_n: int,
    expanded_n_total: int,
    base_n: int,
    calc_second_order: bool,
    scramble: bool,
) -> None:
    """Print a compact summary of the generated unique Sobol design."""
    duplicates_removed = expanded_n_total - unique_n
    duplicate_fraction = duplicates_removed / expanded_n_total if expanded_n_total else 0.0
    order_label = "second-order" if calc_second_order else "first/total-order"
    print(
        "gsax.sample: "
        f"D={n_params}, mode={order_label}, base_n={base_n}, "
        f"requested_unique>={target_n}, returned_unique={unique_n}, "
        f"expanded_rows={expanded_n_total}, duplicates_removed={duplicates_removed} "
        f"({duplicate_fraction:.1%}), scramble={scramble}"
    )


def sample(
    problem: Problem,
    n_samples: int,
    *,
    calc_second_order: bool = True,
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
    verbose: bool = True,
) -> SamplingResult:
    """Generate unique Sobol/Saltelli samples for model evaluation.

    The function first builds the standard expanded Saltelli design for a
    candidate ``base_n``. It then removes exact duplicate rows while
    preserving first-occurrence order. If the resulting unique matrix is still
    smaller than the requested evaluation budget, ``base_n`` is doubled and
    the process repeats until enough unique rows are available.

    Args:
        problem: Problem definition with parameter names and bounds.
        n_samples: Minimum desired number of unique model evaluations.
        calc_second_order: If ``True``, include BA cross-matrices so that
            second-order Sobol indices can be computed.  This increases
            the expanded Saltelli step from ``D + 2`` to ``2*D + 2``.
        scramble: Whether to apply Owen scrambling to the Sobol sequence.
        seed: Random seed or generator for reproducibility.
        verbose: If ``True`` (default), print a short summary describing the
            requested unique count, returned unique count, expanded Saltelli
            size, and how many duplicate rows were removed.

    Returns:
        SamplingResult with a unique sample matrix plus expansion metadata for
        later Sobol analysis.
    """
    D = problem.num_vars
    step = _saltelli_step(D, calc_second_order)
    target_n = max(1, n_samples)
    # Start from the smallest power-of-two Sobol base size that could plausibly
    # satisfy the requested unique budget if there were no duplicates.
    base_n = _next_power_of_2(math.ceil(target_n / step))

    while True:
        expanded_samples = _build_expanded_samples(
            problem,
            base_n,
            calc_second_order=calc_second_order,
            scramble=scramble,
            seed=seed,
        )
        unique_samples, expanded_to_unique = _stable_unique_rows(expanded_samples)
        # Low-dimensional Saltelli designs can contain exact duplicate rows.
        # Keep increasing base_n until the user-facing matrix is large enough.
        if unique_samples.shape[0] >= target_n:
            break
        base_n *= 2

    sample_ids = np.arange(unique_samples.shape[0], dtype=np.int64)
    if verbose:
        _print_sampling_summary(
            n_params=D,
            target_n=target_n,
            unique_n=unique_samples.shape[0],
            expanded_n_total=expanded_samples.shape[0],
            base_n=base_n,
            calc_second_order=calc_second_order,
            scramble=scramble,
        )

    return SamplingResult(
        samples=unique_samples,
        sample_ids=sample_ids,
        expanded_n_total=expanded_samples.shape[0],
        expanded_to_unique=expanded_to_unique,
        base_n=base_n,
        n_params=D,
        calc_second_order=calc_second_order,
        problem=problem,
    )
