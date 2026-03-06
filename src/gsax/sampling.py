"""Saltelli's sampling scheme using Sobol sequences for global sensitivity analysis.

Saltelli's scheme constructs a sample matrix that enables efficient estimation
of first-order, second-order, and total-order Sobol sensitivity indices from a
single set of model evaluations.  The method works as follows:

1. Draw two independent quasi-random base matrices **A** and **B**, each of
   shape ``(N, D)``, from a ``2D``-dimensional Sobol sequence (``N = base_n``,
   ``D = num_params``).

2. For every base sample index *i* (``0 .. N-1``), emit a *group* of rows:

       [ A_i,  AB_i_0, AB_i_1, .., AB_i_{D-1},
         (BA_i_0, BA_i_1, .., BA_i_{D-1}),  B_i ]

   - **AB_i_j** -- copy of ``A_i`` with column *j* replaced by ``B_i[j]``.
     Used for first-order and total-order index estimation.
   - **BA_i_j** -- copy of ``B_i`` with column *j* replaced by ``A_i[j]``.
     Only present when ``calc_second_order=True``; used for second-order
     index estimation.

3. Each group therefore has ``step`` rows, where
   ``step = 2*D + 2`` (second order) or ``D + 2`` (first order only).
   The total number of model evaluations is ``n_total = N * step``.

4. Finally the unit-hypercube samples in ``[0, 1]^D`` are linearly scaled to
   the parameter bounds specified in the :class:`Problem`.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats.qmc import Sobol

from gsax.problem import Problem


@dataclass(frozen=True)
class SamplingResult:
    """Result container for Saltelli sampling.

    Attributes:
        samples: The full sample matrix, shape ``(n_total, D)``, with values
            scaled to the parameter bounds.  Rows are interleaved in groups
            of ``step`` (see module docstring for layout).
        base_n: Number of base Sobol points (``N``).  Always a power of 2,
            as required by ``scipy.stats.qmc.Sobol``.
        n_params: Number of input parameters (``D``).
        calc_second_order: Whether BA cross-matrices were included.
        problem: The :class:`Problem` that defined parameter names and bounds.
    """

    samples: np.ndarray   # shape (n_total, D), scaled to bounds
    base_n: int           # N -- power-of-2 Sobol sequence length
    n_params: int         # D -- number of input parameters
    calc_second_order: bool
    problem: Problem

    @property
    def n_total(self) -> int:
        """Total rows: ``base_n * step`` where step = 2D+2 or D+2."""
        return self.samples.shape[0]


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= *n*."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def sample(
    problem: Problem,
    n_samples: int,
    *,
    calc_second_order: bool = True,
    scramble: bool = True,
    seed: int | np.random.Generator | None = None,
) -> SamplingResult:
    """Generate a Saltelli sample matrix for Sobol sensitivity analysis.

    The function draws a ``(base_n, 2D)`` Sobol sequence, splits it into two
    independent base matrices **A** and **B** of shape ``(N, D)``, then
    interleaves rows according to Saltelli's radial scheme (see module
    docstring for the full layout).

    Args:
        problem: Problem definition with parameter names and bounds.
        n_samples: Minimum desired number of model evaluations.  The actual
            count (``n_total``) will be >= this because ``base_n`` is rounded
            up to the next power of 2 (a requirement of Sobol sequences).
        calc_second_order: If ``True``, include BA cross-matrices so that
            second-order Sobol indices can be computed.  This increases
            ``step`` from ``D + 2`` to ``2*D + 2``.
        scramble: Whether to apply Owen scrambling to the Sobol sequence.
        seed: Random seed or generator for reproducibility.

    Returns:
        SamplingResult with the scaled sample matrix and metadata.
    """
    D = problem.num_vars  # number of input parameters

    # step = rows per base-sample group:
    #   2*D + 2  -->  A_i + D AB's + D BA's + B_i   (second order)
    #   D + 2    -->  A_i + D AB's + B_i             (first order only)
    step = 2 * D + 2 if calc_second_order else D + 2

    # base_n (= N) must be a power of 2 for Sobol sequences.
    # n_total = base_n * step >= n_samples.
    base_n = _next_power_of_2(math.ceil(n_samples / step))

    # Draw a single 2D-dimensional Sobol sequence of length N.
    # The first D columns become matrix A; the last D become matrix B.
    sampler = Sobol(d=2 * D, scramble=scramble, seed=seed)
    base = sampler.random(base_n)  # shape (N, 2D), values in [0, 1]

    A = base[:, :D]  # shape (N, D) -- first base matrix
    B = base[:, D:]  # shape (N, D) -- second base matrix

    # Build the interleaved sample matrix.
    # For each base index i the group layout is:
    #   row 0          : A_i                          shape (D,)
    #   rows 1..D      : AB_i_j  (j = 0 .. D-1)      shape (D,) each
    #   rows D+1..2D   : BA_i_j  (j = 0 .. D-1)      [only if second order]
    #   last row       : B_i                          shape (D,)
    rows = []
    for i in range(base_n):
        rows.append(A[i])  # A_i -- shape (D,)

        # AB cross-matrices: copy A_i, swap in column j from B_i.
        for j in range(D):
            AB_j = A[i].copy()       # shape (D,)
            AB_j[j] = B[i, j]
            rows.append(AB_j)

        if calc_second_order:
            # BA cross-matrices: copy B_i, swap in column j from A_i.
            for j in range(D):
                BA_j = B[i].copy()   # shape (D,)
                BA_j[j] = A[i, j]
                rows.append(BA_j)

        rows.append(B[i])  # B_i -- shape (D,)

    # Stack all groups into a single matrix.
    samples_unit = np.array(rows)  # shape (n_total, D), values in [0, 1]

    # Scale from the unit hypercube [0, 1]^D to the user-specified parameter
    # bounds:  x_scaled = x_unit * (high - low) + low
    bounds = np.array(problem.bounds)  # shape (D, 2)
    low, high = bounds[:, 0], bounds[:, 1]  # each shape (D,)
    samples = samples_unit * (high - low) + low  # shape (n_total, D)

    return SamplingResult(
        samples=samples,
        base_n=base_n,
        n_params=D,
        calc_second_order=calc_second_order,
        problem=problem,
    )
