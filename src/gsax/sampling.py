import math
from dataclasses import dataclass

import numpy as np
from scipy.stats.qmc import Sobol

from gsax.problem import Problem


@dataclass(frozen=True)
class SamplingResult:
    samples: np.ndarray  # (n_total, D) scaled to bounds
    base_n: int
    n_params: int
    calc_second_order: bool
    problem: Problem

    @property
    def n_total(self) -> int:
        return self.samples.shape[0]


def _next_power_of_2(n: int) -> int:
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
    D = problem.num_vars
    step = 2 * D + 2 if calc_second_order else D + 2
    base_n = _next_power_of_2(math.ceil(n_samples / step))

    # Generate 2D-dimensional Sobol sequence of length base_n
    sampler = Sobol(d=2 * D, scramble=scramble, seed=seed)
    base = sampler.random(base_n)  # (base_n, 2D) in [0,1]

    # Split into A and B halves
    A = base[:, :D]  # (base_n, D)
    B = base[:, D:]  # (base_n, D)

    # Build sample matrix: for each base sample, emit row blocks
    rows = []
    for i in range(base_n):
        rows.append(A[i])  # A_i

        # AB matrices: start from A, replace column j with B's column j
        for j in range(D):
            AB_j = A[i].copy()
            AB_j[j] = B[i, j]
            rows.append(AB_j)

        if calc_second_order:
            # BA matrices: start from B, replace column j with A's column j
            for j in range(D):
                BA_j = B[i].copy()
                BA_j[j] = A[i, j]
                rows.append(BA_j)

        rows.append(B[i])  # B_i

    samples_unit = np.array(rows)  # (n_total, D) in [0,1]

    # Scale to bounds
    bounds = np.array(problem.bounds)  # (D, 2)
    low, high = bounds[:, 0], bounds[:, 1]
    samples = samples_unit * (high - low) + low

    return SamplingResult(
        samples=samples,
        base_n=base_n,
        n_params=D,
        calc_second_order=calc_second_order,
        problem=problem,
    )
