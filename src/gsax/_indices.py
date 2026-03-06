"""Sobol sensitivity index estimators (first-order, total-order, second-order).

Implements the Saltelli (2010) estimators for variance-based global
sensitivity analysis using a Sobol quasi-random sampling design.

Notation
--------
- N : number of base samples drawn from the input parameter space.
- A, B : two independent (N, D) input sample matrices, where D is the
  number of parameters.  The arrays passed to these functions are the
  corresponding *model outputs*, each of shape (N,).
- AB_j : model output when column j of A is replaced by column j of B
  (all other columns remain from A).  Shape (N,).
- BA_j : model output when column j of B is replaced by column j of A
  (all other columns remain from B).  Shape (N,).

Variance estimation
-------------------
All estimators normalise by a *pooled* output variance computed over the
concatenation of A and B, i.e. ``var(concat(A, B))`` with shape (2N,).
Pooling both base-sample vectors gives a more robust variance estimate
than using A or B alone, because it doubles the effective sample size
while remaining unbiased (A and B are identically distributed).
"""

import jax.numpy as jnp
from jax import Array


def first_order(A: Array, AB_j: Array, B: Array) -> Array:
    """Estimate the first-order (main-effect) Sobol index for parameter j.

    The first-order index S1_j measures the fraction of output variance
    explained by parameter j alone, excluding any interactions with other
    parameters.

    Uses the Saltelli (2010) estimator::

        S1_j = E[B * (AB_j - A)] / Var(Y)

    where E[.] denotes the sample mean over N points and Var(Y) is the
    pooled output variance.

    Derivation: because B and AB_j share every input except j (which in
    AB_j comes from B), the product ``B * (AB_j - A)`` isolates the
    contribution of varying parameter j while averaging out all others.

    Args:
        A: (N,) model outputs evaluated on base sample matrix A.
        AB_j: (N,) model outputs from the cross-matrix where column j
            of A is replaced by column j of B.
        B: (N,) model outputs evaluated on base sample matrix B.

    Returns:
        Scalar Array with the first-order index S1_j.  Returns NaN when
        the output variance is zero (constant model).
    """
    # Pooled output vector of shape (2N,) for a robust variance estimate.
    y = jnp.concatenate([A, B])
    # Scalar total output variance (denominator for all Sobol indices).
    var = jnp.var(y)
    # Saltelli estimator numerator: element-wise product then mean, shape ().
    numerator = jnp.mean(B * (AB_j - A))
    # Guard against division by zero when the model output is constant.
    return jnp.where(var == 0, jnp.nan, numerator / var)


def total_order(A: Array, AB_j: Array, B: Array) -> Array:
    """Estimate the total-order Sobol index for parameter j.

    The total-order index ST_j measures the fraction of output variance
    caused by parameter j *including* all its interactions with other
    parameters (of any order).

    Uses the Jansen (1999) estimator::

        ST_j = (1/2) * E[(A - AB_j)^2] / Var(Y)

    A and AB_j differ only in parameter j.  The squared difference
    therefore captures the total effect of changing j while every other
    parameter is held fixed, averaged over the input space.

    Args:
        A: (N,) model outputs evaluated on base sample matrix A.
        AB_j: (N,) model outputs from the cross-matrix where column j
            of A is replaced by column j of B.
        B: (N,) model outputs evaluated on base sample matrix B.

    Returns:
        Scalar Array with the total-order index ST_j.  Returns NaN when
        the output variance is zero (constant model).
    """
    # Pooled output vector of shape (2N,) for variance estimation.
    y = jnp.concatenate([A, B])
    # Scalar total output variance.
    var = jnp.var(y)
    # Mean squared difference isolates total effect of parameter j, shape ().
    numerator = 0.5 * jnp.mean((A - AB_j) ** 2)
    # Guard against division by zero when the model output is constant.
    return jnp.where(var == 0, jnp.nan, numerator / var)


def second_order(A: Array, AB_j: Array, AB_k: Array, BA_j: Array, B: Array) -> Array:
    """Estimate the second-order Sobol interaction index between parameters j and k.

    The second-order index S2_jk measures the fraction of output variance
    due to the *joint* interaction between parameters j and k, after
    subtracting their individual main effects.

    Uses the Saltelli (2002) estimator::

        V_jk  = E[BA_j * AB_k - A * B] / Var(Y)
        S2_jk = V_jk - S1_j - S1_k

    Here V_jk captures the combined first- and second-order variance
    contribution of (j, k).  Subtracting S1_j and S1_k isolates the
    pure interaction term.

    In ``BA_j * AB_k``: BA_j has parameter j from A (rest from B), and
    AB_k has parameter k from B (rest from A).  Their product correlates
    the effects of j and k across the two sample sets.

    Args:
        A: (N,) model outputs evaluated on base sample matrix A.
        AB_j: (N,) model outputs from the cross-matrix where column j
            of A is replaced by column j of B.
        AB_k: (N,) model outputs from the cross-matrix where column k
            of A is replaced by column k of B.
        BA_j: (N,) model outputs from the cross-matrix where column j
            of B is replaced by column j of A.
        B: (N,) model outputs evaluated on base sample matrix B.

    Returns:
        Scalar Array with the second-order interaction index S2_jk.
        Returns NaN when the output variance is zero (constant model).
    """
    # Pooled output vector of shape (2N,) for variance estimation.
    y = jnp.concatenate([A, B])
    # Scalar total output variance.
    var = jnp.var(y)
    # V_jk: combined first- and second-order variance of (j, k), shape ().
    Vjk = jnp.where(var == 0, jnp.nan, jnp.mean(BA_j * AB_k - A * B) / var)
    # Subtract individual main effects to isolate the pure interaction.
    Sj = first_order(A, AB_j, B)
    Sk = first_order(A, AB_k, B)
    return Vjk - Sj - Sk
