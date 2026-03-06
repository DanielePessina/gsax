"""Defines the SAResult dataclass for storing sensitivity analysis results."""

from dataclasses import dataclass

from jax import Array

from gsax.problem import Problem


@dataclass
class SAResult:
    """Sobol sensitivity analysis results.

    Stores first-order (S1), total-order (ST), and optionally second-order (S2)
    Sobol indices, with optional bootstrap confidence intervals.

    Shapes follow the convention ``(T, K, D)`` for time-resolved analyses or
    ``(K, D)`` when the time dimension is squeezed, where *K* is the number of
    outputs and *D* the number of parameters.

    Confidence interval arrays (``*_conf``) have an extra leading dimension of
    size 2 representing ``[lower, upper]`` bounds.
    """

    S1: Array  # (T, K, D) or (K, D) if time squeezed
    ST: Array
    S2: Array | None  # (T, K, D, D) or (K, D, D), None if not computed
    problem: Problem
    S1_conf: Array | None = None  # (2, T, K, D) or squeezed; [lower, upper]
    ST_conf: Array | None = None
    S2_conf: Array | None = None
    nan_counts: dict[str, int] | None = None

    def __repr__(self) -> str:
        """Return a concise summary showing index shapes."""
        shapes = {
            "S1": self.S1.shape,
            "ST": self.ST.shape,
            "S2": self.S2.shape if self.S2 is not None else None,
        }
        if self.S1_conf is not None:
            shapes["S1_conf"] = self.S1_conf.shape
            shapes["ST_conf"] = self.ST_conf.shape
            if self.S2_conf is not None:
                shapes["S2_conf"] = self.S2_conf.shape
        return f"SAResult({shapes})"
