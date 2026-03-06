from dataclasses import dataclass

from jax import Array

from gsax.problem import Problem


@dataclass
class SAResult:
    S1: Array  # (T, K, D) or (K, D) if time squeezed
    S1_conf: Array
    ST: Array
    ST_conf: Array
    S2: Array | None  # (T, K, D, D) or (K, D, D), None if not computed
    S2_conf: Array | None
    problem: Problem

    def __repr__(self) -> str:
        shapes = {
            "S1": self.S1.shape,
            "ST": self.ST.shape,
            "S2": self.S2.shape if self.S2 is not None else None,
        }
        return f"SAResult({shapes})"
