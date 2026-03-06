import jax.numpy as jnp
import numpy as np
from jax import Array

from gsax.problem import Problem

PROBLEM = Problem.from_dict(
    {
        "x1": (-np.pi, np.pi),
        "x2": (-np.pi, np.pi),
        "x3": (-np.pi, np.pi),
    }
)

# Analytical solutions for A=7, B=0.1
ANALYTICAL_S1 = [0.3139, 0.4424, 0.0]
ANALYTICAL_ST = [0.5576, 0.4424, 0.2437]
ANALYTICAL_S2 = {(0, 2): 0.2437}  # x1-x3 interaction; others ~0


def evaluate(X: Array, A: float = 7.0, B: float = 0.1) -> Array:
    """f(x) = sin(x1) + A*sin^2(x2) + B*x3^4*sin(x1)"""
    return jnp.sin(X[:, 0]) + A * jnp.sin(X[:, 1]) ** 2 + B * X[:, 2] ** 4 * jnp.sin(X[:, 0])
