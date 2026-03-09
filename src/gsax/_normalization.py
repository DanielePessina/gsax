"""Shared output standardization helpers for analysis entrypoints."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _prenormalize_outputs(Y: Array) -> tuple[Array, Array, Array, Array]:
    """Standardize outputs over the sample axis.

    Args:
        Y: Output array with shape ``(N, ...)``. The first axis is treated as
            the sample axis, and all trailing axes are normalized
            independently.

    Returns:
        A tuple ``(Y_norm, y_mean, y_std, safe_scale)`` where:
            - ``Y_norm`` is the centered/scaled output array.
            - ``y_mean`` is the per-output-slice mean.
            - ``y_std`` is the per-output-slice original standard deviation.
            - ``safe_scale`` is the divisor actually used, with zeros replaced
              by ``1.0`` to avoid division by zero.
    """
    y_mean = jnp.mean(Y, axis=0)
    y_std = jnp.std(Y, axis=0)
    safe_scale = jnp.where(y_std == 0, jnp.ones_like(y_std), y_std)
    Y_norm = (Y - y_mean) / safe_scale
    return Y_norm, y_mean, y_std, safe_scale
