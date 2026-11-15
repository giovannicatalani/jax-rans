"""Precision guards and reductions; field arrays keep their selected dtype.

FP32 reductions accumulate in FP64, but vector products and the expensive CFD
operators stay in FP32. FP64 follows the original expressions.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array


def floor_like(a: Array) -> float:
    """The existing 1e-300 guard, raised to the smallest normal for FP32."""
    return max(1e-300, np.finfo(a.dtype).tiny)


def l2_norm(a: Array, *, dtype=None) -> Array:
    """Euclidean norm, scaled before squaring in FP32 to avoid overflow.

    ``dtype=float64`` retains the reduction's precision for scalar decisions;
    the default keeps field-vector normalization in the field dtype.
    """
    if a.dtype != jnp.float32:
        norm = jnp.linalg.norm(a)
        return norm if dtype is None else norm.astype(dtype)
    scale = jnp.max(jnp.abs(a))
    safe = jnp.where(scale > 0, scale, 1.0)
    square = (a / safe) ** 2
    root = jnp.sqrt(jnp.sum(square, dtype=jnp.float64))
    if dtype is not None:
        return scale.astype(dtype) * root.astype(dtype)
    return scale * root.astype(a.dtype)


def weighted_rms(a: Array, weight: Array) -> Array:
    """Volume-weighted RMS, with FP64 accumulation in the FP32 path."""
    if a.dtype != jnp.float32:
        return jnp.sqrt(jnp.sum(weight * a ** 2) / jnp.sum(weight))
    scale = jnp.max(jnp.abs(a))
    safe = jnp.where(scale > 0, scale, 1.0)
    square = weight * (a / safe) ** 2
    mean = jnp.sum(square, dtype=jnp.float64) / jnp.sum(weight, dtype=jnp.float64)
    return scale * jnp.sqrt(mean).astype(a.dtype)
