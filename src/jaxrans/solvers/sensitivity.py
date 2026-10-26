"""Derivatives of a converged steady solution with respect to parameters.

At a converged state ``R(U*(p), p) = 0``, so for outputs ``J(U, p)``

    dR/dU dU/dp = -dR/dp,         dJ/dp = dJ/dp|_U + dJ/dU dU/dp

one linear solve per parameter gives the derivative of every output (the
tangent, or forward, mode). ``dR/dU`` is the Jacobian of the residual that
was converged, never assembled: its products come from ``jax.linearize``.

The solve is defect correction, ``x <- x + G (b - A x)``, with ``G`` an
approximate inverse of ``A``: for a steady solver, the linear solve of its own
pseudo-time Newton step, so the iteration is that solver linearized at the
converged state, and converges as it does. Krylov methods on ``A`` itself,
preconditioned by the same line solves, stall on the exact Jacobian of a
second-order scheme (docs/dev/notes.md).

Equation-agnostic: ``residual(U, p)`` and ``outputs(U, p)`` (a dict of
scalars) are any differentiable functions, ``p`` a dict of scalars.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array, lax

Residual = Callable[[Array, dict], Array]
Outputs = Callable[[Array, dict], dict]


def defect_correction(matvec: Callable[[Array], Array], b: Array, approximate_inverse: Callable[[Array], Array],
                      tol: float, maxiter: int) -> tuple[Array, Array]:
    """``x <- x + G (b - A x)`` from zero until ``|b - A x| <= tol |b|`` or
    ``maxiter`` iterations; returns ``x`` and its relative residual."""
    b_norm = jnp.maximum(jnp.linalg.norm(b), jnp.finfo(b.dtype).tiny)

    def unconverged(carry):
        k, _, r = carry
        return (k < maxiter) & (jnp.linalg.norm(r) > tol * b_norm)

    def step(carry):
        k, x, r = carry
        x = x + approximate_inverse(r)
        return k + 1, x, b - matvec(x)

    _, x, r = lax.while_loop(unconverged, step, (0, jnp.zeros_like(b), b))
    return x, jnp.linalg.norm(r) / b_norm


def tangent(residual: Residual, outputs: Outputs, U: Array, p: dict, parameters: tuple[str, ...],
            approximate_inverse: Callable[[Array], Array], tol: float, maxiter: int) -> tuple[dict, dict]:
    """``(dJ/dp as {output: {parameter: value}}, {parameter: relative linear residual})``."""
    _, jacobian = jax.linearize(lambda u: residual(u, p), U)
    columns, errors = {}, {}
    for name in parameters:
        dp = {k: jnp.ones_like(v) if k == name else jnp.zeros_like(v) for k, v in p.items()}
        rhs = jax.jvp(lambda q: residual(U, q), (p,), (dp,))[1]
        dU, errors[name] = defect_correction(jacobian, -rhs, approximate_inverse, tol, maxiter)
        columns[name] = jax.jvp(outputs, (U, p), (dU, dp))[1]
    names = columns[parameters[0]]
    return {out: {name: columns[name][out] for name in parameters} for out in names}, errors
