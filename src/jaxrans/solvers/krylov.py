"""Right-preconditioned restarted GMRES.

Solves ``A x = b`` as ``(A M^-1) y = b``, ``x = M^-1 y``. GMRES minimizes the
residual of the system it is given, so with the preconditioner on the right the
minimized quantity is the *true* residual ``|b - A x|``: an iteration can never
leave it larger than the zero vector does. Left preconditioning
(``jax.scipy.sparse.linalg.gmres``) minimizes ``|M^-1 (b - A x)|`` instead, and
the true residual is then unconstrained -- and the CFL ramp in ``implicit.py``
keys off the true reduction (docs/dev/notes.md).

Orthogonalization is classical Gram-Schmidt applied twice (CGS2): as stable as
modified Gram-Schmidt in practice and two matrix products per step instead of a
sequential loop (Giraud, Langou and Rozloznik, *Comput. Math. Appl.* 50 (2005)
1069). The Arnoldi basis is a fixed-size array, so each restart runs the full
``restart`` iterations; convergence is tested between restarts.

Reference: Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM
(2003), Algorithms 6.9 and 9.5.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import Array, lax

from ..core.precision import l2_norm


def _dot(V: Array, w: Array) -> Array:
    """``V_k . w`` for every basis vector, ``(k,)``."""
    return jnp.tensordot(V, w, axes=w.ndim, precision=lax.Precision.HIGHEST)


def gmres_right(
    matvec: Callable[[Array], Array],
    b: Array,
    precondition: Callable[[Array], Array],
    restart: int,
    maxiter: int,
    tol: float,
    *,
    basis_dtype=None,
    return_action: bool = False,
) -> Array | tuple[Array, Array]:
    """Restarted GMRES(``restart``), right-preconditioned (``precondition: r -> M^-1 r``),
    for ``matvec`` on any array shape; stops at ``|b - A x| <= tol |b|`` between
    cycles, at most ``maxiter`` cycles. Zero for ``b = 0``.

    ``basis_dtype`` optionally stores/orthogonalizes Krylov vectors and applies
    the preconditioner in that dtype. Operator inputs, accumulated corrections
    and restart residuals retain ``b.dtype``; the small Hessenberg stays FP64.
    Leaving it unset preserves the ordinary field-precision path.
    ``return_action`` also returns the last true operator action, avoiding a
    duplicate JVP in the caller. Restart residuals are carried between cycles.
    """
    b_norm = l2_norm(b)
    vector_dtype = b.dtype if basis_dtype is None else jnp.dtype(basis_dtype)
    tiny = jnp.finfo(vector_dtype).tiny

    def apply_operator(v):
        # JVP tangents must have precisely the primal (state) dtype.
        return matvec(v.astype(b.dtype)).astype(vector_dtype)

    def cycle(x: Array, r: Array) -> Array:
        beta = l2_norm(r)
        V = jnp.zeros((restart + 1,) + b.shape, vector_dtype).at[0].set(
            (r / jnp.maximum(beta, tiny)).astype(vector_dtype))
        # Only this tiny least-squares problem uses FP64 in an FP32 solve.
        H = jnp.zeros((restart + 1, restart), jnp.float64)

        def arnoldi(j, carry):
            V, H = carry
            w = (matvec(precondition(V[j])) if basis_dtype is None
                 else apply_operator(precondition(V[j])))
            active = (jnp.arange(restart + 1) <= j).astype(vector_dtype)
            h = _dot(V, w) * active
            w = w - jnp.tensordot(h, V, axes=1, precision=lax.Precision.HIGHEST)
            h2 = _dot(V, w) * active                  # the second pass of CGS2
            w = w - jnp.tensordot(h2, V, axes=1, precision=lax.Precision.HIGHEST)
            h = h + h2
            norm = l2_norm(w)
            H = H.at[:, j].set(h.at[j + 1].set(norm))
            V = V.at[j + 1].set(w / jnp.maximum(norm, tiny))
            return V, H

        V, H = lax.fori_loop(0, restart, arnoldi, (V, H))
        e1 = jnp.zeros(restart + 1, H.dtype).at[0].set(beta)
        # H was measured with field-precision vectors: do not interpret FP32
        # round-off as extra independent Krylov directions just because the
        # small factorization itself is done in double.
        rcond = jnp.finfo(vector_dtype).eps * max(H.shape)
        y = jnp.linalg.lstsq(H, e1, rcond=rcond)[0].astype(vector_dtype)
        correction = precondition(jnp.tensordot(y, V[:restart], axes=1, precision=lax.Precision.HIGHEST))
        return x + (correction if basis_dtype is None else correction.astype(b.dtype))

    def unconverged(carry):
        k, x, r, action = carry
        return (k < maxiter) & (l2_norm(r) > tol * b_norm)

    def body(carry):
        k, x, r, _ = carry
        x = cycle(x, r)
        action = matvec(x)
        return k + 1, x, b - action, action

    x0 = jnp.zeros_like(b)
    _, x, _, action = lax.while_loop(unconverged, body, (0, x0, b, x0))
    return (x, action) if return_action else x
