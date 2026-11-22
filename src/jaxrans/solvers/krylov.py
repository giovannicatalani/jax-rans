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


def _dot(V: Array, w: Array) -> Array:
    """``V_k . w`` for every basis vector, ``(k,)``."""
    return jnp.tensordot(V, w, axes=w.ndim)


def gmres_right(
    matvec: Callable[[Array], Array],
    b: Array,
    precondition: Callable[[Array], Array],
    restart: int,
    maxiter: int,
    tol: float,
    *,
    return_action: bool = False,
    action_from_arnoldi: bool = False,
) -> Array | tuple[Array, Array]:
    """Restarted GMRES(``restart``), right-preconditioned (``precondition: r -> M^-1 r``),
    for ``matvec`` on any array shape; stops at ``|b - A x| <= tol |b|`` between
    cycles, at most ``maxiter`` cycles. Zero for ``b = 0``.

    The residual ``b - A x`` that ends one cycle starts the next (one operator
    product per cycle, not two). ``return_action`` also returns ``A x``, so a
    caller that needs it does not pay another product. ``action_from_arnoldi``
    takes that action from the Arnoldi relation ``A M^-1 V_m y = V_{m+1} H y``
    instead of a product per cycle: equal in exact arithmetic, to
    orthogonalization round-off with CGS2. Meant for the few cycles of a Newton
    solve; long restarted sequences (the adjoint) keep the true residual.
    """
    b_norm = jnp.linalg.norm(b)
    tiny = jnp.finfo(b.dtype).tiny
    rcond = float(jnp.finfo(b.dtype).eps) * (restart + 1)

    def cycle(x: Array, r: Array) -> tuple[Array, Array]:
        beta = jnp.linalg.norm(r)
        V = jnp.zeros((restart + 1,) + b.shape, b.dtype).at[0].set(r / jnp.maximum(beta, tiny))
        H = jnp.zeros((restart + 1, restart), b.dtype)

        def arnoldi(j, carry):
            V, H = carry
            w = matvec(precondition(V[j]))
            active = (jnp.arange(restart + 1) <= j).astype(b.dtype)
            h = _dot(V, w) * active
            w = w - jnp.tensordot(h, V, axes=1)
            h2 = _dot(V, w) * active                  # the second pass of CGS2
            w = w - jnp.tensordot(h2, V, axes=1)
            h = h + h2
            norm = jnp.linalg.norm(w)
            H = H.at[:, j].set(h.at[j + 1].set(norm))
            V = V.at[j + 1].set(w / jnp.maximum(norm, tiny))
            return V, H

        V, H = lax.fori_loop(0, restart, arnoldi, (V, H))
        y = hessenberg_lstsq(H, beta, rcond)
        correction = precondition(jnp.tensordot(y, V[:restart], axes=1))
        return x + correction, jnp.tensordot(H @ y, V, axes=1)

    def unconverged(carry):
        k, _, r, _ = carry
        return (k < maxiter) & (jnp.linalg.norm(r) > tol * b_norm)

    def body(carry):
        k, x, r, action = carry
        x, delta = cycle(x, r)
        action = action + delta if action_from_arnoldi else matvec(x)
        return k + 1, x, b - action, action

    x0 = jnp.zeros_like(b)
    _, x, _, action = lax.while_loop(unconverged, body, (0, x0, b, x0))
    return (x, action) if return_action else x


def hessenberg_lstsq(H: Array, beta: Array, rcond: float) -> Array:
    """``argmin |beta e1 - H y|`` for the ``(m + 1, m)`` upper Hessenberg ``H``
    of a GMRES cycle, by Givens rotations unrolled at trace time (Saad,
    Algorithm 6.9): about ``m^2`` scalar operations that XLA fuses, where
    ``jnp.linalg.lstsq`` launches a vendor SVD every cycle.

    A direction whose pivot is below ``rcond`` times the largest pivot is
    dropped (``y_j = 0``), the analogue of ``lstsq``'s singular-value cutoff,
    so a Krylov breakdown is not read as a direction; ``y = 0`` when ``H``
    and ``beta`` vanish."""
    m = H.shape[1]
    h = [[H[i, j] for j in range(m)] for i in range(m + 1)]
    zero = jnp.zeros((), H.dtype)
    g = [beta.astype(H.dtype)] + [zero] * m
    cs, sn = [], []
    for j in range(m):
        for i in range(j):                                    # earlier rotations on column j
            a, c = h[i][j], h[i + 1][j]
            h[i][j], h[i + 1][j] = cs[i] * a + sn[i] * c, cs[i] * c - sn[i] * a
        a, c = h[j][j], h[j + 1][j]
        r = jnp.sqrt(a * a + c * c)
        safe = jnp.where(r > 0, r, 1.0)
        cs.append(jnp.where(r > 0, a / safe, 1.0))
        sn.append(jnp.where(r > 0, c / safe, 0.0))
        h[j][j], h[j + 1][j] = r, zero
        g[j], g[j + 1] = cs[j] * g[j], -sn[j] * g[j]
    pivots = [jnp.abs(h[j][j]) for j in range(m)]
    largest = pivots[0]
    for p in pivots[1:]:
        largest = jnp.maximum(largest, p)
    y = [zero] * m
    for j in reversed(range(m)):
        rest = g[j]
        for k in range(j + 1, m):
            rest = rest - h[j][k] * y[k]
        keep = pivots[j] > rcond * largest
        y[j] = jnp.where(keep, rest / jnp.where(keep, h[j][j], 1.0), 0.0)
    return jnp.stack(y)
