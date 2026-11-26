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
1069). The Arnoldi basis is a fixed-size array. A cycle can stop early using
the residual tracked by the Givens rotations.

Reference: Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM
(2003), Algorithms 6.9 and 9.5.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import Array, lax

from ..core.precision import l2_norm


def gmres_right(
    matvec: Callable[[Array], Array],
    b: Array,
    precondition: Callable[[Array], Array],
    restart: int,
    maxiter: int,
    tol: float,
    *,
    basis_dtype=None,
    operator_dtype=None,
    return_action: bool = False,
    action_from_arnoldi: bool = False,
    early_exit: bool = False,
) -> Array | tuple[Array, Array]:
    """Restarted GMRES(``restart``), right-preconditioned (``precondition: r -> M^-1 r``),
    for ``matvec`` on any array shape; stops at ``|b - A x| <= tol |b|`` between
    cycles, at most ``maxiter`` cycles. Zero for ``b = 0``.

    ``basis_dtype`` optionally stores/orthogonalizes Krylov vectors and applies
    the preconditioner in that dtype. Operator inputs, accumulated corrections
    and restart residuals retain ``b.dtype``; the small Hessenberg stays FP64.
    ``operator_dtype`` overrides the inputs to ``matvec`` for both Arnoldi and
    the true action at each restart.
    Leaving it unset preserves the ordinary field-precision path.
    ``return_action`` also returns the last true operator action, avoiding a
    duplicate JVP in the caller. Restart residuals are carried between cycles.
    ``action_from_arnoldi`` accumulates that action from the Arnoldi relation
    (``A M^-1 V_m y = V_{m+1} H y``) instead of one more operator product per
    cycle: equal in exact arithmetic, to orthogonalization round-off with
    CGS2. Meant for few cycles (the Newton solve); long restarted sequences
    keep the true residual (the default).
    ``early_exit`` ends a cycle as soon as its least-squares residual (exact
    in exact arithmetic, from the Hessenberg matrix) meets ``tol |b|``,
    instead of always building ``restart`` vectors; under ``vmap`` a cycle
    runs until every case has met it.
    """
    b_norm = l2_norm(b)
    vector_dtype = b.dtype if basis_dtype is None else jnp.dtype(basis_dtype)
    matvec_dtype = b.dtype if operator_dtype is None else jnp.dtype(operator_dtype)
    precision = (lax.Precision.DEFAULT if basis_dtype is not None and vector_dtype == jnp.float32
                 else lax.Precision.HIGHEST)
    tiny = jnp.finfo(vector_dtype).tiny

    def apply_operator(v):
        # JVP tangents must have precisely the primal (state) dtype.
        return matvec(v.astype(matvec_dtype))

    def orthogonalize(width):
        def cgs2(V, w, j):
            basis = V[:width]
            active = jnp.arange(width) <= j
            h = jnp.where(active, jnp.tensordot(basis, w, axes=w.ndim, precision=precision), 0)
            w = w - jnp.tensordot(h, basis, axes=1, precision=precision)
            h2 = jnp.where(active, jnp.tensordot(basis, w, axes=w.ndim, precision=precision), 0)
            w = w - jnp.tensordot(h2, basis, axes=1, precision=precision)
            return jnp.pad(h + h2, (0, restart + 1 - width)), w
        return cgs2

    # Static prefixes keep the contractions batched while reading at most
    # twice as many basis vectors as are active in an Arnoldi step.
    widths = [2 ** k for k in range((int(restart) - 1).bit_length())] + [restart]
    limits = jnp.asarray(widths[:-1])
    prefixes = tuple(orthogonalize(width) for width in widths)

    def cycle(x: Array, r: Array) -> tuple[Array, Array]:
        beta = l2_norm(r)
        V = jnp.zeros((restart + 1,) + b.shape, vector_dtype).at[0].set(
            (r / jnp.maximum(beta, tiny)).astype(vector_dtype))
        # Only this tiny least-squares problem uses FP64 in an FP32 solve.
        H = jnp.zeros((restart + 1, restart), jnp.float64)

        def arnoldi(j, carry):
            V, H = carry
            w = apply_operator(precondition(V[j])).astype(vector_dtype)
            h, w = lax.switch(jnp.sum(j >= limits, dtype=jnp.int32), prefixes, V, w, j)
            norm = l2_norm(w)
            H = H.at[:, j].set(h.astype(H.dtype).at[j + 1].set(norm))
            V = V.at[j + 1].set(w / jnp.maximum(norm, tiny))
            return V, H

        # H was measured with field-precision vectors: do not interpret FP32
        # round-off as extra independent Krylov directions just because the
        # small factorization itself is done in double.
        rcond = float(jnp.finfo(vector_dtype).eps) * max(H.shape)
        if early_exit:
            target = tol * b_norm.astype(jnp.float64)
            g = jnp.zeros(restart + 1, H.dtype).at[0].set(beta.astype(H.dtype))
            R = jnp.zeros((restart, restart), H.dtype)
            cs, sn = jnp.ones(restart, H.dtype), jnp.zeros(restart, H.dtype)

            def more(carry):
                j, *_, met = carry
                return (j < restart) & ~met

            def step(carry):
                j, V, H, R, g, cs, sn, _ = carry
                V, H = arnoldi(j, (V, H))
                column = _rotate_column(H[:, j], cs, sn)
                pivot, c, s = _givens(column[j], column[j + 1])
                cs, sn = cs.at[j].set(c), sn.at[j].set(s)
                column = column.at[j].set(pivot).at[j + 1].set(0)
                R = R.at[:, j].set(column[:-1])
                tail = -s * g[j]
                g = g.at[j].set(c * g[j]).at[j + 1].set(tail)
                diagonal = jnp.abs(jnp.diag(R))
                dropped = jnp.any((jnp.arange(restart) <= j) & (diagonal <= rcond * diagonal.max()))
                small = jnp.abs(tail) <= target

                def rank_deficient_residual():
                    y = _backsolve(R, g[:-1], rcond)
                    return jnp.hypot(l2_norm(g[:-1] - R @ y), tail) <= target

                # A discarded pivot can leave residual above the trailing row.
                met = lax.cond(small & dropped, rank_deficient_residual, lambda: small)
                return j + 1, V, H, R, g, cs, sn, met

            _, V, H, R, g, _, _, _ = lax.while_loop(
                more, step, (0, V, H, R, g, cs, sn, jnp.asarray(False)))
            y64 = _backsolve(R, g[:-1], rcond)
        else:
            V, H = lax.fori_loop(0, restart, arnoldi, (V, H))
            y64 = hessenberg_lstsq(H, beta.astype(jnp.float64), rcond)
        y = y64.astype(vector_dtype)
        correction = precondition(jnp.tensordot(y, V[:restart], axes=1, precision=lax.Precision.HIGHEST))
        # A M^-1 V_m y = V_{m+1} H y (the Arnoldi relation): the operator
        # action of the correction, without another Jacobian-vector product.
        delta = jnp.tensordot((H @ y64).astype(vector_dtype), V, axes=1, precision=lax.Precision.HIGHEST)
        if basis_dtype is not None:
            correction, delta = correction.astype(b.dtype), delta.astype(b.dtype)
        return x + correction, delta

    def unconverged(carry):
        k, x, r, action = carry
        return (k < maxiter) & (l2_norm(r) > tol * b_norm)

    def body(carry):
        k, x, r, action = carry
        x, delta = cycle(x, r)
        action = action + delta if action_from_arnoldi else apply_operator(x).astype(b.dtype)
        return k + 1, x, b - action, action

    x0 = jnp.zeros_like(b)
    _, x, _, action = lax.while_loop(unconverged, body, (0, x0, b, x0))
    return (x, action) if return_action else x


def _givens(a: Array, b: Array) -> tuple[Array, Array, Array]:
    pivot = jnp.sqrt(a * a + b * b)
    safe = jnp.where(pivot > 0, pivot, 1.0)
    return pivot, jnp.where(pivot > 0, a / safe, 1.0), jnp.where(pivot > 0, b / safe, 0.0)


def _rotate_column(column: Array, cs: Array, sn: Array) -> Array:
    # Apply the earlier rotations in parallel. Each sends the previous
    # rotation's lower entry x to -s*x + c*column[i+1].
    def compose(left, right):
        a, b = left
        c, d = right
        return c * a, c * b + d

    factors, offsets = lax.associative_scan(compose, (-sn, cs * column[1:]))
    lower = factors * column[0] + offsets
    incoming = jnp.concatenate([column[:1], lower[:-1]])
    return jnp.concatenate([cs * incoming + sn * column[1:], lower[-1:]])


def _backsolve(R: Array, rhs: Array, rcond: float) -> Array:
    pivots = jnp.abs(jnp.diag(R))
    keep = pivots > rcond * pivots.max()
    safe = jnp.where(keep[:, None], R, jnp.eye(R.shape[0], dtype=R.dtype))
    rhs = jnp.where(keep, rhs, 0)
    return lax.linalg.triangular_solve(safe, rhs[:, None], left_side=True, lower=False)[:, 0]


def hessenberg_lstsq(H: Array, beta: Array, rcond: float) -> Array:
    """``argmin |beta e1 - H y|`` for the ``(m + 1, m)`` upper Hessenberg ``H``
    of a GMRES cycle, by Givens rotations unrolled at trace time: about
    ``m^2`` scalar operations before triangular substitution, where
    ``jnp.linalg.lstsq`` dispatches a vendor SVD (cuSOLVER) every cycle.

    Directions whose pivot is below ``rcond`` times the largest pivot are
    dropped (``y_j = 0``), the analogue of ``lstsq``'s singular-value cutoff:
    a Krylov breakdown or round-off is not read as a direction. ``y = 0``
    when ``H`` and ``beta`` vanish."""
    m = H.shape[1]
    h = [[H[i, j] for j in range(m)] for i in range(m + 1)]
    zero = jnp.zeros((), H.dtype)
    g = [beta.astype(H.dtype)] + [zero] * m
    cs, sn = [], []
    for j in range(m):
        for i in range(j):                                    # earlier rotations on column j
            a, b = h[i][j], h[i + 1][j]
            h[i][j], h[i + 1][j] = cs[i] * a + sn[i] * b, cs[i] * b - sn[i] * a
        a, b = h[j][j], h[j + 1][j]
        r, c, s = _givens(a, b)
        cs.append(c)
        sn.append(s)
        h[j][j], h[j + 1][j] = r, zero
        g[j], g[j + 1] = c * g[j], -s * g[j]
    return _backsolve(jnp.stack([jnp.stack(row) for row in h[:-1]]), jnp.stack(g[:-1]), rcond)
