"""Preconditioners for the implicit solver.

The approximate Jacobian is the first-order upwind one with scalar (Rusanov)
dissipation. With the face flux ``1/2 (F.S(U_L) + F.S(U_R)) - 1/2 lambda_f
(U_R - U_L)`` and ``S`` outward from the cell, its blocks are

    dR_c/dU_c  = 1/2 sum_f lambda_f I            (sum_f S_out = 0 removes A)
    dR_c/dU_nb = 1/2 A(U_nb, S_out) - 1/2 lambda_f I

and the pseudo-time term adds ``V/dt = sum_f lambda_f / CFL`` to the diagonal.

``line_j`` keeps the off-diagonal blocks along the atlas's line chains (cells
linked along each block's wall-normal index, ``j`` in 2D, continued through a
block's own cut: the C-grid wake) and solves each chain exactly,
block-tridiagonal; the other couplings are left to the Krylov iteration. The SA row is solved on
scalar chains of its own (upwind convection, thin-layer diffusion, the
destruction derivative). Mavriplis, AIAA J. 37 (1999).
"""

from __future__ import annotations

from typing import Callable, Protocol

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..core.linalg import factor_cyclic_reduction, solve_cyclic_reduction
from ..core.state import RHO, Rows, dot
from ..mesh.atlas import Atlas
from ..numerics import turbulence
from ..numerics.slabs import SlabAtlas
from ..numerics.residual import lower_upper, cell_wave_speed_sum, primitive_from_conservative
from ..physics import euler, navier_stokes
from ..physics import spalart_allmaras as sa


def families(fields: tuple[Array, ...]) -> Array:
    """Per-family face fields ``(..., *face shape)`` flattened and concatenated
    in family order, ``(..., n_faces)``: what the line links index."""
    return jnp.concatenate([f.reshape(f.shape[:f.ndim - len(fields)] + (-1,)) for f in fields], axis=-1)


class Preconditioner(Protocol):
    def __call__(self, U: Array, lambdas: tuple[Array, ...], cfl: Array) -> Callable[[Array], Array]:
        """Freeze at a state and the face wave speeds (one array per face
        family); return ``r -> M^-1 r`` on canvas fields."""
        ...


PreconditionerFactory = Callable[..., Preconditioner]
"""``(atlas, gamma, viscosity=None, model=None, distance=None) -> Preconditioner``."""

PRECONDITIONERS: dict[str, PreconditionerFactory] = {}


def register_preconditioner(name: str):
    def decorate(fn: PreconditionerFactory) -> PreconditionerFactory:
        if name in PRECONDITIONERS:
            raise KeyError(f"preconditioner '{name}' already registered")
        PRECONDITIONERS[name] = fn
        return fn

    return decorate


def get_preconditioner(name: str) -> PreconditionerFactory:
    try:
        return PRECONDITIONERS[name]
    except KeyError:
        raise KeyError(f"unknown preconditioner '{name}'. Available: {sorted(PRECONDITIONERS)}") from None


def diagonal(lambdas: tuple[Array, ...], cfl: Array) -> Array:
    """``sum_f lambda_f (1/CFL + 1/2)`` per cell."""
    return cell_wave_speed_sum(lambdas) * (1.0 / cfl + 0.5)


@register_preconditioner("none")
def identity(atlas: Atlas, gamma: float, viscosity=None, model=None, distance=None) -> Preconditioner:
    """No preconditioning: the control."""
    return lambda U, lambdas, cfl: (lambda r: r)


# --- Chains --------------------------------------------------------------------


class ChainLayout:
    """Gathering canvas fields onto the line chains and back, by flat index."""

    def __init__(self, atlas: Atlas):
        lines = atlas.lines
        valid = lines.cells >= 0
        inverse = np.full(int(np.prod(atlas.shape)), -1, dtype=np.int64)
        inverse[lines.cells[valid]] = np.flatnonzero(valid)     # canvas cell -> chain slot
        self.dim = atlas.dim
        self.valid = jnp.asarray(valid)
        self.cells = jnp.asarray(np.where(valid, lines.cells, 0))
        self.inverse = jnp.asarray(inverse)
        self.up_face, self.down_face = jnp.asarray(lines.up_face), jnp.asarray(lines.down_face)
        self.up_sign, self.down_sign = jnp.asarray(lines.up_sign), jnp.asarray(lines.down_sign)

    def _flat(self, field: Array) -> Array:
        return field.reshape(field.shape[:field.ndim - self.dim] + (-1,))

    def gather(self, field: Array) -> Array:
        """A canvas field ``(..., *canvas shape)`` on the chains, ``(..., n_lines, length)``."""
        return self._flat(field)[..., self.cells]

    def scatter(self, chains: Array, field: Array) -> Array:
        """Chain values ``(..., n_lines, length)`` back onto the canvas field
        ``field``; cells on no chain keep its value."""
        found = self.inverse >= 0
        values = chains.reshape(chains.shape[:-2] + (-1,))[..., jnp.where(found, self.inverse, 0)]
        return jnp.where(found, values, self._flat(field)).reshape(field.shape)

    def faces(self, fields: tuple[Array, ...], side: str) -> Array:
        """Per-family face fields ``(..., *face shape)`` at each chain link's
        face (``side``: ``"up"`` or ``"down"``), ``(..., n_lines, length)``;
        links that do not exist (sign 0) may read any face."""
        return families(fields)[..., self.up_face if side == "up" else self.down_face]


class SliceChains:
    """``ChainLayout`` for a 3D canvas split over devices in slabs of its first
    axis (``numerics/slabs.py``) whose chains are its lines along one other
    axis ``axis`` over ``span``: every block's line axis, every block spanning
    exactly ``span`` along it. The gathers are slices and transposes (a flat
    gather of a split canvas copies it to every device); every line of the
    canvas carries a chain, those outside the blocks inert (identity, no
    links, zero right-hand side), so the split axis is never sliced; the same
    values."""

    def __init__(self, atlas: Atlas, axis: int, span: slice):
        self.axis, self.span = axis, span
        inside = self._cut(atlas.interior)
        valid = np.moveaxis(inside, axis, -1)
        self.shape = valid.shape
        valid = valid.reshape(-1, valid.shape[-1])
        links = valid & np.roll(valid, -1, axis=1)
        links[:, -1] = False
        self.inside = jnp.asarray(inside)
        self.valid = jnp.asarray(valid)
        self.up_sign = jnp.asarray(links.astype(np.int64))
        self.down_sign = -self.up_sign

    def _cut(self, field, offset: int = 0):
        """``field`` (``(..., *canvas shape)``) along ``axis`` over ``span``, shifted by ``offset``."""
        index = [slice(None)] * 3
        index[self.axis] = slice(self.span.start + offset, self.span.stop + offset)
        return field[(Ellipsis,) + tuple(index)]

    def _chains(self, cut: Array) -> Array:
        moved = jnp.moveaxis(cut, cut.ndim - 3 + self.axis, -1)
        return moved.reshape(moved.shape[:-3] + (-1, moved.shape[-1]))

    def gather(self, field: Array) -> Array:
        return self._chains(self._cut(field))

    def scatter(self, chains: Array, field: Array) -> Array:
        block = chains.reshape(chains.shape[:-2] + self.shape)
        block = jnp.moveaxis(block, -1, block.ndim - 3 + self.axis)
        index = [slice(None)] * 3
        index[self.axis] = self.span
        inside = (Ellipsis,) + tuple(index)
        return field.at[inside].set(jnp.where(self.inside, block, field[inside]))

    def faces(self, fields: tuple[Array, ...], side: str) -> Array:
        return self._chains(self._cut(fields[self.axis], 1))          # the face above each cell


Layout = ChainLayout | SliceChains


def chain_layout(atlas: Atlas) -> Layout:
    """The chains' layout: by slices on a split 3D atlas whose blocks all run
    their lines along one canvas axis (not the split one) over the same
    span, by flat index otherwise."""
    if isinstance(atlas, SlabAtlas) and atlas.dim == 3:
        from ..mesh.atlas3d import line_axis

        axes = {line_axis(block) for block in atlas.mesh}
        spans = {(w[a].start, w[a].stop) for w in atlas.windows for a in axes}
        if len(axes) == 1 and len(spans) == 1 and axes != {0}:
            (axis,), ((start, stop),) = axes, spans
            layout = SliceChains(atlas, axis, slice(start, stop))
            if int(layout.valid.sum()) == int((atlas.lines.cells >= 0).sum()):
                return layout
    return ChainLayout(atlas)


def _scan_major(blocks: Array) -> Array:
    """``(m, m, n_lines, length)`` -> ``(length, n_lines, m, m)``."""
    return jnp.transpose(blocks, (3, 2, 0, 1))


def chain_solver(lower: Array, diag: Array, upper: Array, layout: Layout):
    """Factor the chains once; return ``r -> x`` on canvas fields ``(m, *canvas shape)``."""
    factor = factor_cyclic_reduction(lower, diag, upper)

    def solve(r: Array) -> Array:
        rhs = jnp.where(layout.valid, layout.gather(r), 0.0)             # (m, n, L)
        x = solve_cyclic_reduction(factor, jnp.transpose(rhs, (2, 1, 0)))  # (L, n, m)
        return layout.scatter(jnp.transpose(x, (2, 1, 0)), r)

    return solve


def flow_chain_blocks(U: Array, lambdas: tuple[Array, ...], cfl: Array, atlas: Atlas,
                      gamma: float, layout: Layout):
    """``(lower, diag, upper)`` of the mean-flow chains, ``(L, n_lines, d+2, d+2)``."""
    # Source geometry may be FP64 even when the Krylov fields are FP32.
    # Round metrics before forming blocks so factors and matvecs stay in U.dtype.
    grid = atlas.with_dtype(U.dtype).grid
    nv = grid.components
    m = Rows(nv).n_flow
    q = primitive_from_conservative(U, gamma, nv)
    eye = jnp.eye(m, dtype=U.dtype)
    d = jnp.where(layout.valid, layout.gather(diagonal(lambdas, cfl)), 1.0)
    diag = d.T[:, :, None, None] * eye
    S, lam = grid.faces, lambdas

    def coupling(side, sign, neighbour):
        """``1/2 A(U_nb, S_out) - 1/2 lambda_f I`` per link, zero where none."""
        block = (0.5 * euler.normal_flux_jacobian(neighbour, sign * layout.faces(S, side), gamma)
                 - 0.5 * layout.faces(lam, side) * eye.reshape(m, m, 1, 1))
        return _scan_major(jnp.where(sign != 0, block, 0.0))

    chains = layout.gather(q)                                             # (m, n, L)
    above = jnp.concatenate([chains[:, :, 1:], chains[:, :, -1:]], axis=2)
    upper = coupling("up", layout.up_sign, above)
    below_blocks = coupling("down", layout.down_sign, chains)  # row k+1 -> cell k
    lower = jnp.concatenate([jnp.zeros_like(below_blocks[:1]), below_blocks[:-1]], axis=0)
    return lower, diag, upper


def sa_chain_blocks(U: Array, lambdas: tuple[Array, ...], cfl: Array, atlas: Atlas,
                    gamma: float, viscosity, model: sa.SAModel, distance: Array, layout: Layout):
    """Scalar chain blocks for the SA row, unknown ``rho nu_hat``, ``(L, n_lines, 1, 1)``.

    Across a face with outward mass flux ``m`` and diffusion ``d = kappa
    |S|^2 / V``, a row gains ``(max(m, 0) + d) / rho`` on its diagonal and
    ``-(max(-m, 0) + d) / rho_nb`` toward its neighbour; the destruction
    derivative ``2 c_w1 f_w nu_inf nu_hat / d_wall^2`` (``f_w`` frozen, no
    production) and the mean flow's ``V/dt`` complete the diagonal.
    """
    grid = atlas.with_dtype(U.dtype).grid
    distance = jnp.asarray(distance, dtype=U.dtype)
    dim, nv = grid.dim, grid.components
    rows = Rows(nv)
    c, nu_inf = model.constants, model.nu_inf
    q = primitive_from_conservative(U, gamma, nv)
    q_faces = atlas.face_averages(q)

    def face_terms(qf, s, vol):
        mu = viscosity(navier_stokes.temperature(qf, gamma, nv))
        kappa = sa.diffusion_coefficient(qf[RHO], qf[rows.nu] * nu_inf, mu, c)
        return qf[RHO] * dot(qf[rows.velocity], s), kappa * dot(s, s) / vol

    terms = [face_terms(qf, s, vol) for qf, s, vol
             in zip(q_faces, grid.faces, atlas.face_averages(grid.volume))]
    rho = q[RHO]
    diag = None
    for a, (m_f, d_f) in enumerate(terms):
        out_lo, out_hi = lower_upper(jnp.maximum(m_f, 0.0) + d_f, a - dim)
        in_lo, in_hi = lower_upper(jnp.maximum(-m_f, 0.0) + d_f, a - dim)
        diag = out_hi + in_lo if diag is None else diag + out_hi + in_lo
    diag = diag / rho

    nu_hat = q[rows.nu]
    g = turbulence.green_gauss(tuple(qf[rows.velocity] for qf in q_faces), grid)
    nu = viscosity(navier_stokes.temperature(q, gamma, nv)) / rho
    s_t = sa.s_tilde(turbulence.vorticity_magnitude(g), nu_hat * nu_inf, nu, distance, c)
    f_w = sa.f_w(sa.r_of(nu_hat * nu_inf, s_t, distance, c), c)
    destruction = jnp.where(nu_hat > 0.0, 2.0 * c.c_w1 * f_w * nu_inf * nu_hat / distance**2, 0.0)
    diag = diag + destruction * grid.volume + cell_wave_speed_sum(lambdas) / cfl

    d_chain = jnp.where(layout.valid, layout.gather(diag), 1.0)
    rho_chain = layout.gather(rho)
    m, dj = tuple(t[0] for t in terms), tuple(t[1] for t in terms)

    def coupling(side, sign, rho_nb):
        block = -(jnp.maximum(-sign * layout.faces(m, side), 0.0) + layout.faces(dj, side)) / rho_nb
        return jnp.where(sign != 0, block, 0.0).T[:, :, None, None]

    above = jnp.concatenate([rho_chain[:, 1:], rho_chain[:, -1:]], axis=1)
    upper = coupling("up", layout.up_sign, above)
    below_blocks = coupling("down", layout.down_sign, rho_chain)
    lower = jnp.concatenate([jnp.zeros_like(below_blocks[:1]), below_blocks[:-1]], axis=0)
    return lower, d_chain.T[:, :, None, None], upper


@register_preconditioner("line_j")
def line_implicit_j(atlas: Atlas, gamma: float, viscosity=None, model=None, distance=None) -> Preconditioner:
    """Exact block-tridiagonal solves along the line chains; the SA row on its
    own scalar chains, its coupling to the mean flow left to the Krylov
    iteration."""
    layout = chain_layout(atlas)
    n_flow = Rows(atlas.components).n_flow

    def prepare(U: Array, lambdas: tuple[Array, ...], cfl: Array):
        solve_flow = chain_solver(*flow_chain_blocks(U[:n_flow], lambdas, cfl, atlas, gamma, layout), layout)
        if U.shape[0] == n_flow:
            return solve_flow
        solve_sa = chain_solver(*sa_chain_blocks(U, lambdas, cfl, atlas, gamma, viscosity,
                                                 model, distance, layout), layout)
        return lambda r: jnp.concatenate([solve_flow(r[:n_flow]), solve_sa(r[n_flow:])], axis=0)

    return prepare
