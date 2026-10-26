"""Wall pressure and friction, and the integrated force coefficients.

Nondimensional as everywhere (core/constants.py): ``p_inf = 1/gamma`` and
``q_inf = M^2 / 2``. Pressure comes from the cell on each wall face, friction
from the viscous flux the residual uses there; the force on the body is

    F = sum_faces [ -(p - p_inf) S + tau . S ],     S into the fluid,

gauge pressure, so an open wall (a plate wetted on one side) is right too.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..core.config import FlowConditions
from ..core.state import RHO, Rows, conservative_to_primitive, freestream_primitive, magnitude, split
from ..mesh.atlas import Atlas
from ..numerics import boundary as bc
from ..numerics.residual import primitive_from_conservative, viscous_fluxes
from ..physics.navier_stokes import Viscosity


def dynamic_pressure(flow: FlowConditions) -> float:
    return 0.5 * flow.mach**2


def wall_loads(U: Array, atlas: Atlas, flow: FlowConditions, freestream: Array | None = None,
               mu_inf: Array | None = None) -> tuple[Array, Array, Array]:
    """Per wall face: gauge pressure ``(n,)``, the area vector into the fluid
    ``(d, n)``, and the viscous force on the body ``(d, n)`` (zero if
    inviscid); ``freestream`` and ``mu_inf`` (traceable) replace ``flow``'s."""
    gamma, dim = flow.gamma, atlas.dim
    rows = Rows(dim)
    q = primitive_from_conservative(U[:rows.n_flow], gamma, dim)
    p = q[rows.pressure].ravel()[atlas.walls.cell] - 1.0 / gamma
    S = bc.wall_vectors(atlas)
    shear = jnp.zeros((dim,) + p.shape, dtype=U.dtype)
    if not flow.inviscid:
        q_inf = jnp.asarray(freestream_primitive(flow, dim) if freestream is None else freestream,
                            dtype=U.dtype)
        conditions = bc.Conditions(q_inf, gamma, bc.uniform_far_field(q, atlas, q_inf, gamma))
        filled = jnp.pad(bc.fill_ghosts(q, atlas, conditions), ((0, 0),) + ((1, 1),) * dim, mode="edge")
        visc = viscous_fluxes(filled, atlas.grid, Viscosity.of(flow, mu_inf), gamma, centres=atlas.centres)
        shear = atlas.walls.into_fluid * atlas.wall_values(tuple(v[rows.velocity] for v in visc))
    return p, S, shear


@dataclass(frozen=True)
class Forces:
    """Lift, drag (pressure + friction) and pitching moment coefficients; the
    moment about ``(0.25, 0)``, nose-up positive, pressure only. ``bodies``:
    ``(cl, cd, cm)`` of each body when there are several (numbered by
    leading-edge ``x``: slat, main, flap)."""

    cl: float
    cd: float
    cm: float
    fx: float
    fy: float
    cd_friction: float = 0.0
    bodies: tuple[tuple[float, float, float], ...] = ()
    fz: float = 0.0
    cy: float = 0.0

    def summary(self) -> str:
        friction = f" (friction {self.cd_friction:+.6f})" if self.cd_friction else ""
        side = f"   CY = {self.cy:+.6f}" if self.cy else ""
        text = f"CL = {self.cl:+.6f}   CD = {self.cd:+.6f}{friction}   CM = {self.cm:+.6f}{side}"
        for k, (cl, cd, cm) in enumerate(self.bodies):
            text += f"\n  body {k}: CL = {cl:+.6f}   CD = {cd:+.6f}   CM = {cm:+.6f}"
        return text


@dataclass(frozen=True)
class ForceReference:
    """What the coefficients are per: ``length`` (chord), ``area`` (3D; in 2D
    the coefficients are per unit span, the area is the length) and the
    ``centre`` the pitching moment is about."""

    length: float = 1.0
    area: float | None = None
    centre: tuple[float, ...] = (0.25, 0.0, 0.0)


def force_coefficients(U: Array, atlas: Atlas, flow: FlowConditions,
                       reference: ForceReference = ForceReference(),
                       freestream: Array | None = None, mu_inf: Array | None = None) -> dict[str, Array]:
    """``cl``, ``cd``, ``cm``, ``cd_friction``, ``cy`` (side force, 3D), the
    total ``force`` ``(d,)`` and each body's ``(cl, cd, cm)``, ``bodies``
    ``(n_bodies, 3)``; traceable in ``U`` and in ``freestream`` (primitives)
    and ``mu_inf``, which replace ``flow``'s (a sensitivity's parameters).

    Drag along the freestream velocity, lift normal to it in the plane of
    incidence (the first and the last axis, core/state.py), per ``q_inf *
    area`` (in 2D per unit span: the area is the length). The pitching moment
    ``sum (r_up f_x - r_x f_up)`` about ``reference.centre``, nose-up
    positive, per ``q_inf * area * length``: pressure only in 2D, pressure and
    friction in 3D (as TMR's CMY)."""
    dim = atlas.dim
    if dim == 3 and reference.area is None:
        raise ValueError("a 3D case needs domain.reference_area")
    q = jnp.asarray(freestream_primitive(flow, dim) if freestream is None else freestream, dtype=U.dtype)
    p, S, shear = wall_loads(U, atlas, flow, q, mu_inf)
    faces = atlas.walls.faces
    rho, velocity, _ = split(q, dim)
    speed = magnitude(velocity)
    drag = velocity / speed
    lift = jnp.stack([-drag[1], drag[0]] if dim == 2 else [-drag[2], jnp.zeros_like(speed), drag[0]])
    q_area = 0.5 * rho * speed**2 * (reference.length if dim == 2 else reference.area)
    pressure = -p * S                                                   # (d, n)
    load = pressure + shear
    arm = pressure if dim == 2 else load
    r = jnp.asarray(faces.mid - np.asarray(reference.centre[:dim]))
    moment = r[:, -1] * arm[0] - r[:, 0] * arm[-1]
    per_face = jnp.stack([lift @ load, drag @ load, moment / reference.length]) / q_area   # (3, n)
    totals, force = per_face.sum(axis=1), load.sum(axis=1)
    return {"cl": totals[0], "cd": totals[1], "cm": totals[2],
            "cd_friction": drag @ shear.sum(axis=1) / q_area,
            "cy": force[1] / q_area if dim == 3 else jnp.zeros_like(q_area),
            "force": force,
            "bodies": jax.ops.segment_sum(per_face.T, jnp.asarray(faces.body), faces.bodies)}


def integrate_forces(U: Array, atlas: Atlas, flow: FlowConditions,
                     reference: ForceReference | float = ForceReference()) -> Forces:
    """``force_coefficients`` as Python floats; ``reference`` (or the
    reference length alone)."""
    if not isinstance(reference, ForceReference):
        reference = ForceReference(length=float(reference))
    c = {k: np.asarray(v) for k, v in force_coefficients(U, atlas, flow, reference).items()}
    force = [float(f) for f in c["force"]] + [0.0]
    bodies = tuple(tuple(float(x) for x in b) for b in c["bodies"]) if len(c["bodies"]) > 1 else ()
    return Forces(float(c["cl"]), float(c["cd"]), float(c["cm"]), force[0], force[1],
                  float(c["cd_friction"]), bodies, fz=force[2], cy=float(c["cy"]))


def surface(U: Array, atlas: Atlas, flow: FlowConditions) -> dict[str, np.ndarray]:
    """Per wall face, body by body in loop order: ``x``, ``y``, ``body``,
    ``cp`` and, viscous, ``cf`` (along the downstream-pointing tangent,
    negative where separated)."""
    if atlas.dim == 3:
        return _surface_3d(U, atlas, flow)
    p, (sx, sy), shear = (np.asarray(a) for a in wall_loads(U, atlas, flow))
    faces = atlas.walls.faces
    out = {"x": faces.mid[:, 0], "y": faces.mid[:, 1], "body": faces.body.astype(float),
           "cp": p / dynamic_pressure(flow)}
    if not flow.inviscid:
        length = np.hypot(sx, sy)
        tx, ty = sy / length, -sx / length
        sign = np.where(tx >= 0.0, 1.0, -1.0)
        out["cf"] = sign * (shear[0] * tx + shear[1] * ty) / length / dynamic_pressure(flow)
    return out


def _surface_3d(U: Array, atlas: Atlas, flow: FlowConditions) -> dict[str, np.ndarray]:
    """Per wall face: its ``block`` and running position on its side ``i``,
    ``k`` (the two indices along a ``j`` side; the wing sections of
    validation/sections.py use them), ``x``, ``y``, ``z``, ``cp`` and, viscous,
    the skin friction vector ``cf_x``, ``cf_y``, ``cf_z`` (wall shear over
    ``q_inf``)."""
    p, S, shear = (np.asarray(a) for a in wall_loads(U, atlas, flow))
    faces = atlas.walls.faces
    mid = faces.mid
    q = dynamic_pressure(flow)
    out = {"block": faces.block.astype(float),
           "i": faces.cell[:, 0].astype(float), "k": faces.cell[:, 1].astype(float),
           "x": mid[:, 0], "y": mid[:, 1], "z": mid[:, 2], "cp": p / q}
    if not flow.inviscid:
        area = np.linalg.norm(S, axis=0)
        for c, name in enumerate("xyz"):
            out[f"cf_{name}"] = shear[c] / area / q
    return out


def pressure_lift(U: Array, atlas: Atlas, freestream: Array, gamma: float) -> Array:
    """Pressure lift coefficient per unit reference area, traceable, for
    freestream primitives ``(rho, V, p)``: a batched polar's per-block monitor.
    Lift is normal to the freestream in the plane of incidence, the first and
    the last axis (core/state.py)."""
    dim = atlas.dim
    rows = Rows(dim)
    p = conservative_to_primitive(U[:rows.n_flow], gamma, dim)[2].ravel()[atlas.walls.cell]
    force = bc.wall_pressure_force(p - freestream[rows.pressure], atlas)
    fx, fy = force[0], force[-1]
    u, v = freestream[1], freestream[dim]
    speed2 = u * u + v * v
    return (fy * u - fx * v) / jnp.sqrt(speed2) / (0.5 * freestream[RHO] * speed2)
