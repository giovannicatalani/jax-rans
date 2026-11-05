"""Wall pressure and friction, and the integrated force coefficients.

Nondimensional as everywhere (core/constants.py): ``p_inf = 1/gamma`` and
``q_inf = M^2 / 2``. Pressure comes from the cell on each wall face, friction
from the viscous flux the residual uses there; the force on the body is

    F = sum_faces [ -(p - p_inf) S + tau . S ],     S into the fluid,

gauge pressure, so an open wall (a plate wetted on one side) is right too.

Swept (2.5D, ``FlowConditions.sweep_deg``): the grid is the section normal to
the sweep line, the loads per unit length along it. In the wing frame (``X``
streamwise, ``Y`` up, ``S`` the span), per unit span, with ``c = cos(sweep)``
and ``s = sin(sweep)``,

    F_X = F_x + (s / c) F_z,    F_Y = F_y / c,    F_S = (c F_z - s F_x) / c

and the coefficients are the 2D ones of the streamwise section, at streamwise
positions ``x / c``: lift and drag in the plane ``X-Y``, per ``q_inf`` (of the
whole freestream) times the reference length, the moment about the centre in
streamwise coordinates, ``cy`` the spanwise force. The pressure part is the
integral round the streamwise section: a face ``(dx, dy)`` of the normal one is
``(dx / c, dy)`` there, area vector ``(S_x, S_y / c)``.
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
from ..mesh.sectional import SectionalAtlas
from ..numerics import boundary as bc
from ..numerics.residual import primitive_from_conservative, swept_atlas, viscous_fluxes
from ..physics.navier_stokes import Viscosity


def dynamic_pressure(flow: FlowConditions) -> float:
    return 0.5 * flow.mach**2


def wall_loads(U: Array, atlas: Atlas, flow: FlowConditions, freestream: Array | None = None,
               mu_inf: Array | None = None) -> tuple[Array, Array, Array]:
    """Per wall face: gauge pressure ``(n,)``, the area vector into the fluid
    ``(d, n)``, and the viscous force on the body ``(d, n)`` (zero if
    inviscid); ``freestream`` and ``mu_inf`` (traceable) replace ``flow``'s.
    Vectors have the atlas's components (3 swept, in the grid's frame)."""
    if isinstance(atlas, SectionalAtlas):
        if atlas.geometry is not None:
            q = primitive_from_conservative(U[:5], flow.gamma, 3)
            p = q[4].ravel()[atlas.walls.cell] - 1.0 / flow.gamma
            return p, bc.wall_vectors(atlas) / atlas.depth, jnp.zeros((3, p.shape[0]), dtype=U.dtype)
        return wall_loads(atlas.section_values(U), atlas.reference, flow, freestream, mu_inf)
    atlas = swept_atlas(atlas, flow)
    gamma, dim, nv = flow.gamma, atlas.dim, atlas.components
    rows = Rows(nv)
    q = primitive_from_conservative(U[:rows.n_flow], gamma, nv)
    p = q[rows.pressure].ravel()[atlas.walls.cell] - 1.0 / gamma
    S = bc.wall_vectors(atlas)
    shear = jnp.zeros((nv,) + p.shape, dtype=U.dtype)
    if not flow.inviscid:
        q_inf = jnp.asarray(freestream_primitive(flow, nv) if freestream is None else freestream,
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
    friction in 3D (as TMR's CMY). Swept: those of the streamwise section in
    the wing frame (module docstring), ``cy`` and ``force[2]`` spanwise."""
    if isinstance(atlas, SectionalAtlas):
        if atlas.geometry is not None:
            q = jnp.asarray(freestream_primitive(flow, 3) if freestream is None else freestream, dtype=U.dtype)
            return _conical_coefficients(U, atlas, q, flow.gamma, reference)
        return force_coefficients(atlas.section_values(U), atlas.reference, flow, reference,
                                  freestream, mu_inf)
    atlas = swept_atlas(atlas, flow)
    dim, nv = atlas.dim, atlas.components
    if dim == 3 and reference.area is None:
        raise ValueError("a 3D case needs domain.reference_area")
    q = jnp.asarray(freestream_primitive(flow, nv) if freestream is None else freestream, dtype=U.dtype)
    p, S, shear = wall_loads(U, atlas, flow, q, mu_inf)
    faces = atlas.walls.faces
    rho, velocity, _ = split(q, nv)
    speed = magnitude(velocity)
    pressure = -p * S                                                   # (d, n)
    load = pressure + shear
    mid, side = faces.mid, None
    if flow.swept:
        c, s = float(np.cos(flow.sweep_rad)), float(np.sin(flow.sweep_rad))

        def wing(f):
            """Per unit length along the sweep line -> the wing frame per unit span, ``(X, Y)``."""
            return jnp.stack([f[0] + (s / c) * f[2], f[1] / c])

        side = (c * load[2] - s * load[0]) / c
        load, pressure, shear = wing(load), wing(pressure), wing(shear)
        velocity = jnp.stack([c * velocity[0] + s * velocity[2], velocity[1]])
        mid = mid / np.array([c, 1.0])
    drag = velocity / speed
    lift = jnp.stack([-drag[1], drag[0]] if dim == 2 else [-drag[2], jnp.zeros_like(speed), drag[0]])
    q_area = 0.5 * rho * speed**2 * (reference.length if dim == 2 else reference.area)
    arm = pressure if dim == 2 else load
    r = jnp.asarray(mid - np.asarray(reference.centre[:dim]))
    moment = r[:, -1] * arm[0] - r[:, 0] * arm[-1]
    per_face = jnp.stack([lift @ load, drag @ load, moment / reference.length]) / q_area   # (3, n)
    totals, force = per_face.sum(axis=1), load.sum(axis=1)
    if side is not None:
        force = jnp.concatenate([force, side.sum()[None]])
    return {"cl": totals[0], "cd": totals[1], "cm": totals[2],
            "cd_friction": drag @ shear.sum(axis=1) / q_area,
            "cy": force[-1] / q_area if side is not None else force[1] / q_area if dim == 3
            else jnp.zeros_like(q_area),
            "force": force,
            "bodies": jax.ops.segment_sum(per_face.T, jnp.asarray(faces.body), faces.bodies)}


def _conical_coefficients(U: Array, atlas: SectionalAtlas, q: Array, gamma: float,
                          reference: ForceReference) -> dict[str, Array]:
    """Euler pressure loads at the streamwise station, per unit geometric span."""
    mid, S = atlas.geometry.surface_vectors(atlas.reference.walls.faces)
    p = conservative_to_primitive(U[:5], gamma, 3)[2].ravel()[atlas.walls.cell] - q[4]
    loads = -p * jnp.asarray(S)
    c, s = np.cos(atlas.geometry.angle), np.sin(atlas.geometry.angle)
    velocity = jnp.stack([c * q[1] + s * q[3], q[2]])
    speed = magnitude(q[1:4])
    drag = velocity / speed
    lift = jnp.stack([-drag[1], drag[0]])
    q_area = 0.5 * q[0] * speed**2 * reference.length
    arm = jnp.asarray(mid) - jnp.asarray(reference.centre[:2])[:, None]
    moment = arm[1] * loads[0] - arm[0] * loads[1]
    per_face = jnp.stack([lift @ loads[:2], drag @ loads[:2], moment / reference.length]) / q_area
    total, force = per_face.sum(axis=1), loads.sum(axis=1)
    return {"cl": total[0], "cd": total[1], "cm": total[2], "cy": force[2] / q_area,
            "cd_friction": jnp.zeros_like(q_area), "force": force,
            "bodies": jax.ops.segment_sum(per_face.T, jnp.asarray(atlas.reference.walls.faces.body),
                                          atlas.reference.walls.faces.bodies)}


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
    negative where separated). Swept: ``x`` in the grid's (normal) section,
    plus ``x_streamwise`` (``x / cos(sweep)``) and, viscous, ``cf_z``, the wall
    shear along the sweep line; ``cp`` and ``cf`` over the whole ``q_inf``."""
    if isinstance(atlas, SectionalAtlas):
        if atlas.geometry is not None:
            faces = atlas.reference.walls.faces
            mid, _ = atlas.geometry.surface_vectors(faces)
            p = np.asarray(conservative_to_primitive(U[:5], flow.gamma, 3)[2]).ravel()[atlas.walls.cell]
            return {"x": faces.mid[:, 0], "y": faces.mid[:, 1], "body": faces.body.astype(float),
                    "x_streamwise": mid[0], "y_streamwise": mid[1],
                    "cp": (p - 1.0 / flow.gamma) / dynamic_pressure(flow)}
        return surface(atlas.section_values(U), atlas.reference, flow)
    if atlas.dim == 3:
        return _surface_3d(U, atlas, flow)
    p, S, shear = (np.asarray(a) for a in wall_loads(U, atlas, flow))
    sx, sy = S[0], S[1]
    faces = atlas.walls.faces
    out = {"x": faces.mid[:, 0], "y": faces.mid[:, 1], "body": faces.body.astype(float),
           "cp": p / dynamic_pressure(flow)}
    if flow.swept:
        out = {"x": out["x"], "x_streamwise": out["x"] / np.cos(flow.sweep_rad),
               **{k: out[k] for k in ("y", "body", "cp")}}
    if not flow.inviscid:
        length = np.hypot(sx, sy)
        tx, ty = sy / length, -sx / length
        sign = np.where(tx >= 0.0, 1.0, -1.0)
        out["cf"] = sign * (shear[0] * tx + shear[1] * ty) / length / dynamic_pressure(flow)
        if flow.swept:
            out["cf_z"] = shear[2] / length / dynamic_pressure(flow)
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
    the last axis (core/state.py); swept, of the in-plane flow."""
    if isinstance(atlas, SectionalAtlas):
        if atlas.geometry is not None:
            return _conical_coefficients(U, atlas, freestream, gamma, ForceReference())["cl"]
        return pressure_lift(atlas.section_values(U), atlas.reference, freestream, gamma)
    dim = atlas.dim
    rows = Rows(atlas.components)
    p = conservative_to_primitive(U[:rows.n_flow], gamma, atlas.components)[2].ravel()[atlas.walls.cell]
    force = bc.wall_pressure_force(p - freestream[rows.pressure], atlas)
    fx, fy = force[0], force[dim - 1]
    u, v = freestream[1], freestream[dim]
    speed2 = u * u + v * v
    return (fy * u - fx * v) / jnp.sqrt(speed2) / (0.5 * freestream[RHO] * speed2)
