"""The swept (2.5D) operator: an infinite swept wing on a 2D grid.

Two exact checks, no approximation in either:

* at zero sweep the 2.5D residual is the 2D residual, with no spanwise
  momentum (the lift and the split of index axes from vector components);
* a 2D grid extruded ``nk`` cells in ``z`` with periodic end planes, carrying
  a spanwise-uniform state with a nonzero, varying spanwise velocity ``w``,
  is a 3D state whose residual is the 2.5D residual times the cell depth in
  every plane. The 3D operator is validated on its own (ONERA M6, the CRM),
  so this checks every spanwise term of the 2.5D one: the fluxes and the
  wave speeds, the conditions on ``w``, the viscous stresses and heating, the
  SA vorticity and QCR. The 3D atlas gives a ghost its partner's centre (right
  for an O seam, where the partner coincides); across a translation the
  periodic image is the partner's centre moved by the period, supplied here.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.core.config import FlowConditions, NumericsConfig
from jaxrans.core.state import freestream_primitive
from jaxrans.mesh.atlas import build_atlas, lift_atlas
from jaxrans.mesh.mesh import assign_conditions
from jaxrans.numerics.residual import make_residual
from tests.karman_trefftz import TMR, make_o_grid

from ..single import conditions, extrude

NK, DZ = 2, 0.7
PHYSICS = {
    "euler": FlowConditions(mach=0.6, alpha_deg=2.5),
    "laminar": FlowConditions(mach=0.5, alpha_deg=1.0, reynolds=5000.0),
    "sa": FlowConditions(mach=0.15, alpha_deg=10.0, reynolds=6e6, sutherland_ratio=198.6 / 540.0,
                         turbulence="sa_noft2"),
    "qcr": FlowConditions(mach=0.7, alpha_deg=2.0, reynolds=6e6, turbulence="sa_noft2_qcr"),
}


@pytest.fixture(scope="module")
def grid2():
    return make_o_grid(32, 44, 4e-4, section=TMR)


def perturbed(op, atlas, seed: int) -> np.ndarray:
    """The freestream on the block cells, each conservative entry perturbed by 3 %."""
    U0 = np.asarray(op.initial_state())
    rng = np.random.default_rng(seed)
    return np.where(atlas.interior, U0 * (1.0 + 0.03 * rng.standard_normal(U0.shape)), U0)


def periodic_images(atlas3, g: int):
    """``atlas3`` with the centres of its ``k`` ghosts (and the edge padding
    beyond them) at their periodic images: ``z = (k + 1/2) dz`` along the whole
    padded ``k`` axis of the single block."""
    centres = np.asarray(atlas3.centres).copy()
    k = np.arange(centres.shape[-1]) - (1 + g)
    centres[2] = np.broadcast_to((k + 0.5) * DZ, centres.shape[1:])
    return replace(atlas3, centres=jnp.asarray(centres))


@pytest.mark.parametrize("far_field", ["characteristic", "characteristic_point_vortex"])
@pytest.mark.parametrize("first_order", [True, False])
@pytest.mark.parametrize("physics", sorted(PHYSICS))
def test_unswept_2p5d_residual_is_the_2d_residual(grid2, physics, first_order, far_field) -> None:
    flow = PHYSICS[physics]
    numerics = NumericsConfig(limiter="van_albada", far_field=far_field)
    atlas = build_atlas(assign_conditions(grid2, conditions(grid2, no_slip=not flow.inviscid)), numerics.n_ghost)
    op2 = make_residual(atlas, flow, numerics, first_order)
    op25 = make_residual(atlas, replace(flow, sweep_deg=0.0), numerics, first_order)
    assert op25.components == 3 and op25.dim == 2

    U2 = perturbed(op2, atlas, 42)
    U25 = np.concatenate([U2[:3], np.zeros_like(U2[:1]), U2[3:]])
    R2 = np.asarray(op2.residual(jnp.asarray(U2)))
    R25 = np.asarray(op25.residual(jnp.asarray(U25)))
    scale = np.abs(R2).max(axis=(1, 2))
    rows = [0, 1, 2] + list(range(4, R25.shape[0]))
    worst = (np.abs(R25[rows] - R2).max(axis=(1, 2)) / scale).max()
    assert worst < 1e-12, f"relative difference {worst:.2e}"
    assert np.abs(R25[3]).max() == 0.0, "spanwise momentum appeared at zero sweep"


@pytest.mark.parametrize("first_order", [True, False])
@pytest.mark.parametrize("physics", sorted(PHYSICS))
def test_periodic_extrusion_residual_is_the_swept_residual(grid2, physics, first_order) -> None:
    flow = replace(PHYSICS[physics], sweep_deg=30.0)
    numerics = NumericsConfig(limiter="van_albada")
    kinds = conditions(grid2, no_slip=not flow.inviscid)
    atlas = build_atlas(assign_conditions(grid2, kinds), numerics.n_ghost)
    op25 = make_residual(atlas, flow, numerics, first_order)
    atlas3 = build_atlas(assign_conditions(extrude(grid2, NK, DZ, ends="periodic"), kinds), numerics.n_ghost)
    atlas3 = periodic_images(atlas3, numerics.n_ghost)
    op3 = make_residual(atlas3, replace(flow, sweep_deg=None), numerics, first_order,
                        freestream=op25.q_inf[:5])

    g = numerics.n_ghost
    U25 = perturbed(op25, atlas, 7)
    assert np.abs(U25[3][atlas.interior]).min() > 0.0
    block = (slice(None), slice(g, -g), slice(g, -g))
    U3 = np.asarray(op3.initial_state()).copy()
    U3[(slice(None),) + (slice(g, -g),) * 3] = np.repeat(U25[block][..., None], NK, axis=-1)

    R25 = np.asarray(op25.residual(jnp.asarray(U25)))[block]
    R3 = np.asarray(op3.residual(jnp.asarray(U3)))[(slice(None),) + (slice(g, -g),) * 3]
    scale = np.abs(R25).max(axis=(1, 2))[:, None, None]
    for k in range(NK):
        worst = float(np.max(np.abs(R3[..., k] / DZ - R25) / scale))
        assert worst < 1e-11, f"plane {k}: relative difference {worst:.2e}"


def test_the_lift_is_idempotent_and_spanwise_flat(grid2) -> None:
    atlas = build_atlas(assign_conditions(grid2, conditions(grid2)), 2)
    lifted = lift_atlas(atlas)
    assert lift_atlas(lifted) is lifted
    assert (lifted.dim, lifted.components, atlas.components) == (2, 3, 2)
    for f2, f3 in zip(atlas.grid.faces, lifted.grid.faces):
        assert np.array_equal(np.asarray(f3[:2]), np.asarray(f2)) and not np.asarray(f3[2]).any()
    for s2, s3 in zip(atlas.stages, lifted.stages):
        for (_, _, n2, m2), (_, _, n3, m3) in zip(s2.conditions, s3.conditions):
            assert np.array_equal(n3[:2], n2) and not n3[2].any() and np.array_equal(m3, m2)


def test_swept_freestream() -> None:
    flow = FlowConditions(mach=0.85, alpha_deg=2.0, sweep_deg=35.0)
    rho, u, v, w, p = freestream_primitive(flow, 3)
    a, s = np.deg2rad(2.0), np.deg2rad(35.0)
    assert np.allclose([u, v, w], 0.85 * np.array([np.cos(a) * np.cos(s), np.sin(a), np.cos(a) * np.sin(s)]),
                       rtol=0, atol=1e-15)
    assert (rho, p) == (1.0, 1.0 / 1.4)
    assert np.isclose(flow.in_plane_mach, np.hypot(u, v), rtol=1e-14)
    # Supersonic freestream, subsonic in the plane: allowed; not unswept.
    assert FlowConditions(mach=1.2, sweep_deg=40.0).in_plane_mach < 1.0
    with pytest.raises(ValueError):
        FlowConditions(mach=1.2)
    with pytest.raises(ValueError):
        FlowConditions(mach=0.5, sweep_deg=90.0)



# --- Forces and the case layer ------------------------------------------------------------


def test_unswept_2p5d_forces_are_the_2d_forces(grid2) -> None:
    from jaxrans.post.forces import force_coefficients
    flow = PHYSICS["sa"]
    numerics = NumericsConfig(limiter="van_albada")
    atlas = build_atlas(assign_conditions(grid2, conditions(grid2, no_slip=True)), numerics.n_ghost)
    op2 = make_residual(atlas, flow, numerics)
    U2 = perturbed(op2, atlas, 5)
    U25 = np.concatenate([U2[:3], np.zeros_like(U2[:1]), U2[3:]])
    c2 = force_coefficients(jnp.asarray(U2), atlas, flow)
    c25 = force_coefficients(jnp.asarray(U25), atlas, replace(flow, sweep_deg=0.0))
    for k in ("cl", "cd", "cm", "cd_friction"):
        assert abs(float(c25[k]) - float(c2[k])) < 1e-13 * max(1.0, abs(float(c2[k]))), k
    assert float(c25["cy"]) == 0.0


def test_swept_pressure_forces_are_the_streamwise_section_integral(grid2) -> None:
    """The pressure part of the wing-frame coefficients is the 2D integral round
    the streamwise section, the grid scaled back by 1 / cos(sweep), with the
    same wall pressures and the same freestream direction and speed. Pressure
    acts normal to the swept generators, so its spanwise part is
    ``F_S = -tan(sweep) F_X``."""
    from jaxrans.mesh.mesh import scale_x
    from jaxrans.post.forces import ForceReference, force_coefficients
    sweep, gamma = 35.0, 1.4
    c = np.cos(np.deg2rad(sweep))
    flow = FlowConditions(mach=0.8, alpha_deg=2.0, sweep_deg=sweep)
    kinds = conditions(grid2)
    normal = build_atlas(assign_conditions(scale_x(grid2, c), kinds), 2)
    streamwise = build_atlas(assign_conditions(grid2, kinds), 2)
    rng = np.random.default_rng(11)
    p = (1.0 / gamma) * (1.0 + 0.2 * rng.standard_normal(normal.shape))
    rest = np.zeros((2,) + normal.shape)
    U25 = np.concatenate([np.ones((1,) + normal.shape), np.zeros((3,) + normal.shape), p[None] / (gamma - 1)])
    U2 = np.concatenate([np.ones((1,) + normal.shape), rest, p[None] / (gamma - 1)])
    reference = ForceReference(1.0, None, (0.3, 0.05, 0.0))
    swept = force_coefficients(jnp.asarray(U25), normal, flow, reference)
    plain = force_coefficients(jnp.asarray(U2), streamwise, replace(flow, sweep_deg=None), reference)
    for k in ("cl", "cd", "cm"):
        assert abs(float(swept[k]) - float(plain[k])) < 1e-12, k
    q_area = 0.5 * 0.8**2
    expected = -np.tan(np.deg2rad(sweep)) * float(plain["force"][0]) / q_area
    assert abs(float(swept["cy"]) - expected) < 1e-12 and abs(expected) > 1e-3


def test_a_swept_case_meshes_the_normal_section(tmp_path) -> None:
    from jaxrans import CaseError
    from jaxrans.case import parse_case
    case_doc = {"general": {"case_name": "swept"},
                "domain": {"grid": "o_grid", "airfoil": "naca0012", "cells_airfoil": 32, "cells_normal": 16,
                           "wall_spacing": 1e-3, "far_field": 20.0},
                "freestream": {"mach": 0.7, "alpha_deg": 1.0, "sweep_deg": 30.0}}
    case = parse_case(case_doc, {}, tmp_path)
    atlas = case.grid()
    assert case.flow.swept and atlas.components == 3
    x = np.asarray(atlas.mesh[0].nodes[0][:, 0])
    assert np.isclose(x.max() - x.min(), np.cos(np.deg2rad(30.0)), rtol=1e-3)
    given = parse_case({**case_doc, "domain": {**case_doc["domain"], "section": "normal"}}, {}, tmp_path).grid()
    xg = np.asarray(given.mesh[0].nodes[0][:, 0])
    assert np.allclose(x, xg * np.cos(np.deg2rad(30.0)), rtol=0, atol=1e-14)
    with pytest.raises(CaseError, match="section"):
        parse_case({**case_doc, "domain": {**case_doc["domain"], "section": "normal"},
                    "freestream": {"mach": 0.7}}, {}, tmp_path)
