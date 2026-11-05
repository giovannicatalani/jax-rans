"""Conical Navier-Stokes/SA: 3D reference, walls and vanishing taper."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.core.config import FlowConditions, NumericsConfig
from jaxrans.core.sectional import SectionalSpec
from jaxrans.geometry.conical import distance_to_section, relaxation_weight
from jaxrans.mesh.atlas import build_atlas
from jaxrans.mesh.block import Grid
from jaxrans.mesh.conical_wall import conical_wall_distance
from jaxrans.mesh.mesh import assign_conditions
from jaxrans.mesh.sectional import build_sectional_atlas
from jaxrans.numerics import boundary as bc
from jaxrans.numerics.turbulence import face_eddy_viscosity
from jaxrans.numerics.residual import make_residual, primitive_from_conservative, viscous_fluxes
from jaxrans.post.forces import force_coefficients, surface
from tests.karman_trefftz import TMR, make_o_grid
from tests.single import extruded_sides


@pytest.fixture(scope="module")
def section():
    return assign_conditions(make_o_grid(16, 24, 1e-3, section=TMR),
                             {"wall": "no_slip_wall", "farfield": "far_field"})


def operator(section, model, taper=.2):
    spec = SectionalSpec(leading_edge_sweep_deg=30., taper_rate=taper)
    atlas = build_sectional_atlas(section, spec, depth=.1)
    flow = FlowConditions(mach=.5, alpha_deg=2., reynolds=1e5, turbulence=model,
                          sweep_deg=spec.reference_sweep_deg, sutherland_ratio=.4)
    numerics = NumericsConfig(limiter="none")
    return make_residual(atlas, flow, numerics), flow, numerics


@pytest.mark.parametrize("model", [None, "sa_noft2", "sa_noft2_qcr"])
def test_viscous_section_matches_the_middle_of_a_five_plane_3d_block(section, model):
    op, flow, numerics = operator(section, model)
    atlas, geom = op.atlas, op.atlas.geometry
    xy = np.asarray(section[0].nodes)
    weight = relaxation_weight(distance_to_section(xy, atlas.reference.walls.faces) / geom.chord, 1, 5)
    nk = 5
    shape = xy.shape[1:] + (nk + 1,)
    points = np.concatenate([np.broadcast_to(xy[..., None], (2,) + shape),
                             np.broadcast_to((np.arange(nk + 1) - nk / 2) * .1, shape)[None]])
    grid = Grid.from_node_array(geom.map_points(points, weight[..., None]),
                               extruded_sides(section[0].sides, nk, "farfield"))
    full = build_atlas(assign_conditions(grid, {"farfield": "far_field", "wall": "no_slip_wall"}))
    distance = full.pack([jnp.asarray(conical_wall_distance(
        np.moveaxis(np.asarray(grid.centres), 0, -1), atlas.reference.walls.faces, geom))], 1.) if model else None
    reference = make_residual(full, replace(flow, sweep_deg=None), numerics,
                              freestream=op.q_inf[:5], distance=distance)
    U = op.initial_state() * jnp.asarray(1 + .015 * np.random.default_rng(5).normal(size=op.initial_state().shape))
    block = atlas.unpack(U)[0]
    V = full.pack([jnp.repeat(block, nk, axis=-1)], reference.initial_state()[:, 0, 0, 0])
    actual = np.asarray(atlas.unpack(op.residual(U))[0])[..., 0]
    expected = np.asarray(full.unpack(reference.residual(V))[0])[..., nk // 2]
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-10)


@pytest.mark.parametrize("model", [None, "sa_noft2", "sa_noft2_qcr"])
def test_viscous_residual_and_outputs_have_the_2p5d_limit(section, model):
    base, flow, _ = operator(section, model, 0.)
    U = base.initial_state() * jnp.asarray(1 + .02 * np.random.default_rng(8).normal(size=base.initial_state().shape))
    expected = np.asarray(base.residual(U) / base.grid.volume)[:, base.atlas.interior]
    loads = force_coefficients(U, base.atlas, flow)
    fields = surface(U, base.atlas, flow)
    errors, force_errors, friction_errors = [], [], []
    for taper in (1e-3, 1e-4, 1e-5):
        op, flow, _ = operator(section, model, taper)
        actual = np.asarray(op.residual(U) / op.grid.volume)[:, op.atlas.interior]
        errors.append(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        got = force_coefficients(U, op.atlas, flow)
        force_errors.append(max(abs(float(got[k] - loads[k])) for k in ("cl", "cd", "cy", "cm", "cd_friction")))
        fields_got = surface(U, op.atlas, flow)
        friction_errors.append(max(np.max(abs(fields_got[k] - fields[k])) for k in ("cf", "cf_z")))
    for series in (errors, force_errors, friction_errors):
        assert all(a / b > 8 for a, b in zip(series, series[1:])), series
    assert errors[-1] < 1e-3


def test_conical_wall_is_adiabatic_and_the_viscous_closure_is_differentiable(section):
    op, flow, _ = operator(section, "sa_noft2_qcr")
    U = op.initial_state() * jnp.asarray(1 + .02 * np.random.default_rng(9).normal(size=op.initial_state().shape))
    q = primitive_from_conservative(U, flow.gamma, 3)
    conditions = bc.Conditions(op.q_inf, flow.gamma, bc.uniform_far_field(q, op.atlas, op.q_inf, flow.gamma))
    filled = jnp.pad(bc.fill_ghosts(q, op.atlas, conditions), ((0, 0),) + ((1, 1),) * 3, mode="edge")
    eddy = face_eddy_viscosity(filled, op.viscosity, flow.gamma, op.model, 3)
    assert np.all(np.asarray(op.atlas.wall_values(eddy)) == 0)
    flux = viscous_fluxes(filled[:5], op.grid, op.viscosity, flow.gamma, centres=op.atlas.centres)
    # At a stationary no-slip wall the energy flux is solely conduction.
    energy = np.asarray(op.atlas.wall_values(tuple(f[4] for f in flux)))
    assert np.max(abs(energy)) < 1e-12
    _, tangent = jax.jvp(op.residual, (U,), (jnp.full_like(U, .001),))
    assert np.isfinite(np.asarray(tangent)[:, op.atlas.interior]).all()


def test_splitting_preserves_all_sa_qcr_residuals(section):
    from tests.verification.test_multiblock import split as split_grid
    mesh, windows = split_grid(section[0], (4, 10), (5, 13))
    single, _, _ = operator(section, "sa_noft2_qcr")
    multi, _, _ = operator(mesh, "sa_noft2_qcr")
    U = single.initial_state() * jnp.asarray(1 + .01 * np.random.default_rng(4).normal(size=single.initial_state().shape))
    block = single.atlas.unpack(U)[0]
    V = multi.atlas.pack([block[:, i, j] for i, j in windows], multi.initial_state()[:, 0, 0, 0])
    expected = single.atlas.unpack(single.residual(U))[0]
    for actual, (i, j) in zip(multi.atlas.unpack(multi.residual(V)), windows):
        np.testing.assert_allclose(actual, expected[:, i, j], atol=2e-12, rtol=2e-10)


def test_viscous_depth_refinement_and_friction_independence(section, record_property):
    base, flow, numerics = operator(section, "sa_noft2_qcr")
    U = base.initial_state() * jnp.asarray(1 + .02 * np.random.default_rng(6).normal(size=base.initial_state().shape))
    residuals, friction = [], []
    for depth in (.2, .1, .05, .025):
        atlas = build_sectional_atlas(section, base.atlas.spec, depth=depth)
        op = make_residual(atlas, flow, numerics)
        residuals.append(np.asarray(op.residual(U) / op.grid.volume)[:, atlas.interior])
        friction.append(float(force_coefficients(U, atlas, flow)["cd_friction"]))
    errors = [np.linalg.norm(r - residuals[-1]) for r in residuals[:-1]]
    assert errors[0] / errors[1] > 3 and errors[1] / errors[2] > 3
    np.testing.assert_allclose(friction, friction[0], rtol=1e-11)
    record_property("depth_errors", errors)
