"""Stage A: a section's genuine 3D stencil contains the existing 2.5D one.

Perturbed states exercise lateral momentum, stresses, viscous work, SA and
QCR; poisoned halo values check that the closure has no extra unknowns.
The comparison is per unit extrusion depth, not between converged lift alone.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.core.config import NumericsConfig
from jaxrans.core.sectional import SectionalSpec
from jaxrans.mesh.atlas import build_atlas
from jaxrans.mesh.block import normal_closure_residual
from jaxrans.mesh.mesh import assign_conditions
from jaxrans.mesh.sectional import build_sectional_atlas
from jaxrans.numerics.residual import make_residual
from jaxrans.post.forces import force_coefficients, pressure_lift, surface
from tests.karman_trefftz import TMR, make_o_grid
from tests.single import conditions
from tests.verification.test_multiblock import split
from tests.verification.test_swept import PHYSICS, perturbed


@pytest.fixture(scope="module")
def grid2():
    return make_o_grid(16, 32, 1e-3, section=TMR)


def operators(grid, physics="sa", sweep=30.0, first_order=False, far_field="characteristic", depth=0.7,
              flux="hllc", g=2):
    flow = replace(PHYSICS[physics], sweep_deg=sweep)
    numerics = NumericsConfig(limiter="van_albada", far_field=far_field, flux=flux, n_ghost=g)
    blocks = grid if isinstance(grid, tuple) else (grid,)
    kinds = {name: kind for block in blocks for name, kind in
             conditions(block, no_slip=not flow.inviscid).items()}
    mesh = assign_conditions(grid, kinds)
    atlas = build_sectional_atlas(mesh, SectionalSpec(leading_edge_sweep_deg=sweep), g, depth)
    plain = make_residual(atlas.reference, flow, numerics, first_order)
    section = make_residual(atlas, flow, numerics, first_order)
    return plain, section, flow


def assert_close(actual, expected, mask, tolerance=1e-11):
    actual, expected = np.asarray(actual)[:, mask], np.asarray(expected)[:, mask]
    scale = np.maximum(np.abs(expected).max(axis=1), 1e-25)
    error = np.abs(actual - expected).max(axis=1) / scale
    assert error.max() < tolerance, error


@pytest.mark.parametrize("physics", sorted(PHYSICS))
@pytest.mark.parametrize("first_order", [False, True])
@pytest.mark.parametrize("far_field", ["characteristic", "characteristic_point_vortex"])
@pytest.mark.parametrize("sweep", [0.0, 30.0])
def test_sectional_residual_is_the_swept_residual(grid2, physics, first_order, far_field, sweep):
    plain, section, _ = operators(grid2, physics, sweep, first_order, far_field)
    U = jnp.asarray(perturbed(plain, plain.atlas, 7))
    # Lateral halos are not independent unknowns, even if the caller poisons them.
    expanded = section.atlas.expand(U)
    expanded = jnp.where(section.atlas.interior[None], expanded, expanded * 1.13)
    R = section.atlas.section_values(section.residual(expanded)) / section.atlas.depth
    assert section.dim == 3 and section.components == 3
    assert section.atlas.cells == plain.atlas.cells
    assert all(b.shape[-1] == 1 for b in section.atlas.mesh)
    assert_close(R, plain.residual(U), plain.atlas.interior)
    if section.model is not None:
        np.testing.assert_array_equal(section.atlas.section_values(section.distance), plain.distance)


@pytest.mark.parametrize("physics", sorted(PHYSICS))
def test_zero_sweep_sectional_residual_is_the_native_2d_residual(grid2, physics):
    _, section, flow = operators(grid2, physics, sweep=0.0)
    atlas2 = build_atlas(assign_conditions(grid2, conditions(grid2, no_slip=not flow.inviscid)))
    plain = make_residual(atlas2, replace(flow, sweep_deg=None), NumericsConfig(limiter="van_albada"))
    U = perturbed(plain, atlas2, 42)
    lifted = jnp.asarray(np.concatenate([U[:3], np.zeros_like(U[:1]), U[3:]]))
    R = section.atlas.section_values(section.residual(section.atlas.expand(lifted))) / section.atlas.depth
    rows = [0, 1, 2] + list(range(4, R.shape[0]))
    assert_close(R[np.asarray(rows)], plain.residual(jnp.asarray(U)), atlas2.interior)
    assert np.abs(np.asarray(R[3])[atlas2.interior]).max() < 1e-13


@pytest.mark.parametrize("depth", [0.03, 2.0])
@pytest.mark.parametrize("flux", ["hllc", "roe"])
def test_depth_and_riemann_solver_do_not_change_the_limit(grid2, depth, flux):
    plain, section, _ = operators(grid2, sweep=-25.0, depth=depth, flux=flux, g=3)
    U = jnp.asarray(perturbed(plain, plain.atlas, 4))
    R = section.atlas.section_values(section.residual(section.atlas.expand(U))) / depth
    assert_close(R, plain.residual(U), plain.atlas.interior)


def test_split_blocks_and_their_ghost_corners_preserve_the_sectional_limit(grid2):
    mesh, _ = split(grid2, (3, 8, 13), (3, 11))
    plain, section, _ = operators(mesh, physics="qcr")
    U = jnp.asarray(perturbed(plain, plain.atlas, 5))
    R = section.atlas.section_values(section.residual(section.atlas.expand(U))) / section.atlas.depth
    assert_close(R, plain.residual(U), plain.atlas.interior)


def test_jacobian_includes_the_sectional_ghost_closure(grid2):
    plain, section, _ = operators(grid2, physics="sa", first_order=True)
    U = jnp.asarray(perturbed(plain, plain.atlas, 5))
    v = U * jnp.asarray(np.random.default_rng(2).normal(size=U.shape)) * 0.01
    _, expected = jax.jvp(plain.residual, (U,), (v,))
    _, actual = jax.jvp(section.residual, (section.atlas.expand(U),), (section.atlas.expand(v),))
    assert_close(section.atlas.section_values(actual) / section.atlas.depth,
                 expected, plain.atlas.interior)


@pytest.mark.parametrize("physics", ["euler", "sa"])
def test_sectional_forces_and_surface_keep_the_2p5d_conventions(grid2, physics):
    plain, section, flow = operators(grid2, physics)
    U = jnp.asarray(perturbed(plain, plain.atlas, 6))
    U3 = section.atlas.expand(U)
    expected = force_coefficients(U, plain.atlas, flow)
    actual = force_coefficients(U3, section.atlas, flow)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-13, atol=1e-13)
    expected_surface, actual_surface = surface(U, plain.atlas, flow), surface(U3, section.atlas, flow)
    for key in expected_surface:
        np.testing.assert_array_equal(actual_surface[key], expected_surface[key])
    np.testing.assert_allclose(pressure_lift(U3, section.atlas, section.q_inf, flow.gamma),
                               pressure_lift(U, plain.atlas, plain.q_inf, flow.gamma), rtol=1e-13)


def test_the_sectional_metrics_preserve_uniform_flow_and_ghost_images(grid2):
    mesh = assign_conditions(grid2, {"wall": "far_field", "farfield": "far_field"})
    atlas = build_sectional_atlas(mesh, SectionalSpec(leading_edge_sweep_deg=30.0), depth=0.2)
    flow = replace(PHYSICS["euler"], sweep_deg=30.0)
    op = make_residual(atlas, flow, NumericsConfig())
    R = np.asarray(op.residual(op.initial_state()))[:, atlas.interior]
    scale = max(float(jnp.abs(s).max()) for s in atlas.grid.faces)
    assert np.abs(R).max() < 1e-12 * scale
    assert np.abs(normal_closure_residual(atlas.mesh[0])).max() < 1e-12 * scale
    dz = np.diff(np.asarray(atlas.centres)[2], axis=-1)
    np.testing.assert_allclose(dz, 0.2, rtol=1e-14, atol=1e-14)
