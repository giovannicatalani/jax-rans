"""Per-case precision, geometric robustness, and actual FP32 CFD stepping."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.case import CaseError, parse_case
from jaxrans.core.precision import floor_like, l2_norm, weighted_rms
from jaxrans.mesh.atlas import build_atlas
from jaxrans.mesh.block import Grid
from jaxrans.mesh.block3d import SIDES_3D, box_sides
from jaxrans.mesh.mesh import assign_conditions
from jaxrans.numerics import boundary as bc
from jaxrans.numerics.gradients import face_gradients
from jaxrans.numerics.residual import make_residual
from jaxrans.numerics.riemann import FLUXES
from jaxrans.simulation import make_stepper
from jaxrans.solvers.krylov import gmres_right
from jaxrans.solvers import preconditioners as pre


def case(precision="fp64", solver="implicit", model="sa_noft2", swept=True):
    doc = {"general": {"case_name": "precision", "precision": precision},
           "domain": {"grid": "o_grid", "airfoil": "naca0012", "cells_airfoil": 32,
                      "cells_normal": 12, "wall_spacing": 0.001, "far_field": 10.},
           "freestream": {"mach": .3, "alpha_deg": 2., "reynolds": 5000.},
           "turbulence": {"model": model}}
    if swept:
        doc["freestream"]["sweep_deg"] = 12.
    numerical = {"convective_fluxes": {"limiter": "none"},
                 "time_integration": {"solver": solver, "cfl": .1},
                 "implicit": {"krylov_subspace": 4, "krylov_restarts": 1, "max_relative_update": .2},
                 "convergence": {"max_iters": 3, "block_iters": 2, "forces_tol": None}}
    return parse_case(doc, numerical)


@pytest.mark.parametrize("precision,dtype", [("fp64", "float64"), ("float64", "float64"),
                                           ("fp32", "float32"), ("float32", "float32")])
def test_precision_selects_fields_without_changing_geometry_or_tolerances(precision, dtype):
    c = case(precision)
    grid = c.grid()
    assert c.numerics.dtype == np.dtype(dtype)
    assert grid.grid.volume.dtype == dtype
    assert all(f.dtype == dtype for f in grid.grid.faces)
    assert grid.centres.dtype == np.float64
    assert all(b.nodes.dtype == np.float64 for b in grid.mesh)
    assert c.numerics.tol == 1e-12
    assert c.implicit == case().implicit
    assert jax.config.jax_enable_x64
    # Metric conversion happens after mesh generation; no grid is regenerated
    # from already-rounded coordinates.
    reference = case().grid()
    np.testing.assert_array_equal(grid.grid.volume, np.asarray(reference.grid.volume).astype(dtype))


def test_missing_precision_retains_fp64_and_invalid_values_are_rejected():
    doc = case().raw["case"]
    del doc["general"]["precision"]
    assert parse_case(doc, {}).numerics.dtype == np.float64
    for value in ("fp16", "FP32", 32, None):
        doc["general"]["precision"] = value
        with pytest.raises(CaseError, match="general.precision"):
            parse_case(doc, {})




@pytest.mark.parametrize("magnitude", [0., 1e-25, 1., 1e25])
def test_fp32_norms_resist_overflow_and_underflow(magnitude):
    a = jnp.asarray([magnitude, -2 * magnitude, 3 * magnitude], dtype=jnp.float32)
    weights = jnp.asarray([1., 2., 3.], dtype=jnp.float32)
    assert floor_like(a) > 0
    norm, rms = jax.jit(lambda x: (l2_norm(x), weighted_rms(x, weights)))(a)
    assert norm.dtype == rms.dtype == np.float32
    np.testing.assert_allclose(norm, np.linalg.norm(np.asarray(a, dtype=np.float64)), rtol=2e-7)
    expected = np.sqrt(np.average(np.asarray(a, dtype=np.float64)**2, weights=np.asarray(weights)))
    np.testing.assert_allclose(rms, expected, rtol=2e-7)


@pytest.mark.parametrize("name", sorted(FLUXES))
def test_fp32_zero_area_face_has_zero_flux_and_finite_state_jvp(name):
    q = jnp.asarray([1., .3, .1, 1/1.4], dtype=jnp.float32)[:, None]
    s = jnp.zeros((2, 1), dtype=jnp.float32)
    flux, tangent = jax.jit(lambda q: jax.jvp(lambda x: FLUXES[name](x, q, s, 1.4),
                                           (q,), (jnp.ones_like(q),)))(q)
    np.testing.assert_array_equal(flux, 0.)
    assert np.isfinite(tangent).all()


@pytest.mark.parametrize("dim", [2, 3])
def test_fp32_gradients_keep_thin_cell_geometry_in_double(dim):
    # At x=1 a spacing of 1e-9 disappears if positions are converted to FP32.
    axes = [1. + np.arange(6) * 1e-9] + [np.arange(6) * .1 for _ in range(dim - 1)]
    centres = jnp.asarray(np.stack(np.meshgrid(*axes, indexing="ij")))
    phi = (centres[0] - 1.).astype(jnp.float32)[None]
    gradients = jax.jit(face_gradients)(phi, centres)
    for gradient in gradients:
        assert gradient.dtype == np.float32
        np.testing.assert_allclose(gradient[0], 1., atol=1e-6)
        np.testing.assert_allclose(gradient[1:], 0., atol=1e-6)


def test_fp32_gmres_reduces_the_true_residual_with_fp32_vectors():
    rng = np.random.default_rng(4)
    A = jnp.asarray(np.eye(12)*6 + rng.normal(size=(12, 12)), dtype=jnp.float32)
    b = jnp.asarray(rng.normal(size=12), dtype=jnp.float32)
    solve = jax.jit(lambda b: gmres_right(lambda v: A @ v, b, lambda r: r,
                                         restart=12, maxiter=2, tol=1e-6))
    x = solve(b)
    assert x.dtype == np.float32
    assert np.linalg.norm(np.asarray(A) @ np.asarray(x) - b) < 2e-6 * np.linalg.norm(b)
    np.testing.assert_array_equal(solve(jnp.zeros_like(b)), 0.)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_point_vortex_preserves_field_dtype_and_double_geometry_differences(precision):
    """A translated double-precision mesh gives the same lifting reference.

    Rounding either coordinate before subtracting loses local offsets at a
    large origin. Keeping offsets in double during the field algebra instead
    promotes the reference and subsequently the residual's Krylov vectors.
    """
    from dataclasses import replace

    c = case(precision, model="none", swept=False)
    atlas = c.grid()
    op = make_residual(atlas, c.flow, c.numerics)
    q_inf = op.q_inf
    q = jnp.broadcast_to(q_inf[:, None, None], (4, *atlas.shape))
    q = q.at[3].add(-.05 * jnp.sin(atlas.grid.centres[1]))
    vectors = bc.wall_vectors(atlas)
    assert vectors.dtype == q.dtype
    np.testing.assert_array_equal(vectors, np.asarray(atlas.walls.into_fluid)
                                  * np.asarray(atlas.wall_values(atlas.grid.faces)))

    def reference(q, centre, midpoint):
        geometry = replace(atlas, quarter_chord=centre)
        return bc.point_vortex_far_field(q, geometry, q_inf, op.gamma)(midpoint)

    reference = jax.jit(reference)
    centre = jnp.asarray([.25, .125], jnp.float64)
    midpoint = centre[:, None] + jnp.asarray([[10., 20., 0.], [20., -10., 15.]], jnp.float64)
    shift = jnp.asarray([1e8, -1e8], jnp.float64)
    near = reference(q, centre, midpoint)
    translated = reference(q, centre + shift, midpoint + shift[:, None])
    assert near.dtype == translated.dtype == q.dtype
    assert np.isfinite(translated).all()
    assert np.max(np.abs(np.asarray(near) - np.asarray(q_inf)[:, None])) > 1e-5
    tolerance = 5e-7 if precision == "fp32" else 5e-14
    np.testing.assert_allclose(translated, near, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("swept", [False, True])
def test_line_preconditioner_uses_field_precision_with_double_geometry(precision, swept):
    """FP64 source metrics and wall distances must not promote Krylov vectors.

    Check the actual chain equations as well as dtype: casting only the final
    preconditioned vector would leave the expensive factors in double.
    """
    c = case(precision, swept=swept)
    atlas = case("fp64", swept=swept).grid()
    op = make_residual(atlas, c.flow, c.numerics)
    U = op.initial_state()
    U = U * jnp.asarray(1. + .05 * np.sin(np.arange(U.shape[-2]))[:, None], U.dtype)
    rhs = jnp.asarray(np.random.default_rng(27).normal(size=U.shape), U.dtype)
    distance = op.distance.astype(jnp.float64)
    layout = pre.chain_layout(atlas)
    n_flow = atlas.components + 2

    def apply(u, r):
        lambdas = op.wave_speeds(u)
        cfl = jnp.asarray(.1, u.dtype)
        flow = pre.flow_chain_blocks(u[:n_flow], lambdas, cfl, atlas, op.gamma, layout)
        scalar = pre.sa_chain_blocks(u, lambdas, cfl, atlas, op.gamma, op.viscosity,
                                     op.model, distance, layout)
        solve = pre.line_implicit_j(atlas, op.gamma, op.viscosity, op.model, distance)
        return flow, scalar, solve(u, lambdas, cfl)(r)

    # Trace before compilation so a promoted factor fails with a clear dtype assertion.
    abstract = jax.eval_shape(apply, U, rhs)
    assert all(a.dtype == U.dtype for a in jax.tree_util.tree_leaves(abstract))
    flow, scalar, z = jax.jit(apply)(U, rhs)
    assert np.isfinite(z).all()
    for blocks, rows in ((flow, slice(None, n_flow)), (scalar, slice(n_flow, None))):
        lower, diag, upper = (np.asarray(a, dtype=np.float64) for a in blocks)
        x = np.asarray(layout.gather(z[rows]), dtype=np.float64).transpose(2, 1, 0)
        b = np.asarray(layout.gather(rhs[rows]), dtype=np.float64).transpose(2, 1, 0)
        x = np.where(np.asarray(layout.valid).T[..., None], x, 0.)
        b = np.where(np.asarray(layout.valid).T[..., None], b, 0.)
        previous = np.concatenate([np.zeros_like(x[:1]), x[:-1]])
        following = np.concatenate([x[1:], np.zeros_like(x[:1])])
        applied = (np.einsum("...ij,...j->...i", lower, previous)
                   + np.einsum("...ij,...j->...i", diag, x)
                   + np.einsum("...ij,...j->...i", upper, following))
        tolerance = 2e-6 if precision == "fp32" else 1e-12
        np.testing.assert_allclose(applied, b, rtol=tolerance, atol=tolerance)
    # Ghosts and gaps do not belong to any chain and keep their RHS.
    outside = np.asarray(layout.inverse).reshape(atlas.shape) < 0
    np.testing.assert_array_equal(np.asarray(z)[:, outside], np.asarray(rhs)[:, outside])
    assert atlas.grid.volume.dtype == distance.dtype == np.float64
    assert atlas.centres.dtype == np.float64


def test_newton_acceptance_retains_sub_ulp_global_norm_decreases(monkeypatch):
    """Fields can move while the rounded FP32 norm ratio is exactly one."""
    from types import SimpleNamespace
    from jaxrans.core.config import ImplicitConfig
    from jaxrans.solvers import implicit

    U = jnp.asarray([1., 0., 0., 1.], jnp.float32).reshape(4, 1, 1)
    residual = U
    du = jnp.zeros_like(U).at[3, 0, 0].set(-jnp.finfo(jnp.float32).eps / 2)
    assert float(l2_norm(residual + du) / l2_norm(residual)) == 1.
    assert float(l2_norm(residual + du, dtype=jnp.float64) /
                 l2_norm(residual, dtype=jnp.float64)) < 1.
    operator = SimpleNamespace(
        gamma=1.4, components=2, atlas=SimpleNamespace(interior=jnp.ones((1, 1), bool)),
        viscosity=None, model=None, distance=None,
        residual=lambda u: residual, norm=lambda r: jnp.asarray(1., jnp.float32),
        wave_speeds=lambda u: (jnp.full((2, 1), .25, jnp.float32),
                                jnp.full((1, 2), .25, jnp.float32)))
    monkeypatch.setattr(implicit, "gmres_right", lambda *a, **kw: du)
    config = ImplicitConfig(cfl=1., preconditioner="none")
    state = implicit.ImplicitState(U, jnp.asarray(1., jnp.float32), jnp.asarray(1., jnp.float32))
    updated, _, report = jax.jit(implicit.make_newton_step(operator, config, diagnostics=True))(state)
    assert bool(report.accepted)
    assert updated.U.dtype == jnp.float32
    assert float(updated.U[3, 0, 0]) < float(U[3, 0, 0])


@pytest.mark.parametrize("solver,model", [("explicit", "none"), ("implicit", "sa_noft2"),
                                         ("implicit", "sa_noft2_qcr")])
def test_fp32_steps_and_jacobian_products_stay_single_and_follow_fp64(solver, model):
    c = case("fp32", solver, model)
    grid = c.grid()
    op = make_residual(grid, c.flow, c.numerics)
    U = op.initial_state()
    residual, tangent = jax.jit(lambda U: jax.jvp(op.residual, (U,), (jnp.ones_like(U),)))(U)
    assert residual.dtype == tangent.dtype == np.float32
    assert np.isfinite(residual).all() and np.isfinite(tangent).all()
    solutions = []
    for precision in ("fp32", "fp64"):
        current = case(precision, solver, model)
        step, state = make_stepper(current, current.grid())
        state, history = jax.jit(lambda s: jax.lax.scan(lambda s, _: step(s), s, None, length=3))(state)
        assert state.U.dtype == (np.float32 if precision == "fp32" else np.float64)
        assert np.isfinite(state.U).all() and np.isfinite(history).all()
        assert not np.array_equal(state.U, np.asarray(U, dtype=state.U.dtype))
        solutions.append(np.asarray(state.U))
    np.testing.assert_allclose(*solutions, atol=2e-5, rtol=2e-4)


def test_fp32_viscous_3d_residual_stays_single():
    nodes = np.stack(np.meshgrid(np.arange(5.), np.arange(5.)*.1, np.arange(5.)*.2, indexing="ij"))
    sides = box_sides((4, 4, 4), **{s: "farfield" for s in SIDES_3D})
    block = Grid.from_node_array(nodes, sides)
    c = case("fp32", model="none", swept=False)
    op = make_residual(build_atlas(assign_conditions(block, {"farfield": "far_field"})), c.flow, c.numerics)
    U = op.initial_state()
    R = jax.jit(op.residual)(U)
    assert R.dtype == np.float32
    assert np.isfinite(R).all()
    np.testing.assert_allclose(R, 0., atol=1e-7)


