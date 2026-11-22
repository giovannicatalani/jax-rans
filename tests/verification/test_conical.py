"""Conical metrics, vanishing-taper limit and manufactured Euler/NS fluxes."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.case import parse_case
from jaxrans.core.config import FlowConditions, NumericsConfig
from jaxrans.core.sectional import SectionalSpec
from jaxrans.core.state import primitive_to_conservative, split
from jaxrans.mesh.block import Grid, normal_closure_residual, rectangle_sides
from jaxrans.mesh.mesh import assign_conditions
from jaxrans.mesh.sectional import build_sectional_atlas
from jaxrans.numerics.residual import make_residual
from jaxrans.post.forces import force_coefficients
from tests.karman_trefftz import TMR, make_o_grid
from tests.unit.test_sectional import document


@pytest.fixture(scope="module")
def section():
    return assign_conditions(make_o_grid(16, 32, 1e-3, section=TMR),
                             {"wall": "slip_wall", "farfield": "far_field"})


def operator(mesh, taper, depth=0.1, flux="hllc", start=1.0, end=5.0):
    spec = SectionalSpec(leading_edge_sweep_deg=30.0, taper_rate=taper)
    atlas = build_sectional_atlas(mesh, spec, depth=depth, relaxation_start=start, relaxation_end=end)
    flow = FlowConditions(mach=0.5, alpha_deg=2.0, sweep_deg=spec.reference_sweep_deg)
    return make_residual(atlas, flow, NumericsConfig(limiter="none", flux=flux))


@pytest.mark.parametrize("taper", [-0.2, 0.2])
@pytest.mark.parametrize("flux", ["roe", "hllc"])
def test_uniform_flow_is_preserved_by_the_relaxed_conical_metrics(section, taper, flux):
    op = operator(section, taper, flux=flux)
    atlas = op.atlas
    stages = tuple(replace(s, conditions=tuple(("far_field", source, normal, mid)
                                               for _, source, normal, mid in s.conditions)) for s in atlas.stages)
    atlas = replace(atlas, stages=stages)
    op = make_residual(atlas, FlowConditions(mach=0.5, alpha_deg=2.0,
                                           sweep_deg=atlas.spec.reference_sweep_deg), NumericsConfig(flux=flux))
    R = np.asarray(op.residual(op.initial_state()))[:, atlas.interior]
    scale = max(float(jnp.abs(s).max()) for s in atlas.grid.faces)
    assert np.abs(R).max() < 2e-12 * scale
    assert np.abs(normal_closure_residual(atlas.mesh[0])).max() < 1e-12 * scale


def test_the_full_conical_operator_converges_continuously_to_2p5d(section, record_property):
    base = operator(section, 0.0)
    rng = np.random.default_rng(4)
    U = base.initial_state() * jnp.asarray(1 + 0.02 * rng.normal(size=base.initial_state().shape))
    expected = np.asarray(base.residual(U) / base.grid.volume)[:, base.atlas.interior]
    errors = []
    for taper in (1e-3, 1e-4, 1e-5):
        op = operator(section, taper)
        actual = np.asarray(op.residual(U) / op.grid.volume)[:, op.atlas.interior]
        errors.append(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    assert all(a / b > 8 for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < 1e-3
    record_property("relative_errors", errors)


def test_depth_refinement_has_a_limit_at_finite_taper(section, record_property):
    residuals = []
    rng = np.random.default_rng(6)
    for depth in (0.2, 0.1, 0.05, 0.025):
        op = operator(section, 0.2, depth=depth)
        U = op.initial_state() * jnp.asarray(1 + 0.02 * rng.normal(size=op.initial_state().shape))
        # Same seeded section state for each depth.
        rng = np.random.default_rng(6)
        residuals.append(np.asarray(op.residual(U) / op.grid.volume)[:, op.atlas.interior])
    errors = [np.linalg.norm(r - residuals[-1]) for r in residuals[:-1]]
    assert errors[0] / errors[1] > 3 and errors[1] / errors[2] > 3, errors
    record_property("depth_errors", errors)


def test_pressure_forces_have_the_same_zero_taper_limit(section):
    errors = []
    base = operator(section, 0.0)
    U = base.initial_state() * jnp.asarray(1 + 0.02 * np.random.default_rng(8).normal(size=base.initial_state().shape))
    flow = FlowConditions(mach=0.5, alpha_deg=2.0, sweep_deg=30.0)
    expected = force_coefficients(U, base.atlas, flow)
    for taper in (1e-3, 1e-4, 1e-5):
        op = operator(section, taper)
        flow = replace(flow, sweep_deg=op.atlas.spec.reference_sweep_deg)
        actual = force_coefficients(U, op.atlas, flow)
        errors.append(max(abs(float(actual[k] - expected[k])) for k in ("cl", "cd", "cm", "cy")))
    assert all(a / b > 8 for a, b in zip(errors, errors[1:])), errors


def test_nonzero_taper_changes_the_pressure_side_force():
    case = parse_case(document(taper_rate=0.2), {"sectional": {"depth": 0.1}})
    atlas = case.grid()
    op = make_residual(atlas, case.flow, case.numerics)
    q = jnp.broadcast_to(op.q_inf[:, None, None, None], op.initial_state().shape)
    x = atlas.grid.centres[0]
    q = q.at[4].add(0.02 * x)
    U = primitive_to_conservative(*split(q, 3), case.flow.gamma)
    c = force_coefficients(U, atlas, case.flow)
    # Independent integration of the actual conical wall at streamwise s=0.
    walls = atlas.reference.walls.faces
    a, b = (atlas.geometry.streamwise_points(p.T) for p in (walls.a, walls.b))
    mid = (a + b) / 2
    dx, dy = b - a
    span_area = (np.tan(np.deg2rad(30.0)) - 0.2 * mid[0]) * dy + 0.2 * mid[1] * dx
    p = np.asarray(q[4]).ravel()[atlas.walls.cell] - op.q_inf[4]
    np.testing.assert_allclose(c["cy"], -np.sum(p * span_area) / (0.5 * case.flow.mach**2), atol=1e-12)
    assert abs(float(c["cy"])) > 1e-5


def test_mapped_wall_conditions_use_the_three_dimensional_surface_normal(section):
    op = operator(section, 0.2)
    conditions = [c for s in op.atlas.stages for c in s.conditions if c[0] == "slip_wall"]
    assert conditions and max(np.abs(c[2][2]).max() for c in conditions) > 1e-3
    # Automatic differentiation must include the Cartesian copy closure.
    U = op.initial_state()
    _, dR = jax.jvp(op.residual, (U,), (jnp.ones_like(U) * 0.001,))
    assert np.isfinite(np.asarray(dR)[:, op.atlas.interior]).all()


def test_splitting_the_conical_slab_preserves_every_euler_residual(section):
    from tests.verification.test_multiblock import split as split_grid

    mesh, windows = split_grid(section[0], (3, 8, 13), (3, 11))
    single, multi = operator(section, 0.2), operator(mesh, 0.2)
    U = single.initial_state() * jnp.asarray(1 + 0.02 * np.random.default_rng(4).normal(size=single.initial_state().shape))
    (block,) = single.atlas.unpack(U)
    V = multi.initial_state()
    for window, (i, j) in zip(multi.atlas.windows, windows):
        V = V.at[(slice(None),) + window].set(block[:, i, j])
    (expected,) = single.atlas.unpack(single.residual(U))
    actual = multi.atlas.unpack(multi.residual(V))
    scale = max(float(jnp.abs(expected).max()), 1e-20)
    for got, (i, j) in zip(actual, windows):
        assert float(jnp.abs(got - expected[:, i, j]).max()) < 1e-11 * scale


@pytest.mark.parametrize("viscous", [False, True])
def test_conical_residual_matches_independent_3d_flux_integrals_at_second_order(record_property, viscous):
    from tests.verification.test_mms3d import source

    errors = []
    for n in (12, 24, 48):
        x, y = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0.15, 0.65, n + 1), indexing="ij")
        mesh = Grid.from_nodes(x, y, rectangle_sides(n, n, **dict.fromkeys(
            ("i_min", "i_max", "j_min", "j_max"), "wall")))
        mesh = assign_conditions(mesh, {"wall": "slip_wall"})
        op = operator(mesh, 0.3, depth=1.0 / n, start=10.0, end=20.0)
        if viscous:
            flow = FlowConditions(mach=.5, reynolds=10., sweep_deg=op.atlas.spec.reference_sweep_deg)
            op = make_residual(op.atlas, flow, NumericsConfig(limiter="none"))
        geometry = op.atlas.geometry
        p, mu = geometry.coefficients

        def primitive(P):
            # Analytic conical field in physical Cartesian coordinates.
            X = (P[0] - p * P[2]) / (1 - mu * P[2])
            Y = P[1] / (1 - mu * P[2])
            return np.stack([1 + 0.05 * np.sin(X) * np.cos(Y),
                             0.45 + 0.07 * np.cos(X + Y), 0.1 + 0.03 * np.sin(X - Y),
                             0.15 + 0.04 * np.cos(Y), 0.7 + 0.03 * np.sin(X + Y)])

        def flux(P):
            rho, u, v, w, pressure = primitive(P)
            V = np.stack([u, v, w])
            energy = pressure / (op.gamma - 1) + 0.5 * rho * np.sum(V * V, axis=0)
            out = [rho * V]
            for a in range(3):
                out.append(rho * V[a] * V + np.eye(3)[a].reshape((3,) + (1,) * rho.ndim) * pressure)
            out.append((energy + pressure) * V)
            out = np.stack(out)
            if viscous:
                X = (P[0] - p * P[2]) / (1 - mu * P[2])
                Y = P[1] / (1 - mu * P[2])
                f = 1 - mu * P[2]
                zeros = np.zeros_like(X)
                dxy = np.stack([np.stack([1 / f, zeros, (mu * P[0] - p) / f**2]),
                                np.stack([zeros, 1 / f, mu * P[1] / f**2])])
                dq = np.stack([
                    np.stack([.05 * np.cos(X) * np.cos(Y), -.05 * np.sin(X) * np.sin(Y)]),
                    np.stack([-.07 * np.sin(X + Y), -.07 * np.sin(X + Y)]),
                    np.stack([.03 * np.cos(X - Y), -.03 * np.cos(X - Y)]),
                    np.stack([zeros, -.04 * np.sin(Y)]),
                    np.stack([.03 * np.cos(X + Y), .03 * np.cos(X + Y)])])
                grad = np.einsum("ab...,bd...->ad...", dq, dxy)
                grad_v = grad[1:4]
                identity = np.eye(3).reshape((3, 3) + (1,) * rho.ndim)
                tau = flow.mu_inf * (grad_v + np.swapaxes(grad_v, 0, 1)
                                     - (2 / 3) * np.trace(grad_v, axis1=0, axis2=1) * identity)
                grad_T = op.gamma * (grad[4] / rho - pressure * grad[0] / rho**2)
                out[1:4] -= tau
                out[4] -= np.sum(V[:, None] * tau, axis=0) + flow.mu_inf / (.72 * (op.gamma - 1)) * grad_T
            return out

        q = jnp.asarray(primitive(np.asarray(op.grid.centres)))
        U = primitive_to_conservative(*split(q, 3), op.gamma)
        actual = np.asarray(op.atlas.unpack(op.residual(U))[0])
        exact = source(np.asarray(op.atlas.mesh[0].nodes), flux)
        volume = np.asarray(op.atlas.mesh[0].volume)
        # Physical wall conditions do not match the manufactured field; remove
        # their stencil reach. Lateral copies do match the analytic conicity.
        error = ((actual - exact) / volume)[..., 3:-3, 3:-3, :]
        errors.append(float(np.sqrt(np.mean(error**2))))
    assert errors[0] / errors[1] > 3.0 and errors[1] / errors[2] > 3.0, errors
    record_property("l2_errors", errors)
    record_property("observed_orders", [float(np.log2(a / b)) for a, b in zip(errors, errors[1:])])
