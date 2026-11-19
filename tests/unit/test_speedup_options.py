"""Opt-in solver options and the single-plane sectional residual.

Every option defaults to the previous behaviour; these tests pin what each
one changes and what it must keep.
"""

from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans import load_case
from jaxrans import campaign as module
from jaxrans.campaign import PolarRequest, run_campaign
from jaxrans.core.config import ImplicitConfig, NumericsConfig
from jaxrans.numerics import boundary as bc
from jaxrans.numerics.limiters import get_limiter
from jaxrans.numerics.residual import (block_residual, make_residual, primitive_from_conservative)
from jaxrans.numerics.riemann import get_flux
from jaxrans.solvers.implicit import LOCAL_RAMP_FRACTION, _ratios, relax, relax_correction
from jaxrans.solvers.krylov import gmres_right

ROOT = Path(__file__).resolve().parents[2]


# --- GMRES early exit ----------------------------------------------------------------


def system(n=40, seed=0):
    rng = np.random.default_rng(seed)
    A = np.eye(n) * 6.0 + 0.3 * rng.standard_normal((n, n))
    return jnp.asarray(A), jnp.asarray(rng.standard_normal(n))


@pytest.mark.parametrize("tol", [1e-2, 1e-6])
def test_early_exit_meets_the_tolerance_like_the_full_cycle(tol):
    A, b = system()
    solve = partial(gmres_right, lambda v: A @ v, b, lambda r: r, 20, 2, tol)
    full, early = solve(), solve(early_exit=True)
    for x in (full, early):
        assert np.linalg.norm(b - A @ x) <= tol * np.linalg.norm(b) * (1 + 1e-10)
    # A well-conditioned system meets 1e-2 in a few vectors: fewer than 20.
    if tol == 1e-2:
        assert not np.allclose(full, early)


def test_early_exit_returns_the_true_action_and_zero_for_zero_rhs():
    A, b = system(seed=3)
    x, action = gmres_right(lambda v: A @ v, b, lambda r: r, 10, 3, 1e-8, return_action=True,
                            action_from_arnoldi=True, early_exit=True)
    np.testing.assert_allclose(action, A @ x, rtol=1e-9, atol=1e-11)
    zero = gmres_right(lambda v: A @ v, jnp.zeros_like(b), lambda r: r, 10, 2, 1e-8, early_exit=True)
    assert not np.any(np.asarray(zero))


# --- relaxation modes ------------------------------------------------------------------


def state(seed=0, shape=(6, 7)):
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.1 * rng.random(shape)
    u, v, w = (0.3 * rng.standard_normal(shape) for _ in range(3))
    p = 1 / 1.4 * (1.0 + 0.1 * rng.random(shape))
    nu = 3.0 + 50.0 * rng.random(shape)
    E = p / 0.4 + 0.5 * rho * (u * u + v * v + w * w)
    U = jnp.asarray(np.stack([rho, rho * u, rho * v, rho * w, E, rho * nu]))
    dU = jnp.asarray(0.05 * rng.standard_normal(U.shape)) * jnp.abs(U)
    dU = dU.at[:, 0, 0].multiply(40.0)                      # one cell asks for a huge step
    return U, dU


def test_global_mode_is_relax():
    U, dU = state()
    a, omega = relax(dU, U, 1.4, 0.2, 3)
    b, omega_b, ramp, scaled = relax_correction(dU, U, 1.4, 0.2, 3, "global")
    np.testing.assert_array_equal(a, b)
    assert omega == omega_b and scaled and bool(ramp) == bool(omega >= 1.0)


def test_local_mode_bounds_each_cell_and_leaves_the_others_untouched():
    U, dU = state()
    out, omega, ramp, scaled = relax_correction(dU, U, 1.4, 0.2, 3, "local")
    flow, sa, _ = _ratios(out, U, 1.4, 3)
    assert float(jnp.max(jnp.maximum(jnp.maximum(*flow), sa))) <= 0.2 * (1 + 1e-12)
    flow0, sa0, _ = _ratios(dU, U, 1.4, 3)
    free = np.asarray(jnp.maximum(jnp.maximum(*flow0), sa0) <= 0.2)
    np.testing.assert_array_equal(np.asarray(out)[:, free], np.asarray(dU)[:, free])
    assert not scaled and 0 < float(omega) < 1
    # One cut cell out of 42 is above LOCAL_RAMP_FRACTION: no ramp.
    cut = int(np.sum(~free))
    assert bool(ramp) == (cut <= LOCAL_RAMP_FRACTION * free.size)


def test_split_mode_damps_the_sa_row_on_its_own():
    U, dU = state(seed=2)
    dU = dU.at[5].multiply(100.0)                           # only nu~ asks for a large step
    out, omega, ramp, scaled = relax_correction(dU, U, 1.4, 0.2, 3, "split")
    _, omega_global = relax(dU, U, 1.4, 0.2, 3)
    assert float(omega) > float(omega_global)               # the flow is not held back by nu~
    np.testing.assert_allclose(out[:5], omega * dU[:5])
    ratio = np.asarray(out[5] / dU[5])
    assert np.allclose(ratio, ratio.flat[0]) and ratio.flat[0] < 1
    assert not scaled


def test_relaxation_validation():
    with pytest.raises(ValueError, match="relaxation"):
        ImplicitConfig(relaxation="cellwise")
    with pytest.raises(ValueError, match="krylov_early_exit"):
        ImplicitConfig(krylov_early_exit=1)
    with pytest.raises(ValueError, match="equation_peak"):
        NumericsConfig(equation_peak="campaign")


# --- single-plane sectional residual ---------------------------------------------------


@pytest.fixture(scope="module")
def sectional_case():
    return load_case(ROOT / "examples/rae2822_polar_reference/case_275d", (
        "case.domain.cells_airfoil=48", "case.domain.cells_wake=12", "case.domain.cells_normal=24",
        "case.general.precision=fp64", "case.domain.sectional.taper_rate=0.2"))


@pytest.mark.parametrize("first_order", [False, True])
def test_sectional_residual_on_one_plane_equals_the_whole_canvas(sectional_case, first_order):
    """The previous evaluation (stencil on every canvas plane, then masked)
    and the single-plane one agree to round-off, residual and JVP."""
    case = sectional_case
    grid = case.grid()
    op = make_residual(grid, case.flow, case.numerics, first_order=first_order)
    atlas, g = op.atlas, case.numerics.n_ghost
    stencil = partial(block_residual, flux=get_flux(case.numerics.flux), limiter=get_limiter(case.numerics.limiter),
                      gamma=op.gamma, n_ghost=g, first_order=first_order, viscosity=op.viscosity,
                      model=op.model, kappa=case.numerics.kappa)
    far_field = bc.FAR_FIELDS[case.numerics.far_field]
    U_0 = op.initial_state()

    def whole_canvas(U):
        q = primitive_from_conservative(U, op.gamma, atlas.components)
        section = far_field(atlas.section_values(q), atlas.reference, op.q_inf, op.gamma)
        conditions = bc.Conditions(op.q_inf, op.gamma, lambda midpoint: section(midpoint[:2]))
        padded = jnp.pad(bc.fill_ghosts(q, atlas, conditions), ((0, 0),) + ((g, g),) * 3, mode="edge")
        return jnp.where(atlas.interior, stencil(padded, atlas.grid, atlas.centres, op.distance), U - U_0)

    U = U_0 * (1 + 0.05 * jax.random.normal(jax.random.PRNGKey(1), U_0.shape))
    V = jax.random.normal(jax.random.PRNGKey(2), U_0.shape) * jnp.abs(U_0)
    np.testing.assert_allclose(op.residual(U), whole_canvas(U), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jax.jvp(op.residual, (U,), (V,))[1], jax.jvp(whole_canvas, (U,), (V,))[1],
                               rtol=1e-11, atol=1e-11)


# --- chain-relative equation thresholds ------------------------------------------------


SMALL = ("case.domain.cells_airfoil=32", "case.domain.cells_normal=16",
         "numerical.convergence.max_iters=60", "numerical.convergence.block_iters=2",
         "numerical.convergence.forces_tol=null", "numerical.convergence.tol=0.001")


@pytest.fixture
def contraction(monkeypatch):
    from jaxrans.simulation import make_stepper

    def factory(case, grid, freestream=None, distance=None):
        _, initial = make_stepper(case, grid, freestream=freestream, distance=distance)
        target = initial.U * (1.02 + 0.01 * jnp.tanh(grid.grid.xc) + freestream[2])

        def step(s):
            residual = jnp.max(jnp.abs(target - s.U))
            return s._replace(U=0.5 * (target + s.U)), residual
        return step, initial

    def norms(case, template, U, q, g):
        _, initial = factory(case, template, freestream=q)
        target = initial.U * (1.02 + 0.01 * jnp.tanh(template.grid.xc) + q[2])
        return jnp.full((U.shape[0],), jnp.max(jnp.abs(target - U)))

    monkeypatch.setattr(module, "make_stepper", factory)
    monkeypatch.setattr(module, "_equation_norms", norms)


@pytest.mark.parametrize("scheduler", ["slots", "rounds"])
def test_chain_peak_carries_the_first_points_level(contraction, scheduler):
    """A continued point starts near its solution: relative to its own small
    peak it must go much lower than the first point did; relative to the
    chain's peak it needs the same level."""
    base = load_case(ROOT / "examples/naca0012_euler", SMALL)
    chained = replace(base, numerics=replace(base.numerics, equation_peak="chain"))
    angles = (0.0, 0.5, 1.0)
    point = run_campaign([PolarRequest("p", base, angles)], scheduler=scheduler).polars["p"]
    chain = run_campaign([PolarRequest("p", chained, angles)], scheduler=scheduler).polars["p"]
    first = chain.results[0].convergence_history[-1]["peak"]
    for a, b in zip(point.results[1:], chain.results[1:]):
        assert b.history.n_iters <= a.history.n_iters
        assert np.all(np.asarray(b.convergence_history[-1]["peak"]) >= np.asarray(first))
        # Never stricter than the point's own thresholds, block by block.
        for ra, rb in zip(a.convergence_history, b.convergence_history):
            assert np.all(np.asarray(rb["thresholds"]) >= np.asarray(ra["thresholds"]))
    # The first point has no earlier peak: identical.
    assert point.results[0].history.n_iters == chain.results[0].history.n_iters
    np.testing.assert_array_equal(point.results[0].U, chain.results[0].U)


# --- sectional Newton step on the plane ------------------------------------------------


@pytest.mark.parametrize("jacobian", ["first_order", "exact"])
def test_sectional_newton_on_the_plane_equals_the_canvas_state(sectional_case, jacobian):
    """Krylov vectors, line solves and acceptance on the independent plane:
    the same steps as with the whole five-plane canvas state."""
    from jaxrans.solvers import implicit

    case = replace(sectional_case, implicit=replace(sectional_case.implicit, jacobian=jacobian))
    grid = case.grid()
    op = make_residual(grid, case.flow, case.numerics)
    jac = None if jacobian == "exact" else make_residual(grid, case.flow, case.numerics, first_order=True,
                                                         distance=op.distance)
    assert op.plane is not None
    canvas_op = replace(op, plane=None)
    canvas_jac = None if jac is None else replace(jac, plane=None)
    on_plane = implicit.make_newton_step(op, case.implicit, jac)
    on_canvas = implicit.make_newton_step(canvas_op, case.implicit, canvas_jac)
    run = lambda step: jax.jit(lambda s: jax.lax.scan(lambda c, _: step(c), s, None, length=8))
    start = implicit.initial_state(op, case.implicit)
    a, ra = run(on_plane)(start)
    b, rb = run(on_canvas)(start)
    np.testing.assert_allclose(a.U, b.U, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(ra, rb, rtol=1e-10)
    assert float(a.cfl) == float(b.cfl)
