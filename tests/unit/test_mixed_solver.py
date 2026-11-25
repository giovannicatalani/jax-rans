"""FP64 Newton/restart residuals with FP32 Jacobian products, vectors and line factors."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.core.config import FlowConditions, ImplicitConfig, NumericsConfig
from jaxrans.mesh.block import Grid, rectangle_sides
from jaxrans.numerics.residual import make_residual
from jaxrans.solvers import implicit, krylov, preconditioners as pre
from jaxrans.solvers.krylov import gmres_right

from ..single import atlas_of


def test_mixed_ill_conditioned_solve_refines_true_residual_in_double():
    diagonal = jnp.logspace(-8., 4., 12, dtype=jnp.float64)
    b = jnp.linspace(.7, 1.3, 12, dtype=jnp.float64)

    def matvec(v):
        assert v.dtype == jnp.float64
        return diagonal * v

    def precondition(r):
        assert r.dtype == jnp.float32
        return r / diagonal.astype(jnp.float32)

    def solve(b, cycles):
        return gmres_right(matvec, b, precondition, restart=2, maxiter=cycles,
                           tol=1e-13, basis_dtype=jnp.float32)

    first = jax.jit(lambda b: solve(b, 1))(b)
    refined = jax.jit(lambda b: solve(b, 4))(b)
    assert refined.dtype == jnp.float64
    assert np.isfinite(refined).all()
    assert np.linalg.norm(b - matvec(refined)) < 1e-12 * np.linalg.norm(b)
    assert np.linalg.norm(b - matvec(refined)) < np.linalg.norm(b - matvec(first))
    np.testing.assert_allclose(refined, b / diagonal, rtol=1e-12)
    np.testing.assert_array_equal(jax.jit(lambda b: solve(b, 4))(jnp.zeros_like(b)), 0.)


def toy_operator():
    atlas = SimpleNamespace(interior=jnp.ones((1, 1), bool))
    atlas.with_dtype = lambda dtype: atlas
    target = jnp.array([1., .1, 0., 2.3]).reshape(4, 1, 1)
    operator = SimpleNamespace(
        gamma=1.4, components=2, atlas=atlas, viscosity=None, model=None, distance=None,
        residual=lambda u: u - target, norm=lambda r: jnp.linalg.norm(r),
        wave_speeds=lambda u: (jnp.full((2, 1), .25), jnp.full((1, 2), .25)))
    state = implicit.ImplicitState(target.at[3].add(.2), jnp.asarray(1.), jnp.asarray(1.))
    return operator, state


@pytest.mark.parametrize("failure", [None, "nonfinite", "stagnation"])
@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_fallback_is_reported_and_double_factors_are_built_only_when_needed(monkeypatch, failure, batch_size):
    operator, state = toy_operator()
    setups = []

    def factory(*args):
        def prepare(u, lambdas, cfl):
            assert all(l.dtype == u.dtype for l in lambdas)
            assert cfl.dtype == u.dtype
            jax.debug.callback(lambda: setups.append(u.dtype))

            def apply(r):
                assert r.dtype == u.dtype
                if u.dtype == jnp.float32 and failure is not None:
                    return jnp.full_like(r, jnp.nan) if failure == "nonfinite" else jnp.zeros_like(r)
                return r

            return apply
        return prepare

    monkeypatch.setattr(implicit, "get_preconditioner", lambda name: factory)
    config = ImplicitConfig(cfl=1., krylov_subspace=2, krylov_restarts=2)
    step = implicit.make_newton_step(operator, config, mixed_precision=True, diagnostics=True)
    if batch_size is not None:
        state = jax.tree.map(lambda a: jnp.stack([a] * batch_size), state)
        step = jax.vmap(step)
    updated, _, report = jax.jit(step)(state)
    jax.effects_barrier()
    assert np.all(np.asarray(report.linear_fp64_fallback) == (failure is not None))
    assert setups == ([jnp.dtype("float32"), jnp.dtype("float64")] if failure
                      else [jnp.dtype("float32")])
    assert np.asarray(report.accepted).all()
    assert np.isfinite(updated.U).all()
    np.testing.assert_allclose(updated.U[..., 3, :, :], 2.4, atol=1e-12)
    assert np.all(np.asarray(report.linear_reduction) < (1e-12 if failure else 1e-6))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_default_newton_matches_explicitly_disabled_mixed_mode(dtype):
    operator, state = toy_operator()
    # Match the operator to a single-precision primal for the FP32 regression.
    target = (state.U.at[3].add(-.2)).astype(dtype)
    operator.residual = lambda u: u - target
    operator.wave_speeds = lambda u: (jnp.full((2, 1), .25, dtype), jnp.full((1, 2), .25, dtype))
    state = jax.tree.map(lambda a: a.astype(dtype), state)
    config = ImplicitConfig(cfl=1., preconditioner="none", krylov_subspace=2)
    default = jax.jit(implicit.make_newton_step(operator, config, diagnostics=True))(state)
    disabled = jax.jit(implicit.make_newton_step(
        operator, config, diagnostics=True, mixed_precision=False))(state)
    for a, b in zip(jax.tree.leaves(default), jax.tree.leaves(disabled)):
        np.testing.assert_array_equal(a, b)
    assert default[0].U.dtype == dtype
    np.testing.assert_allclose(default[0].U[3], 2.4, atol=2e-7)
    assert not bool(default[2].linear_fp64_fallback)


def test_mixed_newton_vmap_preserves_case_and_diagnostic_shapes():
    operator, state = toy_operator()
    config = ImplicitConfig(cfl=1., preconditioner="none", krylov_subspace=2)
    batch = jax.tree.map(lambda a: jnp.stack([a, a]), state)
    result, norms, report = jax.jit(jax.vmap(implicit.make_newton_step(
        operator, config, diagnostics=True, mixed_precision=True)))(batch)
    assert result.U.shape == (2, 4, 1, 1)
    assert result.U.dtype == norms.dtype == jnp.float64
    assert all(a.shape == (2,) for a in report)
    assert report.linear_fp64_fallback.dtype == jnp.bool_
    assert np.asarray(report.accepted).all()
    assert not np.asarray(report.linear_fp64_fallback).any()


def test_vmap_fallback_preserves_the_successful_mixed_case(monkeypatch):
    operator, state = toy_operator()

    def factory(*args):
        def prepare(u, lambdas, cfl):
            if u.dtype == jnp.float32:
                return lambda r: jnp.where(u[3, 0, 0] > 2.6, jnp.zeros_like(r), r)
            return lambda r: r
        return prepare

    monkeypatch.setattr(implicit, "get_preconditioner", lambda name: factory)
    config = ImplicitConfig(cfl=1., krylov_subspace=2)
    step = implicit.make_newton_step(operator, config, diagnostics=True, mixed_precision=True)
    single = jax.jit(step)(state)
    second = state._replace(U=state.U.at[3].add(.2))
    batch = jax.tree.map(lambda a, b: jnp.stack([a, b]), state, second)
    updated, _, report = jax.jit(jax.vmap(step))(batch)
    np.testing.assert_array_equal(report.linear_fp64_fallback, [False, True])
    np.testing.assert_array_equal(updated.U[0], single[0].U)
    np.testing.assert_allclose(updated.U[:, 3, 0, 0], [2.4, 2.5], atol=1e-8)
    assert np.asarray(report.accepted).all()


def test_small_rans_step_keeps_factors_basis_and_jvp_single_but_residual_double(monkeypatch):
    nodes = np.stack(np.meshgrid(np.linspace(0., 1., 5), np.linspace(0., .3, 4), indexing="ij"))
    grid = Grid.from_node_array(nodes, rectangle_sides(
        4, 3, j_min="wall", j_max="farfield", i_min="farfield", i_max="farfield"))
    flow = FlowConditions(mach=.2, reynolds=5000., turbulence="sa_noft2")
    operator = make_residual(atlas_of(grid, no_slip=True), flow,
                             NumericsConfig(limiter="none"), first_order=True)
    config = ImplicitConfig(cfl=.1, cfl_min=.01, krylov_subspace=3, krylov_restarts=2,
                            max_relative_update=.2)
    state = implicit.initial_state(operator, config)
    factored, basis = [], []
    original_factor = pre.factor_cyclic_reduction
    original_gmres = implicit.gmres_right
    original_linearize = jax.linearize
    original_backsolve = krylov._backsolve

    def backsolve(h, rhs, rcond):
        assert h.shape == (config.krylov_subspace, config.krylov_subspace)
        assert h.dtype == rhs.dtype == jnp.float64
        return original_backsolve(h, rhs, rcond)

    def linearize(fun, u):
        value, tangent = original_linearize(fun, u)
        assert value.dtype == u.dtype
        return value, tangent

    def factor(lower, diagonal, upper):
        result = original_factor(lower, diagonal, upper)
        assert all(a.dtype == diagonal.dtype for a in jax.tree.leaves(result))
        factored.append(diagonal.dtype)
        return result

    def gmres(matvec, b, precondition, **kwargs):
        assert b.dtype == jnp.float64

        def action(v):
            expected = kwargs["operator_dtype"] or b.dtype
            assert v.dtype == expected
            result = matvec(v)
            assert result.dtype == expected
            return result

        def apply(r):
            basis.append(r.dtype)
            expected = kwargs["basis_dtype"] or b.dtype
            assert r.dtype == expected
            result = precondition(r)
            assert result.dtype == expected
            return result

        return original_gmres(action, b, apply, **kwargs)

    monkeypatch.setattr(pre, "factor_cyclic_reduction", factor)
    monkeypatch.setattr(implicit, "gmres_right", gmres)
    monkeypatch.setattr(jax, "linearize", linearize)
    monkeypatch.setattr(krylov, "_backsolve", backsolve)
    result, norm, report = jax.jit(implicit.make_newton_step(
        operator, config, mixed_precision=True, diagnostics=True))(state)
    # Both branches are traced, but both flow and SA factors must be FP32 in
    # the mixed branch. Runtime laziness is checked separately above.
    assert factored[:2] == [jnp.float32, jnp.float32]
    assert jnp.float32 in basis
    assert result.U.dtype == result.cfl.dtype == norm.dtype == jnp.float64
    assert report.linear_reduction.dtype == report.damping.dtype == jnp.float64
    assert np.isfinite(result.U).all() and np.isfinite(norm)
    assert bool(report.accepted)
    assert not bool(report.linear_fp64_fallback)
    assert operator.atlas.grid.volume.dtype == operator.distance.dtype == jnp.float64


@pytest.mark.parametrize("basis_dtype", [None, jnp.float32])
def test_gmres_reuses_restart_residual_and_returns_true_action(basis_dtype):
    calls = []
    diagonal = jnp.array([1., 2., 5., 11.], dtype=jnp.float64)
    b = jnp.ones(4, dtype=jnp.float64)

    def matvec(v):
        jax.debug.callback(lambda: calls.append(1))
        return diagonal * v

    x, action = jax.jit(lambda b: gmres_right(
        matvec, b, lambda r: r, restart=2, maxiter=2, tol=1e-30,
        basis_dtype=basis_dtype, return_action=True))(b)
    jax.effects_barrier()
    # Two Arnoldi actions and one true residual action per cycle.
    assert len(calls) == 6
    np.testing.assert_allclose(action, diagonal * x, rtol=1e-14)
    assert np.linalg.norm(b - action) < np.linalg.norm(b)


@pytest.mark.parametrize("batch_size", [None, 1, 2])
def test_stagnated_fp64_fallback_cuts_cfl_without_tiny_updates(monkeypatch, batch_size):
    operator, state = toy_operator()
    calls = []

    def near_null(matvec, b, precondition, **kwargs):
        jax.debug.callback(lambda: calls.append(kwargs["basis_dtype"]))
        correction = b * 1e-11
        return correction, matvec(correction)

    monkeypatch.setattr(implicit, "gmres_right", near_null)
    config = ImplicitConfig(cfl=1., preconditioner="none")
    step = implicit.make_newton_step(operator, config, mixed_precision=True, diagnostics=True)
    if batch_size is not None:
        state = jax.tree.map(lambda a: jnp.stack([a] * batch_size), state)
        step = jax.vmap(step)
    updated, _, report = jax.jit(step)(state)
    jax.effects_barrier()
    assert len(calls) == 2  # both the mixed direction and its FP64 retry
    assert np.asarray(report.linear_fp64_fallback).all()
    assert np.asarray(report.linear_stagnated).all()
    assert not np.asarray(report.accepted).any()
    np.testing.assert_array_equal(updated.U, state.U)
    np.testing.assert_allclose(updated.cfl, config.cfl_min)
