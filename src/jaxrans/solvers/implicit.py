"""Pseudo-transient Newton-Krylov (JFNK) stepping.

Each step solves the backward-Euler system

    (V/dt I + dR/dU) dU = -R(U),        U <- U + dU

with the CFL that sets ``V/dt`` ramped up as the iteration succeeds. At low CFL
this is a robust implicit march; as ``V/dt -> 0`` it becomes Newton's method on
``R(U) = 0``. That is pseudo-transient continuation.

Jacobian-vector products from ``jax.linearize``, right-preconditioned GMRES
(``krylov.py``). ``ImplicitConfig.jacobian``: ``exact`` linearizes the residual
solved; ``first_order`` (default) the first-order one, the second-order
residual staying on the right-hand side (defect correction): same fixed point,
no limiter kinks in the operator, and the system the preconditioner
approximates.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from ..core.config import ImplicitConfig
from ..core.precision import floor_like, l2_norm
from ..core.state import RHO, Rows, conservative_to_primitive, dot, is_physical
from ..numerics.residual import ResidualOperator, cell_wave_speed_sum
from . import explicit
from .krylov import gmres_right
from .preconditioners import get_preconditioner


class ImplicitState(NamedTuple):
    """``U (nvar, *canvas shape)``, the CFL and the previous ``|R|``: traced, so the ramp needs no recompilation."""

    U: Array
    cfl: Array
    residual_norm: Array


class NewtonDiagnostics(NamedTuple):
    """Per-step facts, before rejected updates are masked out of the state."""

    accepted: Array
    correction_finite: Array
    physical: Array
    linear_reduction: Array
    damping: Array
    cfl: Array
    changed: Array
    density_min: Array
    pressure_min: Array
    nonlinear_ratio: Array
    backtracks: Array
    linear_fp64_fallback: Array


def initial_state(
    operator: ResidualOperator, config: ImplicitConfig, dtype=None
) -> ImplicitState:
    """The explicit path's freestream start, at the starting CFL."""
    dtype = operator.q_inf.dtype if dtype is None else dtype
    return ImplicitState(
        U=explicit.initial_state(operator, dtype=dtype).U,
        cfl=jnp.asarray(config.cfl, dtype=dtype),
        residual_norm=jnp.asarray(jnp.inf, dtype=dtype),
    )


def relax(dU: Array, U: Array, gamma: float, fraction: float | None, dim: int = 2) -> tuple[Array, Array]:
    """Scale ``dU`` by one global factor so no cell's density or pressure moves
    by more than ``fraction`` of itself, nor ``nu~`` by more than ``fraction``
    of ``max(|nu~|, nu_inf)`` (``nu_hat`` is stored over ``nu_inf``, so the
    floor is 1). Returns ``(omega dU, omega)``; ``omega = 1`` when ``fraction``
    is ``None``, and the update is then returned untouched.

    The pressure change is linearized, ``dp = (gamma - 1)(dE - V.d(rho V) +
    |V|^2/2 d rho)``, which is what a small step does to it.
    """
    one = jnp.ones((), dU.dtype)
    if fraction is None:
        return dU, one
    rows = Rows(dim)
    rho, velocity, p = conservative_to_primitive(U, gamma, dim)
    d_rho = dU[RHO]
    d_e = dU[rows.energy]
    for k in range(dim):
        d_e = d_e - velocity[k] * dU[1 + k]
    d_p = (gamma - 1.0) * (d_e + 0.5 * dot(velocity, velocity) * d_rho)
    ratios = [jnp.abs(d_rho) / rho, jnp.abs(d_p) / p]
    if U.shape[0] > rows.n_flow:
        nu_hat = U[rows.nu] / rho
        d_nu_hat = (dU[rows.nu] - nu_hat * d_rho) / rho
        ratios.append(jnp.abs(d_nu_hat) / jnp.maximum(jnp.abs(nu_hat), 1.0))
    worst = jnp.max(jnp.stack([jnp.max(r) for r in ratios]))
    omega = jnp.minimum(one, fraction / jnp.maximum(worst, floor_like(worst)))
    return omega * dU, omega


def make_newton_step(
    operator: ResidualOperator,
    config: ImplicitConfig = ImplicitConfig(),
    jacobian_operator: ResidualOperator | None = None,
    *, diagnostics: bool = False, mixed_precision: bool = False,
) -> Callable[[ImplicitState], tuple[ImplicitState, Array] | tuple[ImplicitState, Array, NewtonDiagnostics]]:
    """``step(state) -> (state, |R(U)| before the update)``; ``jacobian_operator``
    (a first-order residual: defect correction) supplies the linear operator,
    ``None`` linearizes ``operator`` itself. ``diagnostics=True`` appends a
    ``NewtonDiagnostics`` result, preserving facts about rejected corrections.
    ``mixed_precision`` keeps the state/operator in FP64, with FP32 line factors
    and Krylov vectors, retrying a failed linear solve entirely in FP64.
    """
    gamma = operator.gamma
    prepare_preconditioner = get_preconditioner(config.preconditioner)(
        operator.atlas, gamma, operator.viscosity, operator.model, operator.distance
    )
    if mixed_precision:
        prepare_low = get_preconditioner(config.preconditioner)(
            operator.atlas.with_dtype(jnp.float32), gamma, operator.viscosity,
            operator.model, None if operator.distance is None else operator.distance.astype(jnp.float32)
        )
    exact_jacobian = jacobian_operator is None
    jacobian_source = operator if exact_jacobian else jacobian_operator

    def step(state: ImplicitState) -> tuple[ImplicitState, Array]:
        U, cfl = state.U, state.cfl
        if mixed_precision:
            U, cfl = U.astype(jnp.float64), cfl.astype(jnp.float64)

        # One linearization serves the whole Krylov solve; the branch is
        # resolved at trace time, not in the loop.
        if exact_jacobian:
            residual, jvp = jax.linearize(operator.residual, U)
        else:
            residual = operator.residual(U)
            _, jvp = jax.linearize(jacobian_source.residual, U)
        norm = operator.norm(residual)

        lambdas = operator.wave_speeds(U)
        volume_over_dt = cell_wave_speed_sum(lambdas) / cfl

        def matvec(dU: Array) -> Array:
            """Action of ``V/dt I + dR/dU``."""
            return volume_over_dt[None] * dU + jvp(dU)

        # Right preconditioning, so GMRES minimizes the true linear residual
        # that the ramp below is judged on (solvers/krylov.py).
        def solve(prepare, state_u, speeds, step_cfl, *, basis_dtype=None):
            return gmres_right(
                matvec, -residual, prepare(state_u, speeds, step_cfl),
                restart=config.krylov_subspace, maxiter=config.krylov_restarts,
                tol=config.krylov_tol, basis_dtype=basis_dtype, return_action=True,
            )

        linear_fp64_fallback = jnp.asarray(False)
        if mixed_precision:
            dU, solved_action = solve(prepare_low, U.astype(jnp.float32),
                       tuple(lam.astype(jnp.float32) for lam in lambdas),
                       cfl.astype(jnp.float32), basis_dtype=jnp.float32)
            initial_size = l2_norm(residual, dtype=jnp.float64)
            solved_size = l2_norm(solved_action + residual, dtype=jnp.float64)
            # Require only a measurable decrease, not the requested Krylov
            # tolerance: a useful inexact Newton direction need not meet it.
            # The margin rejects FP32 stagnation without demanding a large gain.
            linear_fp64_fallback = (
                ~jnp.all(jnp.isfinite(dU)) | ~jnp.isfinite(solved_size)
                | ((initial_size > 0) & (solved_size >= (1. - 1e-8) * initial_size))
            )
            # A per-case cond becomes an eager select under vmap, executing
            # both solves even for a one-case campaign. A bounded while skips
            # FP64 factorization when no case needs it; vmap masks inactive
            # cases if another case does require the fallback.
            _, (dU, solved_action) = jax.lax.while_loop(
                lambda carry: linear_fp64_fallback & (carry[0] < 1),
                lambda carry: (carry[0] + 1, solve(prepare_preconditioner, U, lambdas, cfl)),
                (jnp.asarray(0, jnp.int32), (dU, solved_action)),
            )
        else:
            dU, solved_action = solve(prepare_preconditioner, U, lambdas, cfl)

        # Bound how far an accepted step may go (ImplicitConfig.max_relative_update).
        dU, omega = relax(dU, U, gamma, config.max_relative_update, operator.components)

        # The ramp keys off how well the linear system was solved (one more
        # matvec), not the residual history: a stagnated GMRES returns a null
        # update that leaves the residual unchanged, which looks like success.
        # Keep the scalar decision in double: casting both norms to FP32
        # first can turn a real decrease into exactly 1 and reject the step.
        # An amortized correction requires a fresh action to preserve rounding
        # and acceptance decisions; the unmodified direction can reuse GMRES.
        action = matvec(dU) if config.max_relative_update is not None else solved_action
        residual_size = jnp.maximum(l2_norm(residual, dtype=jnp.float64), floor_like(residual))

        def linear_ratio(factor):
            return l2_norm(factor * action + residual, dtype=jnp.float64) / residual_size

        correction_finite = jnp.all(jnp.isfinite(dU))
        growth_limit = config.nonlinear_growth_limit

        def merit(value):
            if config.nonlinear_merit == "equations":
                return jnp.max(operator.equation_norms_from_residual(value)).astype(jnp.float64)
            return operator.norm(value).astype(jnp.float64)

        baseline = merit(residual) if growth_limit is not None else jnp.asarray(1., jnp.float64)

        def evaluate(factor):
            candidate = U + factor * dU
            physical = jnp.all(is_physical(candidate, gamma, operator.components)) & correction_finite
            reduction = linear_ratio(factor)
            accept = physical & (reduction < 1.)
            ratio = jnp.asarray(jnp.nan, jnp.float64)
            if growth_limit is not None:
                nonlinear = jax.lax.cond(
                    physical, lambda _: merit(operator.residual(candidate)),
                    lambda _: jnp.asarray(jnp.inf, jnp.float64), None)
                ratio = nonlinear / jnp.maximum(baseline, floor_like(residual))
                accept = accept & jnp.isfinite(nonlinear) & (ratio <= growth_limit)
            return accept, physical, reduction, ratio

        factor = jnp.ones((), U.dtype)
        accept, physical, linear_reduction, nonlinear_ratio = evaluate(factor)
        backtracks = jnp.asarray(0, jnp.int32)
        if config.backtracking_steps:
            def retry(carry):
                k, factor, *_ = carry
                factor = factor * jnp.asarray(.5, U.dtype)
                return (k + 1, factor, *evaluate(factor))

            backtracks, factor, accept, physical, linear_reduction, nonlinear_ratio = jax.lax.while_loop(
                lambda c: (c[0] < config.backtracking_steps) & ~c[2], retry,
                (backtracks, factor, accept, physical, linear_reduction, nonlinear_ratio))
        candidate = U + factor * dU
        omega = omega * factor

        # Switched evolution relaxation: grow the CFL by the residual reduction
        # achieved, capped, and only when the Krylov solve earned it. Mulder and
        # van Leer, J. Comput. Phys. 59 (1985).
        reduction = state.residual_norm / jnp.maximum(norm, floor_like(norm))
        cfl_next = jnp.where(
            accept & (linear_reduction < config.krylov_success) & (omega >= 1.0),
            cfl * jnp.clip(reduction, 1.0, config.cfl_growth),
            jnp.where(accept, cfl, cfl / config.cfl_cutback),
        )

        updated = ImplicitState(
            U=jnp.where(accept, candidate, U),
            cfl=jnp.clip(cfl_next, config.cfl_min, config.cfl_max),
            residual_norm=jnp.where(accept, norm, state.residual_norm),
        )
        if diagnostics:
            mask = operator.atlas.interior
            rho, _, pressure = conservative_to_primitive(candidate, gamma, operator.components)
            return updated, norm, NewtonDiagnostics(
                accept, correction_finite, physical, linear_reduction, omega, updated.cfl,
                accept & jnp.any(candidate != U),
                jnp.min(jnp.where(mask, rho, jnp.inf)),
                jnp.min(jnp.where(mask, pressure, jnp.inf)), nonlinear_ratio, backtracks,
                linear_fp64_fallback)
        return updated, norm

    return step
