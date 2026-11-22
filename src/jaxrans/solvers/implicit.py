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
    linear_stagnated: Array


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


LOCAL_RAMP_FRACTION = 0.01
"""``relaxation = "local"``: the CFL may still grow while at most this
fraction of block cells has its correction cut."""


def _ratios(dU: Array, U: Array, gamma: float, dim: int):
    """Per cell: ``(|d rho| / rho, |dp| / p)`` and, with SA, ``|d nu~| / max(|nu~|, 1)``
    (``relax``'s definitions, the pressure change linearized)."""
    rows = Rows(dim)
    rho, velocity, p = conservative_to_primitive(U, gamma, dim)
    d_rho = dU[RHO]
    d_e = dU[rows.energy]
    for k in range(dim):
        d_e = d_e - velocity[k] * dU[1 + k]
    d_p = (gamma - 1.0) * (d_e + 0.5 * dot(velocity, velocity) * d_rho)
    flow = [jnp.abs(d_rho) / rho, jnp.abs(d_p) / p]
    sa = None
    if U.shape[0] > rows.n_flow:
        nu_hat = U[rows.nu] / rho
        d_nu_hat = (dU[rows.nu] - nu_hat * d_rho) / rho
        sa = jnp.abs(d_nu_hat) / jnp.maximum(jnp.abs(nu_hat), 1.0)
    return flow, sa, rho


def relax_correction(dU: Array, U: Array, gamma: float, fraction: float | None, dim: int,
                     mode: str = "global", interior: Array | None = None):
    """``max_relative_update`` applied as ``mode`` says; returns ``(dU,
    omega, ramp_ok, scaled)``: the bounded correction, the damping reported
    and used in the acceptance margin, whether the CFL may grow, and whether
    ``dU`` is the input times ``omega`` (else its operator action must be
    evaluated again).

    ``global``: ``relax``, one factor for everything; a handful of cells can
    hold the whole field (and the CFL ramp) back. ``split``: the mean-flow
    rows get the factor of density and pressure, the SA row its own, from
    the change of ``rho nu~`` over ``rho max(|nu~|, 1)``; the CFL keys off the
    mean-flow factor. ``local``: every cell its own factor, the same bound as
    ``global`` met cell by cell; the CFL may grow while at most
    ``LOCAL_RAMP_FRACTION`` of block cells are cut.
    """
    one = jnp.ones((), dU.dtype)
    if fraction is None:
        return dU, one, jnp.asarray(True), True
    if mode == "global":
        dU, omega = relax(dU, U, gamma, fraction, dim)
        return dU, omega, omega >= 1.0, True
    flow, sa, rho = _ratios(dU, U, gamma, dim)
    if mode == "split":
        worst = jnp.maximum(jnp.max(flow[0]), jnp.max(flow[1]))
        omega = jnp.minimum(one, fraction / jnp.maximum(worst, floor_like(worst)))
        if sa is None:
            return omega * dU, omega, omega >= 1.0, True
        rows = Rows(dim)
        nu_hat = U[rows.nu] / rho
        own = jnp.max(jnp.abs(dU[rows.nu]) / (rho * jnp.maximum(jnp.abs(nu_hat), 1.0)))
        omega_sa = jnp.minimum(one, fraction / jnp.maximum(own, floor_like(own)))
        scale = jnp.concatenate([jnp.full((rows.n_flow,), omega, dU.dtype), omega_sa[None].astype(dU.dtype)])
        return scale.reshape((-1,) + (1,) * (dU.ndim - 1)) * dU, omega, omega >= 1.0, False
    if mode == "local":
        worst = jnp.maximum(flow[0], flow[1]) if sa is None else jnp.maximum(jnp.maximum(flow[0], flow[1]), sa)
        factor = jnp.minimum(one, fraction / jnp.maximum(worst, floor_like(worst)))
        cut = factor < 1.0
        if interior is not None:
            cut, counted = cut & interior, jnp.sum(interior)
            omega = jnp.sum(jnp.where(interior, factor, 0.0)) / counted
        else:
            counted, omega = cut.size, jnp.mean(factor)
        ramp_ok = jnp.sum(cut) <= LOCAL_RAMP_FRACTION * counted
        return factor[None] * dU, omega.astype(dU.dtype), ramp_ok, False
    raise ValueError(f"unknown relaxation {mode!r}")


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
    # A sectional canvas is solved on its independent plane (numerics/
    # residual.py ``SectionalPlane``): residual, Jacobian products, Krylov
    # vectors and line solves on plane fields, the plane written back.
    plane = getattr(operator, "plane", None) if getattr(jacobian_source, "plane", None) is not None else None
    if plane is not None:
        residual_of, jacobian_of, norm_of = plane.residual, jacobian_source.plane.residual, plane.norm
        interior, solve_options = plane.interior, {"plane": plane.k}
    else:
        residual_of, jacobian_of, norm_of = operator.residual, jacobian_source.residual, operator.norm
        interior, solve_options = operator.atlas.interior, {}

    def step(state: ImplicitState) -> tuple[ImplicitState, Array]:
        U, cfl = state.U, state.cfl
        if mixed_precision:
            U, cfl = U.astype(jnp.float64), cfl.astype(jnp.float64)
        canvas_U = U                                   # the preconditioner's blocks use the canvas
        canvas_lambdas = operator.wave_speeds(U)
        if plane is not None:
            U, lambdas = plane.take(U), plane.faces(canvas_lambdas)
        else:
            lambdas = canvas_lambdas

        # One linearization serves the whole Krylov solve; the branch is
        # resolved at trace time, not in the loop.
        if exact_jacobian:
            residual, jvp = jax.linearize(residual_of, U)
        else:
            residual = residual_of(U)
            _, jvp = jax.linearize(jacobian_of, U)
        norm = norm_of(residual)

        volume_over_dt = cell_wave_speed_sum(lambdas) / cfl

        def matvec(dU: Array) -> Array:
            """Action of ``V/dt I + dR/dU``."""
            return volume_over_dt[None] * dU + jvp(dU)

        # Right preconditioning, so GMRES minimizes the true linear residual
        # that the ramp below is judged on (solvers/krylov.py).
        def solve(prepare, state_u, speeds, step_cfl, *, basis_dtype=None):
            return gmres_right(
                matvec, -residual, prepare(state_u, speeds, step_cfl, **solve_options),
                restart=config.krylov_subspace, maxiter=config.krylov_restarts,
                tol=config.krylov_tol, basis_dtype=basis_dtype, return_action=True,
                # FP32 Krylov vectors in a mixed solve keep the true action:
                # it is what detects their stagnation (the FP64 fallback below).
                action_from_arnoldi=basis_dtype is None,
                early_exit=config.krylov_early_exit,
            )

        linear_fp64_fallback = jnp.asarray(False)
        if mixed_precision:
            dU, solved_action = solve(prepare_low, canvas_U.astype(jnp.float32),
                       tuple(lam.astype(jnp.float32) for lam in canvas_lambdas),
                       cfl.astype(jnp.float32), basis_dtype=jnp.float32)
            initial_size = l2_norm(residual, dtype=jnp.float64)
            solved_size = l2_norm(solved_action + residual, dtype=jnp.float64)
            # Require only a measurable decrease, not the requested Krylov
            # tolerance: a useful inexact Newton direction need not meet it.
            # The margin rejects FP32 stagnation without demanding a large gain.
            linear_fp64_fallback = (
                ~jnp.all(jnp.isfinite(dU)) | ~jnp.isfinite(solved_size)
                | ((initial_size > 0) &
                   (solved_size >= (1. - max(1e-8, config.krylov_min_progress)) * initial_size))
            )
            # A per-case cond becomes an eager select under vmap, executing
            # both solves even for a one-case campaign. A bounded while skips
            # FP64 factorization when no case needs it; vmap masks inactive
            # cases if another case does require the fallback.
            _, (dU, solved_action) = jax.lax.while_loop(
                lambda carry: linear_fp64_fallback & (carry[0] < 1),
                lambda carry: (carry[0] + 1, solve(prepare_preconditioner, canvas_U, canvas_lambdas, cfl)),
                (jnp.asarray(0, jnp.int32), (dU, solved_action)),
            )
        else:
            dU, solved_action = solve(prepare_preconditioner, canvas_U, canvas_lambdas, cfl)

        # Bound how far an accepted step may go (ImplicitConfig.max_relative_update).
        dU, omega, ramp_ok, scaled = relax_correction(
            dU, U, gamma, config.max_relative_update, operator.components, config.relaxation,
            interior)

        # The ramp keys off how well the linear system was solved, not the
        # residual history: a stagnated GMRES returns a null
        # update that leaves the residual unchanged, which looks like success.
        # Keep the scalar decision in double: casting both norms to FP32
        # first can turn a real decrease into exactly 1 and reject the step.
        # The relaxation only scales the direction, and the operator is linear:
        # the action of omega dU is omega times GMRES's action of dU.
        action = omega * solved_action if scaled else matvec(dU)
        residual_size = jnp.maximum(l2_norm(residual, dtype=jnp.float64), floor_like(residual))

        def linear_ratio(factor):
            return l2_norm(factor * action + residual, dtype=jnp.float64) / residual_size

        correction_finite = jnp.all(jnp.isfinite(dU))
        growth_limit = config.nonlinear_growth_limit

        def merit(value):
            if config.nonlinear_merit == "equations":
                if plane is not None:             # the other planes are no block cells
                    value = plane.put(jnp.zeros_like(canvas_U), value)
                return jnp.max(operator.equation_norms_from_residual(value)).astype(jnp.float64)
            return norm_of(value).astype(jnp.float64)

        baseline = merit(residual) if growth_limit is not None else jnp.asarray(1., jnp.float64)

        def evaluate(factor):
            candidate = U + factor * dU
            physical = jnp.all(is_physical(candidate, gamma, operator.components)) & correction_finite
            reduction = linear_ratio(factor)
            # Scale the required gain with damping: a useful direction may
            # legitimately need a tiny physical step. Undamped near-null
            # solves (including failed FP64 retries) must instead cut CFL.
            required_gain = (jnp.asarray(config.krylov_min_progress, jnp.float64)
                             * omega.astype(jnp.float64) * factor)
            accept = physical & (reduction < 1. - required_gain)
            ratio = jnp.asarray(jnp.nan, jnp.float64)
            if growth_limit is not None:
                nonlinear = jax.lax.cond(
                    physical, lambda _: merit(residual_of(candidate)),
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
            accept & (linear_reduction < config.krylov_success) & ramp_ok & (factor >= 1.0),
            cfl * jnp.clip(reduction, 1.0, config.cfl_growth),
            jnp.where(accept, cfl, cfl / config.cfl_cutback),
        )

        kept = jnp.where(accept, candidate, U)
        updated = ImplicitState(
            U=kept if plane is None else plane.put(canvas_U, kept),
            cfl=jnp.clip(cfl_next, config.cfl_min, config.cfl_max),
            residual_norm=jnp.where(accept, norm, state.residual_norm),
        )
        if diagnostics:
            mask = interior
            rho, _, pressure = conservative_to_primitive(candidate, gamma, operator.components)
            return updated, norm, NewtonDiagnostics(
                accept, correction_finite, physical, linear_reduction, omega, updated.cfl,
                accept & jnp.any(candidate != U),
                jnp.min(jnp.where(mask, rho, jnp.inf)),
                jnp.min(jnp.where(mask, pressure, jnp.inf)), nonlinear_ratio, backtracks,
                linear_fp64_fallback,
                jnp.isfinite(linear_reduction) &
                (linear_reduction >= 1. - config.krylov_min_progress * omega.astype(jnp.float64)))
        return updated, norm

    return step
