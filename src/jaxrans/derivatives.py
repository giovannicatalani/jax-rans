"""Derivatives of a converged result's force coefficients with respect to
the flow parameters.

    from jaxrans import load_case, run, sensitivity

    result = run(load_case("examples/naca0012_laminar", ("numerical.convergence.forces_tol=null",)))
    s = sensitivity(result, outputs=("cl", "cd"), parameters=("alpha", "mach"))
    s.derivatives["cl"]["alpha"]            # dCL/dalpha, per degree

The parameters enter through the freestream: ``alpha`` (degrees) turns its
velocity, ``mach`` scales it and, when viscous, ``mu_inf = M / Re`` with it
(the Reynolds number held). The residual is the case's own, at the result's
state; one linear solve per parameter (``solvers/sensitivity.py``),
iterated with the case's own implicit Newton step. It is the derivative of
the discrete solution, which exists where the discretization is
differentiable (``minmod`` and ``van_albada`` have kinks: use ``van_albada_smooth``
or no limiter) and
which the result approximates as well as it is converged (to round-off for
derivatives to many digits).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from .core.config import FlowConditions
from .core.state import freestream_vector
from .numerics.residual import cell_wave_speed_sum, make_residual
from .post.forces import force_coefficients
from .simulation import Result
from .solvers.krylov import gmres_right
from .solvers.preconditioners import get_preconditioner
from .solvers.sensitivity import tangent

PARAMETERS: tuple[str, ...] = ("alpha", "mach")
OUTPUTS: tuple[str, ...] = ("cl", "cd", "cm", "cd_friction")


@dataclass(frozen=True)
class Sensitivity:
    """``derivatives[output][parameter]`` (per degree for ``alpha``);
    ``linear_residuals``: each parameter's linear solve, relative residual;
    ``seconds``: the solves' wall clock, ``compile_seconds`` before it."""

    derivatives: dict[str, dict[str, float]]
    linear_residuals: dict[str, float]
    seconds: float
    compile_seconds: float

    def summary(self) -> str:
        lines = [f"d{out.upper()}/d{name} = {value:+.8e}"
                 for out, row in self.derivatives.items() for name, value in row.items()]
        worst = max(self.linear_residuals.values())
        return "\n".join(lines + [f"{len(self.linear_residuals)} linear solve(s), "
                                  f"largest relative residual {worst:.1e}, {self.seconds:.1f} s "
                                  f"(+ {self.compile_seconds:.1f} s compilation)"])


def flow_inputs(flow: FlowConditions, dim: int, p: dict) -> tuple[Array, Array | None]:
    """Freestream primitives ``(d+2,)`` and ``mu_inf`` (``None`` if inviscid)
    at parameters ``p`` (``alpha`` in degrees, ``mach``), traceable."""
    q = freestream_vector(p["mach"], jnp.deg2rad(p["alpha"]), flow.gamma, dim, xp=jnp)
    return q, None if flow.inviscid else p["mach"] / flow.reynolds


def steady_problem(result: Result, outputs: tuple[str, ...]) -> tuple[Callable, Callable, dict]:
    """``(residual(U, p), coefficients(U, p), p)``: the case's residual and
    its force coefficients ``outputs`` as functions of the state and of the
    parameters ``p`` (``PARAMETERS``, at the case's values)."""
    case, atlas = result.case, result.grid
    flow, numerics = case.flow, case.numerics
    reference = case.grid_spec.force_reference()
    distance = make_residual(atlas, flow, numerics).distance

    def residual(U: Array, p: dict) -> Array:
        q, mu_inf = flow_inputs(flow, atlas.dim, p)
        return make_residual(atlas, flow, numerics, freestream=q, mu_inf=mu_inf, distance=distance).residual(U)

    def coefficients(U: Array, p: dict) -> dict:
        c = force_coefficients(U, atlas, flow, reference, *flow_inputs(flow, atlas.dim, p))
        return {name: c[name] for name in outputs}

    p = {"alpha": jnp.asarray(flow.alpha_deg, numerics.dtype), "mach": jnp.asarray(flow.mach, numerics.dtype)}
    return residual, coefficients, p


def newton_inverse(result: Result) -> Callable[[Array], Callable[[Array], Array]]:
    """``U -> (r -> G r)``, ``G`` the linear solve of the case's implicit
    Newton step (solvers/implicit.py) at the end of its CFL ramp, frozen at
    ``U``: GMRES on ``V/dt + dR~/dU`` (``R~`` the first-order residual unless
    the case's ``jacobian`` is ``exact``), right-preconditioned, with the
    case's Krylov settings."""
    case, atlas = result.case, result.grid
    flow, numerics, implicit = case.flow, case.numerics, case.implicit
    operator = make_residual(atlas, flow, numerics)
    approximate = operator if implicit.jacobian == "exact" else make_residual(
        atlas, flow, numerics, first_order=True, distance=operator.distance)
    prepare = get_preconditioner(implicit.preconditioner)(
        atlas, flow.gamma, operator.viscosity, operator.model, operator.distance)

    def at(U: Array) -> Callable[[Array], Array]:
        _, jacobian = jax.linearize(approximate.residual, U)
        lambdas = operator.wave_speeds(U)
        volume_over_dt = cell_wave_speed_sum(lambdas) / implicit.cfl_max

        def matvec(v: Array) -> Array:
            return volume_over_dt[None] * v + jacobian(v)

        precondition = prepare(U, lambdas, implicit.cfl_max)
        return lambda r: gmres_right(matvec, r, precondition, implicit.krylov_subspace,
                                     implicit.krylov_restarts, implicit.krylov_tol)

    return at


def sensitivity(result: Result, outputs: tuple[str, ...] = ("cl", "cd"),
                parameters: tuple[str, ...] = ("alpha",), tol: float = 1e-10,
                maxiter: int = 10_000) -> Sensitivity:
    """Derivatives of ``outputs`` (``OUTPUTS``) with respect to ``parameters``
    (``PARAMETERS``) at ``result``'s state, one linear solve per parameter:
    defect correction with ``newton_inverse`` to ``tol`` relative, at most
    ``maxiter`` steps (solvers/sensitivity.py)."""
    for kind, given, known in (("output", outputs, OUTPUTS), ("parameter", parameters, PARAMETERS)):
        unknown = [g for g in given if g not in known]
        if unknown or not given:
            raise ValueError(f"unknown {kind}(s) {unknown or given}; available: {known}")
    residual, coefficients, p = steady_problem(result, tuple(outputs))
    inverse_at = newton_inverse(result)

    @jax.jit
    def compute(U: Array):
        return tangent(residual, coefficients, U, p, tuple(parameters), inverse_at(U), tol, maxiter)

    started = time.perf_counter()
    compiled = compute.lower(result.U).compile()
    compiled_at = time.perf_counter()
    derivatives, errors = jax.device_get(compiled(result.U))
    return Sensitivity({out: {name: float(v) for name, v in row.items()} for out, row in derivatives.items()},
                       {k: float(v) for k, v in errors.items()},
                       time.perf_counter() - compiled_at, compiled_at - started)
