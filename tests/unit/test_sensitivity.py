"""Sensitivities: the traceable force coefficients, and the tangent solve
against dense Jacobians.

The derivative formulas hold at any state, converged or not (they are the
linear algebra of the implicit function theorem), so a few implicit steps on
a coarse grid make a non-trivial state small enough for ``jax.jacfwd``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from jaxrans import load_case, run, sensitivity
from jaxrans.core.state import freestream_primitive
from jaxrans.derivatives import flow_inputs, steady_problem
from jaxrans.post.forces import force_coefficients, integrate_forces

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
COARSE = ("case.domain.cells_airfoil=32", "case.domain.cells_normal=16", "case.domain.far_field=20.0",
          "case.domain.wall_spacing=0.001", "numerical.time_integration.solver=implicit",
          "numerical.convergence.max_iters=10", "numerical.convergence.block_iters=10")
OUTPUTS = ("cl", "cd", "cm")


@pytest.fixture(scope="module")
def laminar():
    """Laminar NACA 0012 on a 64 x 16 C-grid, ten implicit steps from freestream."""
    return run(load_case(EXAMPLES / "naca0012_laminar", COARSE + ("case.domain.cells_wake=16",)))


def dense_derivatives(result, parameter: str) -> dict[str, float]:
    """``dJ/dp|_U - dJ/dU (dR/dU)^-1 dR/dp`` with every Jacobian dense."""
    residual, outputs, p = steady_problem(result, OUTPUTS)
    U = result.U
    A = jax.jacfwd(lambda u: residual(u.reshape(U.shape), p).ravel())(U.ravel())
    dR_dp = jax.jacfwd(lambda q: residual(U, q))(p)[parameter]
    dJ_dU = jax.jacfwd(lambda u: outputs(u, p))(U)
    dJ_dp = jax.jacfwd(lambda q: outputs(U, q))(p)
    dU = -np.linalg.solve(np.asarray(A), np.asarray(dR_dp).ravel())
    return {out: float(dJ_dp[out][parameter] + np.asarray(dJ_dU[out]).ravel() @ dU) for out in OUTPUTS}


def test_flow_inputs_at_the_case_values_are_the_freestream(laminar) -> None:
    _, _, p = steady_problem(laminar, OUTPUTS)
    q, mu_inf = flow_inputs(laminar.case.flow, 2, p)
    assert np.allclose(q, freestream_primitive(laminar.case.flow), rtol=1e-15, atol=1e-16)
    assert float(mu_inf) == pytest.approx(laminar.case.flow.mu_inf, rel=1e-15)


def test_traceable_coefficients_are_the_integrated_forces(laminar) -> None:
    case = laminar.case
    forces = integrate_forces(laminar.U, laminar.grid, case.flow, case.grid_spec.force_reference())
    c = jax.jit(lambda U: force_coefficients(U, laminar.grid, case.flow, case.grid_spec.force_reference()))(laminar.U)
    for name in ("cl", "cd", "cm", "cd_friction"):
        assert float(c[name]) == pytest.approx(getattr(forces, name), rel=1e-13, abs=1e-15)


@pytest.mark.slow   # ~6 min on a CPU: two dense 5440 x 5440 Jacobians
def test_tangent_matches_the_dense_solve(laminar) -> None:
    s = sensitivity(laminar, OUTPUTS, ("alpha", "mach"), tol=1e-12)
    assert max(s.linear_residuals.values()) < 1e-12
    for parameter in ("alpha", "mach"):
        dense = dense_derivatives(laminar, parameter)
        for out in OUTPUTS:
            assert s.derivatives[out][parameter] == pytest.approx(dense[out], rel=1e-8), (parameter, out)


def test_unknown_names_are_refused(laminar) -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        sensitivity(laminar, parameters=("reynolds",))
    with pytest.raises(ValueError, match="unknown output"):
        sensitivity(laminar, outputs=("lift",))
