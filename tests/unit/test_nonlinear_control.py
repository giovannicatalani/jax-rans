"""Nonlinear safeguards must judge the actual damped correction."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.core.config import ImplicitConfig
from jaxrans.solvers import implicit


def toy(monkeypatch, config, correction=3.):
    U = jnp.array([1., 0., 0., 1.]).reshape(4, 1, 1)

    def residual(u):
        return jnp.zeros_like(u).at[3].set(u[3] ** 2 - 4.)

    operator = SimpleNamespace(
        gamma=1.4, components=2, atlas=SimpleNamespace(interior=jnp.ones((1, 1), bool)),
        viscosity=None, model=None, distance=None, residual=residual,
        norm=lambda r: jnp.abs(r[0, 0, 0]),
        equation_norms_from_residual=lambda r: jnp.abs(r[:, 0, 0]),
        wave_speeds=lambda u: (jnp.ones((2, 1)) * .025, jnp.ones((1, 2)) * .025))
    dU = jnp.zeros_like(U).at[3].set(correction)
    def forced_solve(matvec, *args, **kwargs):
        return (dU, matvec(dU)) if kwargs.get("return_action") else dU

    monkeypatch.setattr(implicit, "gmres_right", forced_solve)
    state = implicit.ImplicitState(U, jnp.asarray(1.), jnp.asarray(3.))
    return jax.jit(implicit.make_newton_step(operator, config, diagnostics=True))(state)


def test_backtracking_recovers_a_bad_full_newton_correction(monkeypatch):
    config = ImplicitConfig(cfl=1., preconditioner="none", nonlinear_growth_limit=1.,
                            nonlinear_merit="equations", backtracking_steps=3)
    updated, _, report = toy(monkeypatch, config)
    assert bool(report.accepted)
    assert int(report.backtracks) == 1
    np.testing.assert_allclose(updated.U[3], 2.5)
    assert report.linear_reduction < 1.
    assert report.nonlinear_ratio <= 1.
    assert report.damping == .5
    assert updated.cfl == 1.  # A shortened correction does not earn a CFL increase.


def test_disabled_backtracking_preserves_rejection_and_cuts_cfl(monkeypatch):
    config = ImplicitConfig(cfl=1., preconditioner="none", nonlinear_growth_limit=1.)
    updated, _, report = toy(monkeypatch, config)
    assert not bool(report.accepted)
    assert int(report.backtracks) == 0
    np.testing.assert_allclose(updated.U[3], 1.)
    assert updated.cfl == config.cfl_min


def test_exhausted_backtracking_never_applies_a_rejected_state(monkeypatch):
    config = ImplicitConfig(cfl=1., preconditioner="none", nonlinear_growth_limit=1., backtracking_steps=1)
    updated, _, report = toy(monkeypatch, config, correction=100.)
    assert not bool(report.accepted)
    assert int(report.backtracks) == 1
    np.testing.assert_allclose(updated.U[3], 1.)


def test_equation_merit_rejects_growth_hidden_from_mass(monkeypatch):
    # Here the mass residual is exactly zero at both states. The energy defect
    # grows from 3 to 5.61 even though the linearized system improves.
    mass = ImplicitConfig(cfl=1., preconditioner="none", nonlinear_growth_limit=1.)
    _, _, mass_report = toy(monkeypatch, mass, correction=2.1)
    assert bool(mass_report.accepted)
    config = ImplicitConfig(cfl=1., preconditioner="none", nonlinear_growth_limit=1.,
                            nonlinear_merit="equations")
    updated, _, report = toy(monkeypatch, config, correction=2.1)
    assert report.linear_reduction < 1.
    assert report.nonlinear_ratio > 1.
    assert not bool(report.accepted)
    np.testing.assert_allclose(updated.U[3], 1.)


def test_near_null_correction_rejects_and_cuts_cfl(monkeypatch):
    config = ImplicitConfig(cfl=1., preconditioner="none")
    updated, _, report = toy(monkeypatch, config, correction=1e-11)
    assert not bool(report.accepted)
    assert bool(report.linear_stagnated)
    assert updated.cfl == config.cfl_min
    np.testing.assert_array_equal(updated.U[3], 1.)


def test_zero_progress_margin_restores_legacy_acceptance(monkeypatch):
    config = ImplicitConfig(cfl=1., preconditioner="none", krylov_min_progress=0.)
    updated, _, report = toy(monkeypatch, config, correction=1e-11)
    assert bool(report.accepted)
    assert not bool(report.linear_stagnated)
    assert float(updated.U[3, 0, 0]) > 1.


def test_useful_direction_with_tiny_physical_damping_remains_accepted(monkeypatch):
    config = ImplicitConfig(cfl=1., preconditioner="none", max_relative_update=1e-10)
    updated, _, report = toy(monkeypatch, config, correction=1.)
    assert bool(report.accepted)
    assert not bool(report.linear_stagnated)
    assert report.damping < 1e-6
    assert float(updated.U[3, 0, 0]) > 1.


@pytest.mark.parametrize("margin", [-1., 1., float("nan")])
def test_invalid_progress_margin_is_rejected(margin):
    with pytest.raises(ValueError, match="krylov_min_progress"):
        ImplicitConfig(krylov_min_progress=margin)
