"""Bilateral taper continuity and independent Jacobian checks.

These are operator tests on a fixed streamwise profile, not a claim of
converged-flow or experimental validation. Existing tests mostly approach
zero from positive taper on a fixed normal section.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.case import parse_case
from jaxrans.numerics.residual import make_residual
from jaxrans.post.forces import force_coefficients
from tests.unit.test_sectional import document


def build(taper, physics, sweep=30.):
    doc = document(taper_rate=taper, leading_edge_sweep_deg=sweep)
    doc['domain']['section'] = 'streamwise'
    if physics != 'euler':
        doc['freestream']['reynolds'] = 1e5
    if physics in ('sa_noft2', 'sa_noft2_qcr'):
        doc['turbulence'] = {'model': physics}
    case = parse_case(doc, {'sectional': {'depth': .1},
                            'convective_fluxes': {'limiter': 'none'}})
    return make_residual(case.grid(), case.flow, case.numerics), case.flow


def state(op, sweep):
    # Same conservative perturbation in aircraft axes at corresponding source
    # indices, rotated into each operator's changing mean-sweep frame.
    U = np.asarray(op.initial_state()).copy()
    angle = np.deg2rad(sweep)
    mx, mz = U[1].copy(), U[3].copy()
    U[1], U[3] = np.cos(angle)*mx + np.sin(angle)*mz, -np.sin(angle)*mx + np.cos(angle)*mz
    U *= 1 + .01*np.random.default_rng(187).normal(size=U.shape)
    mx, mz = U[1].copy(), U[3].copy()
    U[1], U[3] = np.cos(angle)*mx - np.sin(angle)*mz, np.sin(angle)*mx + np.cos(angle)*mz
    return jnp.asarray(U)


def observables(op, flow):
    U = state(op, flow.sweep_deg)
    R = np.asarray(op.residual(U) / op.grid.volume)[:, op.atlas.interior].copy()
    angle = flow.sweep_rad
    mx, mz = R[1].copy(), R[3].copy()
    R[1], R[3] = np.cos(angle)*mx + np.sin(angle)*mz, -np.sin(angle)*mx + np.cos(angle)*mz
    loads = force_coefficients(U, op.atlas, flow)
    return R, np.array([float(loads[k]) for k in ('cl', 'cd', 'cm', 'cy')])


@pytest.mark.parametrize('physics', ['euler', 'laminar', 'sa_noft2', 'sa_noft2_qcr'])
def test_bilateral_streamwise_taper_limit_and_matching_one_sided_slopes(physics):
    base = observables(*build(0., physics))
    errors, slope_jumps = [], []
    for h in (1e-3, 1e-4, 1e-5):
        plus = observables(*build(h, physics))
        minus = observables(*build(-h, physics))
        errors.append([max(np.linalg.norm(p-b), np.linalg.norm(m-b))
                       for p, m, b in zip(plus, minus, base)])
        # A discontinuity or unequal limiting one-sided derivatives survives
        # this division by h; a smooth response instead decreases as O(h).
        slope_jumps.append([np.linalg.norm((p-b)/h - (b-m)/h)
                            for p, m, b in zip(plus, minus, base)])
    errors, slope_jumps = np.array(errors), np.array(slope_jumps)
    assert np.all(np.isfinite(errors))
    assert np.all(errors[:-1] > 7*errors[1:]), errors
    # Pressure loads can be exactly affine in taper for this prescribed
    # field. Permit only the O(machine epsilon / h) differencing floor.
    floor = np.array([max(1., np.linalg.norm(b)) for b in base]) * 1e-8
    assert np.all((slope_jumps[0] > 5*slope_jumps[1]) | (slope_jumps[1] < floor)), slope_jumps
    assert np.all((slope_jumps[1] > 3*slope_jumps[2]) | (slope_jumps[2] < floor)), slope_jumps


@pytest.mark.parametrize('taper', [-.2, 0., .2])
@pytest.mark.parametrize('physics', ['euler', 'sa_noft2_qcr'])
def test_reduced_residual_jvp_matches_independent_central_difference(taper, physics):
    op, flow = build(taper, physics)
    U = state(op, flow.sweep_deg)
    # Only independent cells vary: ghost closure must participate in dR/dU.
    direction = jnp.asarray(np.random.default_rng(42).normal(size=U.shape)) * .01
    direction = jnp.where(op.atlas.interior, direction, 0.)
    _, tangent = jax.jvp(op.residual, (U,), (direction,))
    exact = np.asarray(tangent / op.grid.volume)[:, op.atlas.interior]
    errors = []
    for eps in (1e-3, 1e-4, 1e-5):
        fd = (op.residual(U + eps*direction) - op.residual(U - eps*direction))/(2*eps)
        fd = np.asarray(fd / op.grid.volume)[:, op.atlas.interior]
        errors.append(np.linalg.norm(fd-exact)/np.linalg.norm(exact))
    assert np.isfinite(errors).all()
    assert min(errors) < 2e-7, errors
