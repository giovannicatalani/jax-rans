"""Trace the RAE campaign's FP32 Krylov path without running a full campaign."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans import load_case
from jaxrans.campaign import _geometry, _with_geometry
from jaxrans.numerics.residual import make_residual
from jaxrans.simulation import make_stepper
from jaxrans.solvers.preconditioners import line_implicit_j


ROOT = Path(__file__).resolve().parents[2]


def rae_case(sweep, jacobian="first_order"):
    return load_case(ROOT / "examples/rae2822_sensitivity", (
        "case.general.precision=fp32", "case.freestream.mach=0.1",
        "case.freestream.alpha_deg=-3", f"case.freestream.sweep_deg={sweep}",
        "case.domain.cells_airfoil=64", "case.domain.cells_wake=16",
        "case.domain.cells_normal=24", "case.domain.wall_spacing=0.001",
        "case.domain.far_field=10", "case.domain.leading_edge_spacing=0.01",
        "case.domain.trailing_edge_spacing=0.02",
        f"numerical.implicit.jacobian={jacobian}",
        "numerical.implicit.krylov_subspace=4", "numerical.implicit.krylov_restarts=1"))


@pytest.mark.parametrize("sweep", ["null", "0.0", "12.0"])
def test_fp32_rae_preconditioned_tangent_with_double_geometry(sweep):
    """A preconditioned field must be accepted by the real residual's JVP.

    Keep source geometry and wall distance in double, as a caller supplying
    geometry to a batched kernel may do. The preconditioner must round its
    own metrics before factorization, without changing the source atlas.
    """
    c = rae_case(sweep)
    atlas = c.grid().with_dtype(jnp.float64)
    op = make_residual(atlas, c.flow, c.numerics)
    U = op.initial_state()
    prepare = line_implicit_j(atlas, op.gamma, op.viscosity, op.model,
                              op.distance.astype(jnp.float64))

    def krylov_action(u, rhs):
        residual, jvp = jax.linearize(op.residual, u)
        z = prepare(u, op.wave_speeds(u), jnp.asarray(.1, u.dtype))(rhs)
        return residual, z, jvp(z)

    result = jax.eval_shape(krylov_action, U, jnp.ones_like(U))
    assert all(a.dtype == np.float32 and a.shape == U.shape for a in result)
    assert atlas.grid.volume.dtype == atlas.centres.dtype == np.float64


@pytest.mark.parametrize("sweep", ["null", "0.0", "12.0"])
@pytest.mark.parametrize("jacobian", ["exact", "first_order"])
def test_fp32_rae_campaign_step_traces_with_batched_geometry(sweep, jacobian):
    """Cover GMRES's first matvec and preconditioned matvec under campaign vmap.

    eval_shape traces the whole Newton step, including nested Krylov loops,
    without compiling or marching the production grid on a GPU.
    """
    c = rae_case(sweep, jacobian)
    template = c.grid()
    op = make_residual(template, c.flow, c.numerics)
    geometry = _geometry(template, op.distance)
    batch = jax.tree_util.tree_map(lambda a: jnp.stack([a, a]), geometry)
    q = jnp.stack([op.q_inf[:template.components + 2]] * 2)

    def one(freestream, geom):
        atlas = _with_geometry(template, geom)
        step, state = make_stepper(c, atlas, freestream=freestream, distance=geom["distance"])
        return step(state)

    state, norm = jax.eval_shape(jax.vmap(one), q, batch)
    assert state.U.shape == (2, template.components + 3, *template.shape)
    assert all(a.dtype == np.float32 for a in jax.tree_util.tree_leaves((state, norm)))
