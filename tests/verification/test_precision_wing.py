"""FP32 must advance the actual thin-wall ICMS wing, not only coarse fixtures."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.campaign import _case, load_campaign
from jaxrans.core.state import is_physical
from jaxrans.geometry.icms import read_icms, write_hug
from jaxrans.post.forces import integrate_forces
from jaxrans.simulation import make_stepper


@pytest.mark.slow
def test_fp32_wing_section_advances_through_first_monitoring_block(tmp_path):
    root = Path(__file__).resolve().parents[2]
    campaign = load_campaign(root / "examples/wing_sections", ("case.general.precision=fp32",))
    section = read_icms(campaign.directory / campaign.sections["file"], 10, "m")[0]
    hug = tmp_path / "profile.hug"
    write_hug(section, hug)
    case = _case(campaign, hug, section, .3, -2.)
    grid = case.grid()
    assert grid.cells == 70592
    step, initial = make_stepper(case, grid)

    def advance(state, _):
        updated, residual = step(state)
        return updated, (residual, jnp.any(updated.U != state.U))

    state, (residuals, accepted) = jax.jit(
        lambda s: jax.lax.scan(advance, s, None, length=20))(initial)
    assert state.U.dtype == jnp.float32
    assert np.isfinite(state.U).all() and np.isfinite(residuals).all()
    assert np.asarray(is_physical(state.U, case.flow.gamma, grid.components)).all()
    assert np.asarray(accepted).all(), "the initial block must accept its Newton corrections"
    assert not np.all(np.asarray(residuals) == float(residuals[0]))
    forces = integrate_forces(state.U, grid, case.flow, case.grid_spec.force_reference())
    assert np.isfinite([forces.cl, forces.cd, forces.cm]).all()
