"""Block diagnostics summarize device work without changing stopping decisions."""

from typing import NamedTuple
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.solvers.driver import ExitReason, run_to_steady
from jaxrans.io.output import ResultWriter


class Report(NamedTuple):
    accepted: object
    cfl: object


def test_diagnostics_are_reduced_per_block_and_trimmed_per_case():
    def step(x):
        updated = x * jnp.asarray([.5, .9])
        return updated, updated, Report(updated > 0, updated)

    state, history = run_to_steady(step, jnp.ones(2), max_iters=50, tol=.1,
                                   block_iters=2, active_mask=[True, True], diagnostics=True)
    assert history.case_iterations == (6, 24)
    first = history.case(0)
    assert len(first.block_diagnostics) == 3
    assert first.block_diagnostics[-1]["iteration"] == 6
    assert first.block_diagnostics[0]["accepted_count"] == 2
    np.testing.assert_allclose(first.block_diagnostics[0]["cfl_min"], .25)
    np.testing.assert_allclose(first.block_diagnostics[0]["cfl_max"], .5)
    np.testing.assert_allclose(state[0], .5**6)


def test_cfl_cutbacks_are_allowed_before_declaring_stall():
    def step(state):
        x, cfl = state
        return (x, jnp.maximum(cfl / 2, .5)), x

    _, history = run_to_steady(step, (jnp.asarray(1.), jnp.asarray(16.)),
                               max_iters=20, tol=0, block_iters=2,
                               control_changed=lambda old, new: old[1] != new[1])
    assert history.reason == ExitReason.STALLED
    assert history.n_iters == 8


def test_cfl_control_prevents_only_its_own_batched_stall():
    def step(state):
        x, cfl = state
        return (x, jnp.maximum(cfl / 2, .5)), x

    _, history = run_to_steady(step, (jnp.ones(2), jnp.array([16., .5])),
                               max_iters=20, tol=0, block_iters=2, active_mask=[True, True],
                               control_changed=lambda old, new: old[1] != new[1])
    assert history.case_iterations == (8, 2)


def test_constant_mass_residual_does_not_hide_progress_in_other_equations():
    class UpdateReport(NamedTuple):
        accepted: object
        changed: object

    def step(x):
        return x * .5, jnp.ones_like(x), UpdateReport(x > 0, x > 0)

    _, history = run_to_steady(step, jnp.ones(2), max_iters=6, tol=0, block_iters=2,
                               active_mask=[True, False], diagnostics=True)
    assert history.case_reasons == (ExitReason.MAX_ITERS, ExitReason.SKIPPED)
    for report in history.block_diagnostics:
        assert report["accepted_count"][1] == report["changed_count"][1] == 0


def test_saved_history_preserves_all_forces_and_stopping_evidence(tmp_path):
    writer = ResultWriter(tmp_path, "case")
    coefficients = np.array([[.2, .01, -.03], [.21, .012, -.031]])
    convergence = ({"thresholds": np.array([1e-8, 1e-7]), "settled": False},)
    diagnostics = ({"accepted_count": np.int64(2), "cfl_min": np.float32(.5)},)
    writer.write_solver_history(coefficients, convergence, diagnostics, 20)
    table = np.loadtxt(tmp_path / "case_forces.dat")
    np.testing.assert_array_equal(table[:, 0], [20, 40])
    np.testing.assert_allclose(table[:, 1:], coefficients)
    report = json.loads((tmp_path / "case_solver_history.json").read_text())
    assert report["newton_blocks"][0]["accepted_count"] == 2
    assert report["convergence"][0]["thresholds"] == [1e-8, 1e-7]


@pytest.mark.skipif(jax.local_device_count() < 2, reason="requires two devices")
def test_diagnostics_follow_the_case_axis_across_devices():
    def step(x):
        updated = x * .5
        return updated, updated, Report(updated > 0, updated)

    _, history = run_to_steady(step, jnp.arange(1., 5.), max_iters=20, tol=.1,
                               block_iters=2, active_mask=[True] * 4,
                               devices=jax.devices()[:2], diagnostics=True)
    assert history.case_iterations == (6, 6, 6, 6)
    np.testing.assert_allclose(history.case(3).block_diagnostics[0]["cfl_min"], 1.)
