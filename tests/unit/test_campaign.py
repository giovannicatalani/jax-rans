"""Continuation orchestration with a contracting toy step and real geometry.

The toy step deliberately depends on both the mesh and the freestream. It
lets us exercise multiple converged angles without a long CFD calculation.
"""

from dataclasses import replace
from pathlib import Path
import json
import os

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans import load_case, run
from jaxrans import campaign as module
from jaxrans.campaign import PolarRequest, main, run_campaign
from jaxrans.polar import alpha_range, at_alpha, run_polar
from jaxrans.simulation import make_stepper
from jaxrans.solvers.driver import ExitReason

ROOT = Path(__file__).resolve().parents[2]
SMALL = ("case.domain.cells_airfoil=32", "case.domain.cells_normal=16",
         "numerical.convergence.max_iters=24", "numerical.convergence.block_iters=2",
         "numerical.convergence.tol=0.05", "numerical.convergence.forces_tol=null")


@pytest.fixture(scope="module")
def cases():
    return [load_case(ROOT / "examples/naca0012_euler", SMALL + (f"case.domain.airfoil={name}",))
            for name in ("naca0012", "naca2412")]


@pytest.fixture(scope="module")
def grids(cases):
    return [c.grid() for c in cases]


@pytest.fixture
def contraction(monkeypatch):
    def factory(case, grid, freestream=None, distance=None):
        _, initial = make_stepper(case, grid, freestream=freestream, distance=distance)
        target = initial.U * (1.02 + 0.01 * jnp.tanh(grid.grid.xc) + freestream[2])

        def step(state):
            residual = jnp.max(jnp.abs(target - state.U))
            return state._replace(U=0.5 * (target + state.U)), residual

        return step, initial

    monkeypatch.setattr(module, "make_stepper", factory)
    return factory


def test_profiles_continue_independently_and_match_individual_chains(cases, grids, contraction):
    angles = [(0.0, 1.0, 2.0), (2.0, 1.0)]
    requests = [PolarRequest(f"p{k}", c, a, g) for k, (c, a, g) in enumerate(zip(cases, angles, grids))]
    batched = run_campaign(requests)
    assert batched.batches == (("p0", "p1"),) and batched.successful
    for request in requests:
        single = run_campaign([request]).polars[request.name]
        batch = batched.polars[request.name]
        assert batch.requested_alphas == tuple(request.alphas)
        for a, b in zip(batch.results, single.results):
            np.testing.assert_allclose(a.U, b.U, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(a.history.residual, b.history.residual, rtol=1e-12, atol=1e-14)
            assert a.history.n_iters == b.history.n_iters
        # Manually march the toy step: each new alpha inherits this profile's U.
        previous = None
        for angle, result in zip(request.alphas, batch.results):
            q = jnp.asarray(module.freestream_primitive(at_alpha(request.case, angle).flow, request.grid.components))
            step, state = contraction(request.case, request.grid, q)
            if previous is not None:
                state = state._replace(U=jnp.where(request.grid.interior, previous, state.U))
                # A warm start differs from a uniform start at this angle.
                assert not np.allclose(state.U, make_stepper(at_alpha(request.case, angle), request.grid)[1].U)
            for _ in range(result.history.n_iters):
                state, _ = step(state)
            np.testing.assert_allclose(result.U, state.U, rtol=1e-13, atol=1e-13)
            previous = result.U


def test_failure_stops_only_its_own_chain_and_is_saved(cases, grids, contraction, monkeypatch, tmp_path):
    def broken(case, grid, freestream=None, distance=None):
        step, initial = contraction(case, grid, freestream, distance)

        def advance(state):
            candidate, residual = step(state)
            # Cambered geometry fails; symmetric geometry continues.
            failed = jnp.mean(grid.grid.yc) > 0.001
            return candidate, jnp.where(failed, jnp.nan, residual)
        return advance, initial

    monkeypatch.setattr(module, "make_stepper", broken)
    campaign = run_campaign([PolarRequest(f"p{k}", c, [0, 1], g)
                             for k, (c, g) in enumerate(zip(cases, grids))])
    assert len(campaign.polars["p0"].results) == 2
    assert len(campaign.polars["p1"].results) == 1
    assert campaign.polars["p1"].results[0].history.reason == ExitReason.NON_FINITE
    assert not campaign.successful
    campaign.write(tmp_path)
    meta = json.loads((tmp_path / "p0" / "polar_metadata.json").read_text())
    assert meta["points"][1]["initial_alpha"] == 0.0
    assert meta["mode"] == "continuation"
    assert json.loads((tmp_path / "p1" / "polar_metadata.json").read_text())["requested_alphas"] == [0, 1]


def test_batch_limit_and_incompatible_physics_form_separate_groups(cases, grids, contraction):
    requests = [PolarRequest(str(k), cases[0], [0], grids[0]) for k in range(3)]
    changed = replace(cases[0], flow=replace(cases[0].flow, mach=0.5))
    requests.append(PolarRequest("other_mach", changed, [0], grids[0]))
    campaign = run_campaign(requests, batch_size=2)
    assert campaign.batches == (("0", "1"), ("2",), ("other_mach",))
    assert campaign.successful


def test_single_polar_mode_uses_continuation(cases, grids, contraction):
    polar = run_polar(replace(cases[0], name="NACA 0012"), [1, 0], grid=grids[0], mode="continuation")
    assert polar.mode == "continuation"
    assert len(polar.results) == 2
    assert polar.timing.iterations == sum(r.history.n_iters for r in polar.results)


def test_grid_sequencing_is_only_prepared_at_first_angle(cases, grids, contraction, monkeypatch):
    starts = []

    def coarse(case, grid):
        starts.append(case.flow.alpha_deg)
        state = make_stepper(case, grid)[1]
        return state.U, ({"cells": grid.cells // 4},)

    monkeypatch.setattr(module, "_coarse_start", coarse)
    campaign = run_campaign([PolarRequest("p", replace(cases[0], sequencing=1), [1, 2], grids[0])])
    rows = campaign.polars["p"].results
    assert starts == [1] and len(rows) == 2
    assert rows[0].levels and not rows[1].levels


def test_restart_inherits_interior_and_resets_implicit_control(cases, grids):
    case = replace(cases[0], numerics=replace(cases[0].numerics, solver="implicit"))
    grid = grids[0]
    _, cold = make_stepper(case, grid)
    previous = cold.U * 1.01
    new_case = at_alpha(case, 3)
    _, warm = make_stepper(new_case, grid, initial=previous)
    _, new_cold = make_stepper(new_case, grid)
    np.testing.assert_allclose(warm.U[:, grid.interior], previous[:, grid.interior])
    np.testing.assert_allclose(warm.U[:, ~grid.interior], new_cold.U[:, ~grid.interior])
    assert float(warm.cfl) == case.implicit.cfl
    assert np.isinf(float(warm.residual_norm))
    with pytest.raises(ValueError, match="initial field"):
        make_stepper(case, grid, initial=previous[:2])


def test_run_restart_bypasses_grid_sequencing(cases, grids, monkeypatch):
    import jaxrans.simulation as simulation
    seen = []
    monkeypatch.setattr(simulation, "_march", lambda *a, **kw: seen.append((a, kw)) or "result")
    case = replace(cases[0], sequencing=2)
    U = make_stepper(case, grids[0])[1].U
    assert run(case, grids[0], initial=U) == "result"
    assert len(seen) == 1 and seen[0][1]["initial"] is U


@pytest.mark.parametrize("args, expected", [((-2, 2, 1), [-2, -1, 0, 1, 2]),
                                         ((2, -2, -2), [2, 0, -2]),
                                         ((0, 1, 0.6), [0, 0.6]), ((0, 0, 1), [0])])
def test_alpha_ranges(args, expected):
    np.testing.assert_allclose(alpha_range(*args), expected)


@pytest.mark.parametrize("args", [(0, 1, 0), (1, 0, 1), (0, 1, -1), (0, float("nan"), 1), (0, 91, 1)])
def test_invalid_alpha_ranges(args):
    with pytest.raises(ValueError):
        alpha_range(*args)


@pytest.mark.parametrize("angles", [[], [[0]], [float("nan")], [91]])
def test_invalid_alphas(cases, angles):
    with pytest.raises(ValueError, match="alphas"):
        run_campaign([PolarRequest("profile", cases[0], angles)])


def test_invalid_campaign_inputs(cases):
    request = PolarRequest("profile", cases[0], [0])
    for requests, kwargs in [([], {}), ([request, request], {}),
                              ([replace(request, name="../bad")], {}),
                              ([request], {"batch_size": 0}), ([request], {"devices": 0})]:
        with pytest.raises(ValueError):
            run_campaign(requests, **kwargs)


def test_changed_connectivity_is_not_batched(cases, grids, contraction):
    grid = grids[0]
    changed = np.array(grid.neighbours, copy=True)
    changed.flat[0] = (int(changed.flat[0]) + 1) % grid.interior.size
    other = replace(grid, neighbours=changed)
    campaign = run_campaign([PolarRequest("a", cases[0], [0], grid),
                             PolarRequest("b", cases[0], [0], other)])
    assert campaign.batches == (("a",), ("b",))


def test_iteration_limit_and_stall_do_not_propagate(cases, grids, contraction, monkeypatch):
    short = replace(cases[0], numerics=replace(cases[0].numerics, max_iters=2, tol=1e-12))
    campaign = run_campaign([PolarRequest("short", short, [0, 1], grids[0])])
    assert len(campaign.polars["short"].results) == 1
    assert campaign.polars["short"].results[0].history.reason == ExitReason.MAX_ITERS

    def frozen(case, grid, freestream=None, distance=None):
        _, state = contraction(case, grid, freestream, distance)
        return lambda s: (s, jnp.asarray(1.0)), state

    monkeypatch.setattr(module, "make_stepper", frozen)
    campaign = run_campaign([PolarRequest("stalled", cases[0], [0, 1], grids[0])])
    assert len(campaign.polars["stalled"].results) == 1
    assert campaign.polars["stalled"].results[0].history.reason == ExitReason.STALLED


def test_repeated_angles_have_distinct_output_directories(cases, grids, contraction, tmp_path):
    polar = run_campaign([PolarRequest("p", cases[0], [0, 0, 0.001], grids[0])]).polars["p"]
    polar.write(tmp_path)
    directories = list(tmp_path.glob("point_*"))
    assert len(directories) == 3
    assert all(list(d.glob("*_metadata.json")) for d in directories)


def test_manifest_resolves_relative_cases_and_writes_results(contraction, tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"profiles": [
        {"name": "naca", "case": os.path.relpath(ROOT / "examples/naca0012_euler", tmp_path),
         "alpha": [1, 0, -1], "overrides": list(SMALL)}]}))
    assert main([str(manifest), "--out", str(tmp_path / "out"), "--quiet"]) == 0
    meta = json.loads((tmp_path / "out" / "campaign_metadata.json").read_text())
    assert meta["profiles"]["naca"]["calculated_alphas"] == [1, 0]


def test_manifest_failure_returns_one_and_retains_failed_point(contraction, tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"profiles": [
        {"name": "naca", "case": str(ROOT / "examples/naca0012_euler"), "alphas": [0, 1],
         "overrides": list(SMALL) + ["numerical.convergence.max_iters=2"]}]}))
    assert main([str(manifest), "--out", str(tmp_path / "out"), "--quiet"]) == 1
    meta = json.loads((tmp_path / "out" / "campaign_metadata.json").read_text())
    assert not meta["successful"]
    assert meta["profiles"]["naca"]["calculated_alphas"] == [0]


def test_manifest_does_not_ignore_invalid_cli_batch_size(tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"batch_size": 2, "profiles": [
        {"name": "naca", "case": str(ROOT / "examples/naca0012_euler"), "alphas": [0]}]}))
    with pytest.raises(SystemExit) as error:
        main([str(manifest), "--out", str(tmp_path / "out"), "--batch-size", "0"])
    assert error.value.code == 2
