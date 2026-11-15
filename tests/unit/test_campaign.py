"""Continuation orchestration with a contracting toy step and real geometry.

The toy step deliberately depends on both the mesh and the freestream. It
lets us exercise multiple converged angles without a long CFD calculation.
"""

from dataclasses import replace
from pathlib import Path
import json
import csv
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
from jaxrans.simulation import equations_settled

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
    def toy_equation_norms(case, template, U, q, g):
        _, initial = factory(case, template, freestream=q)
        target = initial.U * (1.02 + 0.01 * jnp.tanh(template.grid.xc) + q[2])
        return jnp.full((U.shape[0],), jnp.max(jnp.abs(target - U)))

    monkeypatch.setattr(module, "_equation_norms", toy_equation_norms)
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


def test_progress_reports_each_profile_residual(cases, grids, contraction):
    requests = [PolarRequest(f"p{k}", c, [0], g) for k, (c, g) in enumerate(zip(cases, grids))]
    old, detailed = [], []
    campaign = run_campaign(requests, progress=lambda *args: old.append(args),
                            progress_with_residual=lambda *args: detailed.append(args))
    assert old == [entry[:4] for entry in detailed]
    for name, _, iteration, _, residual in detailed:
        history = campaign.polars[name].results[0].history.residual
        if iteration <= len(history):
            assert residual == pytest.approx(history[iteration - 1])


def test_progress_reports_drag_and_moment_for_each_profile(cases, grids, contraction):
    requests = [PolarRequest(f"p{k}", c, [0], g) for k, (c, g) in enumerate(zip(cases, grids))]
    records = []
    campaign = run_campaign(requests, progress_with_coefficients=lambda *args: records.append(args))
    assert records
    for name, alpha, iteration, cl, residual, cd, cm in records:
        result = campaign.polars[name].results[0]
        block = iteration // result.case.block_iters - 1
        if block < len(result.coefficients_history):
            np.testing.assert_allclose([cl, cd, cm], result.coefficients_history[block])
            assert residual == pytest.approx(result.history.residual[iteration - 1])


def test_force_settling_continues_to_next_angle(cases, grids, contraction):
    numerics = replace(cases[0].numerics, tol=1e-30, forces_tol=1.0, max_iters=40)
    case = replace(cases[0], numerics=numerics)
    campaign = run_campaign([PolarRequest("p", case, [0, 1], grids[0])])
    polar = campaign.polars["p"]
    assert list(polar.alphas) == [0, 1]
    assert all(result.history.reason == ExitReason.SETTLED for result in polar.results)


def test_one_unconverged_equation_prevents_settling(cases):
    numerics = replace(cases[0].numerics, forces_tol=1e-4)
    peak = np.ones(4)
    current = np.array([1e-9, 1e-9, 1e-9, 1e-2])
    assert not equations_settled(current, peak, numerics)
    current[-1] = 1e-9
    assert equations_settled(current, peak, numerics)


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


def test_manifest_resolves_relative_cases_and_writes_results(contraction, tmp_path, capsys):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"profiles": [
        {"name": "naca", "case": os.path.relpath(ROOT / "examples/naca0012_euler", tmp_path),
         "alpha": [1, 0, -1], "overrides": list(SMALL)}]}))
    assert main([str(manifest), "--out", str(tmp_path / "out")]) == 0
    output = capsys.readouterr().out
    assert "residual " in output and " CL " in output
    assert " Cx(CD) " in output and " Cm " in output
    with (tmp_path / "out" / "points.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["alpha_deg"]) for row in rows] == [1, 0]
    assert all(row["status"] and row["residual_energy"] and row["cl"] for row in rows)
    summary = (tmp_path / "out" / "summary.txt").read_text()
    assert summary in output
    assert "2/2 angles calculated" in summary
    assert "maximum CL" in summary and "minimum CD" in summary
    assert "iterations" in summary and "residual" in summary
    meta = json.loads((tmp_path / "out" / "campaign_metadata.json").read_text())
    assert meta["profiles"]["naca"]["calculated_alphas"] == [1, 0]
    point_meta = next((tmp_path / "out" / "naca" / "alpha_+1.00").glob("*_metadata.json"))
    equations = json.loads(point_meta.read_text())["extra"]["equation_residuals"]
    assert set(equations) == {"mass", "momentum_x", "momentum_y", "energy"}
    assert all(np.isfinite(row["l2_volume"]) and np.isfinite(row["linf"]) for row in equations.values())


def test_manifest_failure_returns_one_and_retains_failed_point(contraction, tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"profiles": [
        {"name": "naca", "case": str(ROOT / "examples/naca0012_euler"), "alphas": [0, 1],
         "overrides": list(SMALL) + ["numerical.convergence.max_iters=2"]}]}))
    assert main([str(manifest), "--out", str(tmp_path / "out"), "--quiet"]) == 1
    meta = json.loads((tmp_path / "out" / "campaign_metadata.json").read_text())
    assert not meta["successful"]
    assert meta["profiles"]["naca"]["calculated_alphas"] == [0]
    summary = (tmp_path / "out" / "summary.txt").read_text()
    assert "incomplete" in summary and "1/2 angles calculated" in summary
    assert "max_iters" in summary and "maximum CL" not in summary


def test_manifest_machs_override_case_and_create_separate_polars(contraction, tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"profiles": [{
        "name": "naca", "case": str(ROOT / "examples/naca0012_euler"), "alphas": [0, 1],
        "machs": [0.5, 0.7], "overrides": list(SMALL) + ["case.freestream.mach=0.6"]}]}))
    assert main([str(manifest), "--out", str(tmp_path / "out"), "--quiet"]) == 0
    for mach in (0.5, 0.7):
        meta = json.loads((tmp_path / "out" / f"naca_mach_{mach}" / "polar_metadata.json").read_text())
        assert meta["case_setup"]["case"]["freestream"]["mach"] == mach
        assert meta["alphas"] == [0, 1]


def test_resume_preserves_converged_points_and_extends_polar(contraction, tmp_path):
    manifest = tmp_path / "campaign.json"
    entry = {"name": "naca", "case": str(ROOT / "examples/naca0012_euler"),
             "alphas": [0], "overrides": list(SMALL)}
    manifest.write_text(json.dumps({"profiles": [entry]}))
    out = tmp_path / "out"
    assert main([str(manifest), "--out", str(out), "--quiet"]) == 0
    point = out / "naca" / "alpha_+0.00"
    original = {p.name: p.read_bytes() for p in point.iterdir()}
    entry["alphas"] = [0, 1]
    manifest.write_text(json.dumps({"profiles": [entry]}))
    assert main([str(manifest), "--out", str(out), "--resume", "--quiet"]) == 0
    assert original == {p.name: p.read_bytes() for p in point.iterdir()}
    meta = json.loads((out / "naca" / "polar_metadata.json").read_text())
    assert meta["alphas"] == [0, 1]
    assert meta["points"][1]["initial_alpha"] == 0
    entry["overrides"] = list(SMALL) + ["case.freestream.mach=0.6"]
    manifest.write_text(json.dumps({"profiles": [entry]}))
    with pytest.raises(SystemExit):
        main([str(manifest), "--out", str(out), "--resume", "--quiet"])
    entry["overrides"] = list(SMALL)
    manifest.write_text(json.dumps({"profiles": [entry]}))
    point_meta = next(point.glob("*_metadata.json"))
    old = json.loads(point_meta.read_text())
    old["config"]["numerics"].pop("equation_tol")
    point_meta.write_text(json.dumps(old))
    with pytest.raises(SystemExit):
        main([str(manifest), "--out", str(out), "--resume", "--quiet"])


def test_resume_migrates_legacy_output_into_mach_directories(contraction, tmp_path):
    manifest = tmp_path / "campaign.json"
    entry = {"name": "naca", "case": str(ROOT / "examples/naca0012_euler"),
             "alphas": [0], "overrides": list(SMALL)}
    manifest.write_text(json.dumps({"profiles": [entry]}))
    out = tmp_path / "out"
    assert main([str(manifest), "--out", str(out), "--quiet"]) == 0
    legacy = (out / "naca" / "polar.dat").read_bytes()
    entry["machs"] = [0.8, 0.7]
    manifest.write_text(json.dumps({"profiles": [entry]}))
    assert main([str(manifest), "--out", str(out), "--resume", "--quiet"]) == 0
    assert (out / "naca" / "polar.dat").read_bytes() == legacy
    for mach in (0.8, 0.7):
        directory = out / f"naca_mach_{mach}"
        assert (directory / "polar.dat").is_file()
        assert json.loads((directory / "polar_metadata.json").read_text())["case_setup"]["case"]["freestream"]["mach"] == mach


@pytest.mark.parametrize("machs", [[], [0.5, 0.5], [0], [-1], [True], ["0.5"], [float("nan")], 0.5])
def test_manifest_rejects_invalid_machs(tmp_path, machs):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"profiles": [{"name": "naca", "case": "unused", "alphas": [0],
                                                    "machs": machs}]}))
    with pytest.raises(SystemExit) as error:
        main([str(manifest), "--out", str(tmp_path / "out")])
    assert error.value.code == 2


def test_manifest_does_not_ignore_invalid_cli_batch_size(tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"batch_size": 2, "profiles": [
        {"name": "naca", "case": str(ROOT / "examples/naca0012_euler"), "alphas": [0]}]}))
    with pytest.raises(SystemExit) as error:
        main([str(manifest), "--out", str(tmp_path / "out"), "--batch-size", "0"])
    assert error.value.code == 2


@pytest.fixture
def adaptive_solver(cases, grids, contraction, monkeypatch):
    """Controlled failures encode their fields so unsafe inheritance is visible."""
    from jaxrans.polar import Polar, ContinuationTiming
    base = run_campaign([PolarRequest("seed", cases[0], [0], grids[0])]).polars["seed"].results[0]
    calls = []
    settings = {"jump": 1.1, "cold_fail": set()}

    def group(requests, grids, geometries, devices, progress, detailed, on_point=None,
              progress_with_coefficients=None, progress_with_equation_residuals=None, _kernels=None):
        request = requests[0]
        alpha = float(request.alphas[0])
        initial = None if request.initial is None else np.asarray(request.initial)
        assert initial is None or not np.any(initial == 777), "a failed field was inherited"
        source = None if initial is None else float(initial[0, np.asarray(grids[0].interior)].flat[0]) - 10
        calls.append((alpha, source))
        success = (alpha not in settings["cold_fail"] if source is None
                   else abs(alpha - source) <= settings["jump"])
        result = replace(base, case=at_alpha(request.case, alpha),
                         U=jnp.full_like(base.U, alpha + 10 if success else 777),
                         history=replace(base.history, reason=ExitReason.SETTLED if success else ExitReason.MAX_ITERS))
        return [Polar(request.case, np.asarray([alpha]), [result], ContinuationTiming(0, 1))]

    monkeypatch.setattr(module, "_run_group", group)
    return calls, settings


@pytest.mark.parametrize("angles, expected", [([0, 2], [0, 2, 1, 2]), ([2, 0], [2, 0, 1, 0]),
                                              ([0, 2, 2], [0, 2, 1, 2, 2])])
def test_adaptive_subdivision_only_inherits_validated_fields(cases, grids, adaptive_solver, angles, expected):
    policy = module.ContinuationPolicy(adaptive=True)
    attempts, requested = [], []
    campaign = run_campaign([PolarRequest("p", cases[0], angles, grids[0])], continuation=policy,
                            on_attempt=lambda n, a, r: attempts.append(r),
                            on_point=lambda n, a, r: requested.append(a))
    polar = campaign.polars["p"]
    assert campaign.successful
    assert requested == angles and list(polar.alphas) == angles
    assert [r.case.flow.alpha_deg for r in attempts] == expected
    assert list(polar.attempts) == attempts
    assert [r.provenance["kind"] for r in attempts[:4]] == ["requested", "requested", "intermediate", "requested"]
    assert attempts[2].provenance["source_attempt"] == 0
    assert attempts[3].provenance["source_attempt"] == 2


@pytest.mark.parametrize("budget,min_step,expected", [(1, .125, [0, 2]), (2, .125, [0, 2, 1]),
                                                       (20, 1.1, [0, 2])])
def test_adaptive_exhaustion_retains_requested_failure(cases, grids, adaptive_solver, budget, min_step, expected):
    policy = module.ContinuationPolicy(True, min_step, budget)
    campaign = run_campaign([PolarRequest("p", cases[0], [0, 2, 3], grids[0], continuation=policy)])
    polar = campaign.polars["p"]
    assert not campaign.successful
    assert list(polar.alphas) == [0, 2]
    assert polar.results[-1].history.reason == ExitReason.MAX_ITERS
    assert [r.case.flow.alpha_deg for r in polar.attempts] == expected


@pytest.mark.parametrize("mode,count", [("stop", 1), ("independent", 2)])
def test_adaptive_no_valid_start(cases, grids, adaptive_solver, mode, count):
    calls, settings = adaptive_solver
    settings["cold_fail"] = {0}
    policy = module.ContinuationPolicy(True, on_failure=mode)
    campaign = run_campaign([PolarRequest("p", cases[0], [0, 2], grids[0], continuation=policy)])
    assert len(campaign.polars["p"].results) == count
    assert calls == [(0, None), (2, None)][:count]
    assert not campaign.successful


def test_adaptive_independent_resets_even_successful_intermediate(cases, grids, adaptive_solver):
    calls, _ = adaptive_solver
    policy = module.ContinuationPolicy(True, max_attempts=2, on_failure="independent")
    campaign = run_campaign([PolarRequest("p", cases[0], [0, 2, 3], grids[0], continuation=policy)])
    assert calls == [(0, None), (2, 0), (1, 0), (3, None)]
    assert not campaign.successful


@pytest.mark.parametrize("kwargs", [{"adaptive": 1}, {"min_step": 0}, {"min_step": True},
                                     {"min_step": float("nan")}, {"max_attempts": 0},
                                     {"max_attempts": True}, {"on_failure": "retry"},
                                     {"on_failure": "independent"}])
def test_continuation_policy_validation(kwargs):
    with pytest.raises(ValueError):
        module.ContinuationPolicy(**kwargs)


@pytest.mark.parametrize("interrupt_after", [1, 2, 3, 4])
def test_adaptive_resume_after_every_attempt(cases, grids, adaptive_solver, tmp_path, interrupt_after):
    calls, _ = adaptive_solver
    request = PolarRequest("p", cases[0], [0, 2], grids[0], continuation=module.ContinuationPolicy(True))
    directory = tmp_path / "p"
    request, _ = module._adaptive_resume(request, directory, False)

    def checkpoint(name, alpha, result):
        module._checkpoint_attempt(directory, result)
        if result.provenance["attempt"] + 1 == interrupt_after:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_campaign([request], on_attempt=checkpoint)
    snapshot = {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    # An interrupted staging directory must never count as a committed attempt.
    partial = directory / "attempts" / ".attempt_000004.tmp"
    partial.mkdir()
    (partial / "partial").write_text("interrupted write")
    request, saved = module._adaptive_resume(request, directory, True)
    callbacks = []
    resumed = run_campaign([request], resume_attempts={"p": saved},
                           on_attempt=lambda n, a, r: (callbacks.append(r.provenance["attempt"]),
                                                       module._checkpoint_attempt(directory, r)))
    assert resumed.successful
    assert calls == [(0, None), (2, 0), (1, 0), (2, 1)]
    assert callbacks == list(range(interrupt_after, 4))
    assert snapshot == {name: (directory / name).read_bytes() for name in snapshot}
    assert len(list((directory / "attempts").glob("attempt_*"))) == 4
    resumed.write(tmp_path)
    meta = json.loads((directory / "polar_metadata.json").read_text())
    assert meta["alphas"] == [0, 2] and len(meta["attempts"]) == 4
    assert [p["source_attempt"] for p in meta["points"]] == [None, 2]
    assert saved[0].coefficients_history.shape[1] == 3
    assert saved[0].convergence_history


def test_adaptive_resume_guards_configuration_policy_angles_and_mesh(cases, grids, adaptive_solver, tmp_path):
    request = PolarRequest("p", cases[0], [0, 2], grids[0], continuation=module.ContinuationPolicy(True))
    module._adaptive_resume(request, tmp_path, False)
    run_campaign([request], on_attempt=lambda n, a, r: module._checkpoint_attempt(tmp_path, r))
    for changed in (replace(request, alphas=[0, 3]),
                    replace(request, continuation=module.ContinuationPolicy(True, max_attempts=3))):
        with pytest.raises(ValueError, match="differ"):
            module._adaptive_resume(changed, tmp_path, True)
    extended, saved = module._adaptive_resume(replace(request, alphas=[0, 2, 3]), tmp_path, True)
    assert run_campaign([extended], resume_attempts={"p": saved}).successful
    folder = tmp_path / "attempts" / "attempt_000000"
    path = next(folder.glob("*_fields.npz"))
    with np.load(path) as fields:
        values = dict(fields)
    values["x_node_0"] = values["x_node_0"] + .01
    np.savez(path, **values)
    with pytest.raises(ValueError, match="mesh differs"):
        module._adaptive_resume(extended, tmp_path, True)


def test_campaign_records_full_coefficients_and_initial_equation_peak(cases, grids, contraction):
    polar = run_campaign([PolarRequest("p", cases[0], [0], grids[0])]).polars["p"]
    result = polar.results[0]
    np.testing.assert_array_equal(result.cl_history, result.coefficients_history[:, 0])
    assert result.coefficients_history.shape == (result.history.n_iters // result.case.block_iters, 3)
    first = result.convergence_history[0]
    assert first["names"] == ["mass", "momentum_x", "momentum_y", "energy"]
    assert np.all(np.asarray(first["peak"]) > np.asarray(first["current"]))
    assert first["thresholds"] and "blocking_equations" in first


def test_adaptive_manifest_resume_repairs_csv_without_duplicates(cases, grids, adaptive_solver, monkeypatch, tmp_path):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"continuation": {"adaptive": True}, "profiles": [
        {"name": "p", "case": str(ROOT / "examples/naca0012_euler"), "alphas": [0, 2],
         "overrides": list(SMALL)}]}))
    output = tmp_path / "out"
    checkpoint = module._checkpoint_attempt

    def interrupted(directory, result):
        checkpoint(directory, result)
        if result.provenance["attempt"] == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(module, "_checkpoint_attempt", interrupted)
    with pytest.raises(KeyboardInterrupt):
        main([str(manifest), "--out", str(output), "--quiet"])
    monkeypatch.setattr(module, "_checkpoint_attempt", checkpoint)
    assert main([str(manifest), "--out", str(output), "--resume", "--quiet"]) == 0
    csv_before = (output / "points.csv").read_bytes()
    with (output / "points.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["attempt"]) for row in rows] == [0, 1, 2, 3]
    assert [row["kind"] for row in rows] == ["requested", "requested", "intermediate", "requested"]
    assert main([str(manifest), "--out", str(output), "--resume", "--quiet"]) == 0
    assert (output / "points.csv").read_bytes() == csv_before
    assert len(adaptive_solver[0]) == 4


def test_continuation_table_failed_equations_with_flat_lift_are_not_settled(cases, grids, contraction):
    polar = run_campaign([PolarRequest("p", cases[0], [0], grids[0])]).polars["p"]
    original = polar.results[0]
    failed = replace(original, cl_history=np.zeros(16),
                     history=replace(original.history, reason=ExitReason.MAX_ITERS))
    assert failed.cl_spread == 0
    assert replace(polar, results=[failed]).table()[0, -1] == 0
    assert replace(polar, results=[failed], mode="batch").table()[0, -1] == 1


@pytest.mark.parametrize("change", ["floors", "force", "strict", "new_force"])
def test_legacy_resume_conservatively_certifies_all_criteria(cases, grids, contraction, tmp_path, change):
    request = PolarRequest("p", cases[0], [0], grids[0])
    result = run_campaign([request]).polars["p"].results[0]
    folder = tmp_path / "point"
    result.write(folder)
    next(folder.glob("*_solver_history.json")).unlink()
    metadata_path = next(folder.glob("*_metadata.json"))
    metadata = json.loads(metadata_path.read_text())
    old = metadata["config"]["numerics"]
    old.update(forces_tol=.01, equation_tol=1e-4, equation_abs_tol=1e-8,
               equation_abs_tols=[["energy", .01]])
    new = replace(cases[0].numerics, forces_tol=.01, equation_tol=1e-4,
                  equation_abs_tol=1e-8, equation_abs_tols=(("energy", .1),))
    if change == "floors":
        new = replace(new, equation_abs_tols=(("energy", .001),))
    elif change == "force":
        new = replace(new, forces_tol=.001)
    elif change == "strict":
        new = replace(new, forces_tol=None)
    else:
        old["forces_tol"] = None
    metadata_path.write_text(json.dumps(metadata))
    changed = replace(request, case=replace(request.case, numerics=new))
    with pytest.raises(ValueError, match="certified"):
        module._load_point(changed, grids[0], folder, 0,
                           {"exit_reason": ExitReason.SETTLED, "iterations": result.history.n_iters})


def test_adaptive_example_resolves_and_validates_case_options():
    path = ROOT / "examples/polar_rae2822_adaptive.json"
    manifest = json.loads(path.read_text())
    policy = module.ContinuationPolicy(**manifest["continuation"])
    entry = manifest["profiles"][0]
    case = load_case(path.parent / entry["case"], tuple(entry["overrides"]))
    assert policy.adaptive and case.numerics.diagnostics
    assert case.implicit.nonlinear_merit == "equations" and case.implicit.backtracking_steps == 4


def test_adaptive_api_write_rejects_changed_physics_preserving_files(cases, grids, adaptive_solver, tmp_path):
    request = PolarRequest("p", cases[0], [0], grids[0], continuation=module.ContinuationPolicy(True))
    original = run_campaign([request]).polars["p"]
    original.write(tmp_path)
    before = {str(path.relative_to(tmp_path)): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}
    changed_case = load_case(ROOT / "examples/naca0012_euler", SMALL + ("case.freestream.mach=0.5",))
    changed = run_campaign([replace(request, case=changed_case)]).polars["p"]
    assert changed.case.flow.mach != original.case.flow.mach
    with pytest.raises(ValueError, match="configuration differs"):
        changed.write(tmp_path)
    assert before == {str(path.relative_to(tmp_path)): path.read_bytes()
                      for path in tmp_path.rglob("*") if path.is_file()}


def test_adaptive_checkpoint_rejects_changed_fields_preserving_files(cases, grids, adaptive_solver, tmp_path):
    request = PolarRequest("p", cases[0], [0], grids[0], continuation=module.ContinuationPolicy(True))
    result = run_campaign([request]).polars["p"].attempts[0]
    module._checkpoint_attempt(tmp_path, result)
    before = {str(path.relative_to(tmp_path)): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file()}
    changed = replace(result, U=result.U + jnp.where(grids[0].interior, .01, 0))
    assert changed.provenance == result.provenance
    with pytest.raises(ValueError, match="conflicting adaptive fields"):
        module._checkpoint_attempt(tmp_path, changed)
    assert before == {str(path.relative_to(tmp_path)): path.read_bytes()
                      for path in tmp_path.rglob("*") if path.is_file()}
