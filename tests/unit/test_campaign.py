"""Strict configuration, audit outputs, and multi-geometry batch equivalence."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from jaxrans import run
from jaxrans.campaign import load_campaign, prepare_campaign, run_campaign
from jaxrans.campaign_batch import run_case_batch
from jaxrans.case import CaseError, load_case
from jaxrans.geometry.airfoil import naca4, read_hug

ROOT = Path(__file__).resolve().parents[2]


def write_campaign(path, *, solver="explicit", model="none"):
    path.mkdir(parents=True, exist_ok=True)
    profiles = []
    for code, y, c, angle in (("0018", 0., 3., 3.), ("0010", 2., 1., -2.)):
        foil = naca4(code, n=31, sharp=False)
        a = np.radians(angle)
        x, z = c*(foil.x*np.cos(a)-foil.y*np.sin(a)), c*(foil.x*np.sin(a)+foil.y*np.cos(a))
        profiles.append(np.c_[x, np.full_like(x, y), z + 0.1*y])
    with (path / "wing.icms").open("w") as stream:
        stream.write("Example header\n")
        np.savetxt(stream, np.concatenate(profiles))
    doc = {
        "general": {"case_name": "test_wing"},
        "domain": {"grid": "c_grid", "cells_airfoil": 32, "cells_wake": 8, "trailing_edge": "close",
                   "cells_normal": 12, "wall_spacing": 0.001, "far_field": 10.},
        "sections": {"file": "wing.icms", "count": 2, "units": "m"},
        "freestream": {"reynolds": 5000.},
        "turbulence": {"model": model},
        "campaign": {"mach": [0.25, 0.45], "alpha_deg": [-1., 2.], "batch_size": 3},
    }
    numerical = {"time_integration": {"solver": solver, "cfl": 0.1},
                 "convergence": {"max_iters": 3, "block_iters": 2, "forces_tol": None},
                 "implicit": {"krylov_subspace": 4, "krylov_restarts": 1}}
    (path / "case.json").write_text(json.dumps(doc))
    (path / "numerical.json").write_text(json.dumps(numerical))
    return doc


def test_example_config_and_ranges():
    c = load_campaign(ROOT / "examples/wing_sections")
    assert c.sections["count"] == 10
    assert c.machs == pytest.approx([.3, .4, .5])
    assert c.alphas == pytest.approx([-2., 0., 2.])
    assert c.template.flow.reynolds == 6e6
    assert not any("kink" in k for k in c.sections)


@pytest.mark.parametrize("block,key,value", [
    ("sections", "kink_indices", [1]), ("sections", "kink_sides", "inboard"),
    ("sections", "kink_side", "inboard"), ("sections", "count", True),
    ("sections", "units", "cm"), ("sections", "header_lines", -1),
    ("sections", "plane_tolerance", float("nan")),
    ("campaign", "batch_size", 0), ("campaign", "mach", []),
    ("campaign", "mach", [True]), ("campaign", "alpha_deg", [1, 1]),
    ("campaign", "mach", {"start": .3, "stop": .8, "step": .2}),
    ("freestream", "alpha_deg", 2), ("freestream", "sweep_deg", 20),
    ("domain", "airfoil", "naca0012"), ("domain", "section", "normal"),
    ("freestream", "reynolds", None),
])
def test_invalid_config_is_rejected(tmp_path, block, key, value):
    doc = write_campaign(tmp_path)
    doc[block][key] = value
    (tmp_path / "case.json").write_text(json.dumps(doc))
    with pytest.raises((CaseError, ValueError)):
        load_campaign(tmp_path)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("campaign")
    write_campaign(directory)
    return prepare_campaign(load_campaign(directory), directory / "output")


def test_preprocessing_is_inspectable_and_preserves_simple_case_schema(prepared):
    output = prepared.directory
    report = json.loads((output / "preprocessing.json").read_text())
    assert report["case_count"] == 8
    assert len(report["source_sha256"]) == 64
    assert "incoming" in report["sweep_rule"]
    assert (output / "wing_geometry.png").stat().st_size > 1000
    for s, case in zip(prepared.sections, prepared.cases):
        folder = output / "sections" / f"section_{s.index:03d}"
        assert (folder / "geometry.png").stat().st_size > 1000
        record = json.loads((folder / "geometry.json").read_text())
        assert record["alpha_correction_deg"] == 0.
        assert record["reconstruction_error_m"] < 1e-12
        assert record["grid_quality"]["min_area"] > 0.
        got = load_case(folder)
        assert got.flow == case.flow
        assert got.flow.alpha_deg == -1.  # no twist correction despite 3 and -2 deg pitch
        assert got.flow.reynolds == 5000.
        foil, = read_hug(folder / "profile.hug")
        np.testing.assert_allclose(foil.x, s.airfoil.x)
        assert not foil.sharp


@pytest.mark.parametrize("solver,model", [("explicit", "none"), ("implicit", "sa_noft2")])
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_batch_matches_individual_runs_on_different_geometries(prepared, solver, model, backend):
    if backend == "gpu" and jax.default_backend() != "gpu":
        pytest.skip("GPU verification requires an initialized JAX GPU backend")
    numerical = replace(prepared.cases[0].numerics, solver=solver, max_iters=4, forces_tol=None)
    cases, grids = [], []
    for s, mach, alpha in ((0, .25, -1), (1, .45, 2), (0, .45, 2)):
        base = prepared.cases[s]
        flow = replace(base.flow, mach=mach, alpha_deg=alpha,
                       turbulence=None if model == "none" else model)
        cases.append(replace(base, flow=flow, numerics=numerical))
        grids.append(prepared.grids[s])
    with jax.default_device(jax.devices(backend)[0]):
        batch = run_case_batch(cases, grids)
        for case, grid, result in zip(cases, grids, batch):
            single = run(case, grid)
            assert result.history.n_iters == 4
            assert np.isfinite(result.U).all()
            assert np.isfinite(single.U).all()
            np.testing.assert_allclose(result.U, single.U, rtol=2e-9, atol=2e-11)
            assert result.forces.cl == pytest.approx(single.forces.cl, rel=2e-9, abs=1e-11)
            assert result.timing.cases == 3


def test_end_to_end_campaign_output_and_precise_case_paths(prepared):
    result = run_campaign(prepared, backend="cpu")
    assert len(result.rows) == 8
    assert len({r["directory"] for r in result.rows}) == 8
    assert {r["mach"] for r in result.rows} == {.25, .45}
    assert {r["alpha_deg"] for r in result.rows} == {-1., 2.}
    assert {r["reynolds"] for r in result.rows} == {5000.}
    assert (result.directory / "campaign.csv").is_file()
    assert json.loads((result.directory / "execution.json").read_text())["completed_cases"] == 8
    for row in result.rows:
        folder = result.directory / row["directory"]
        assert load_case(folder).flow.alpha_deg == row["alpha_deg"]
        assert load_case(folder).flow.sweep_deg == row["sweep_deg"]
        assert list(folder.glob("*_fields.npz"))
        assert row["exit_reason"] != "converged"  # three steps are not a converged polar
        assert row["iterations"] == 3  # partial last block, no overshoot


def test_grid_sequencing_is_explicitly_rejected(tmp_path):
    write_campaign(tmp_path)
    path = tmp_path / "numerical.json"
    numerical = json.loads(path.read_text())
    numerical["convergence"]["grid_sequencing"] = 1
    path.write_text(json.dumps(numerical))
    with pytest.raises(CaseError, match="grid_sequencing"):
        load_campaign(tmp_path)


def test_aircraft_incidence_and_reynolds_are_unchanged_across_campaign(prepared):
    # Configuration records, rather than output coefficients, establish the
    # incidence convention independently of aerodynamic convergence.
    from jaxrans.campaign import _case
    for section in prepared.sections:
        c = _case(prepared.campaign, prepared.directory / "unused.hug", section, .45, 2.)
        assert c.flow.alpha_deg == 2.
        assert c.flow.reynolds == 5000.
        assert abs(section.chord_pitch_deg) > 1
        assert c.flow.sweep_deg == section.sweep_deg


def test_converged_and_failed_lanes_do_not_stop_other_cases(prepared, monkeypatch):
    """Exercise per-case stopping with an analytically known tiny iteration."""
    from types import SimpleNamespace
    import jax.numpy as jnp
    import jaxrans.campaign_batch as batch_module
    from jaxrans.solvers.explicit import SolverState

    def stepper(case, grid, qi, distance, mui):
        def step(state):
            iteration = state.U[0, 0]
            norm = jnp.where(qi[1] < .3, .1, .8)**iteration
            norm = jnp.where(qi[1] > .6, jnp.nan, norm)
            return SolverState(state.U + 1), norm
        return step, SolverState(jnp.zeros((1, 1)))

    monkeypatch.setattr(batch_module, "make_stepper", stepper)
    monkeypatch.setattr(batch_module, "make_residual", lambda *a: SimpleNamespace(distance=None))
    monkeypatch.setattr(batch_module, "integrate_forces", lambda U, *a: SimpleNamespace(
        cl=float(U[0, 0]), cd=0., cm=0.))
    base = prepared.cases[0]
    numerics = replace(base.numerics, max_iters=8, tol=.01, forces_tol=None)
    cases = [replace(base, flow=replace(base.flow, mach=m, alpha_deg=0, sweep_deg=0), numerics=numerics)
             for m in (.25, .45, .75)]
    results = run_case_batch(cases, [prepared.grids[0]]*3)
    assert [r.history.reason for r in results] == ["converged", "max_iters", "non_finite"]
    assert [r.history.n_iters for r in results] == [4, 8, 2]
    # State freezes at each lane's own stop block, not the batch's last block.
    assert [float(r.U[0, 0]) for r in results] == [4., 8., 2.]


def test_constant_density_residual_does_not_stall_changing_fields(prepared, monkeypatch):
    from types import SimpleNamespace
    import jax.numpy as jnp
    import jaxrans.campaign_batch as batch_module
    from jaxrans.solvers.explicit import SolverState

    def stepper(case, grid, qi, distance, mui):
        return (lambda s: (SolverState(s.U + 1), jnp.asarray(1.))), SolverState(jnp.zeros((1, 1)))

    monkeypatch.setattr(batch_module, "make_stepper", stepper)
    monkeypatch.setattr(batch_module, "make_residual", lambda *a: SimpleNamespace(distance=None))
    monkeypatch.setattr(batch_module, "integrate_forces", lambda U, *a: SimpleNamespace(
        cl=float(U[0, 0]), cd=0., cm=0.))
    base = prepared.cases[0]
    case = replace(base, numerics=replace(base.numerics, max_iters=4, forces_tol=None))
    result, = run_case_batch([case], [prepared.grids[0]])
    assert result.history.reason == "max_iters"
    assert float(result.U[0, 0]) == 4.


def test_monitor_preserves_nan_corrections_hidden_by_state_rejection(prepared, monkeypatch, tmp_path):
    import csv
    from types import SimpleNamespace
    import jax.numpy as jnp
    import jaxrans.campaign_batch as batch_module
    from jaxrans.campaign import _write_solver_progress
    from jaxrans.solvers.explicit import SolverState
    from jaxrans.solvers.implicit import NewtonDiagnostics

    def stepper(case, grid, qi, distance, mui, *, diagnostics=False):
        def step(s):
            finite = qi[1] > .3
            report = NewtonDiagnostics(finite, finite, finite, jnp.where(finite, .1, jnp.nan),
                                       jnp.asarray(1.), jnp.asarray(.5))
            return SolverState(s.U + finite), jnp.asarray(1.), report
        assert diagnostics
        return step, SolverState(jnp.zeros((1, 1)))

    # Initial-state creation uses the ordinary API; the diagnostic step alone
    # opts into the extra result.
    def factory(case, grid, qi, distance, mui, *, diagnostics=False):
        if not diagnostics:
            return None, SolverState(jnp.zeros((1, 1)))
        return stepper(case, grid, qi, distance, mui, diagnostics=True)

    monkeypatch.setattr(batch_module, "make_stepper", factory)
    monkeypatch.setattr(batch_module, "make_residual", lambda *a: SimpleNamespace(distance=None))
    monkeypatch.setattr(batch_module, "integrate_forces", lambda U, *a: SimpleNamespace(
        cl=float(U[0, 0]), cd=0., cm=0.))
    base = prepared.cases[0]
    numerics = replace(base.numerics, solver="implicit", max_iters=4, forces_tol=None)
    cases = [replace(base, numerics=numerics, flow=replace(base.flow, mach=m, alpha_deg=0, sweep_deg=0))
             for m in (.25, .45)]
    path = tmp_path / "solver_progress.csv"
    monitor = lambda r: _write_solver_progress(path, r, [(0, 0, 0), (0, 1, 0)], prepared.campaign, 0)
    results = run_case_batch(cases, [prepared.grids[0]]*2, monitor=monitor)
    assert [r.history.reason for r in results] == ["stalled", "max_iters"]
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3  # the stalled lane is logged once; the live one twice
    assert rows[0]["nonfinite_corrections"] == "2"
    assert rows[0]["changed_steps"] == "0"
    assert rows[1]["accepted_steps"] == "2"
    assert rows[1]["changed_steps"] == "2"


def test_cli_campaign_check_prepares_artifacts_and_applies_overrides(tmp_path, monkeypatch, capsys):
    from jaxrans.__main__ import main
    import jaxrans.campaign as campaign_module

    directory = tmp_path / "input"
    write_campaign(directory)
    output = tmp_path / "check"

    def forbidden(*args, **kwargs):
        pytest.fail("--check must not start CFD calculations")

    monkeypatch.setattr(campaign_module, "run_campaign", forbidden)
    assert main([str(directory), "--check", "--out", str(output),
                 "--set=case.campaign.mach=[0.35]", "--set=case.campaign.alpha_deg=[2.0]",
                 "--set=case.campaign.batch_size=1", "--set=numerical.convergence.max_iters=7"]) == 0
    report = json.loads((output / "preprocessing.json").read_text())
    assert report["case_count"] == 2
    assert report["batch_size"] == 1
    assert report["mach"] == [.35]
    assert report["alpha_deg"] == [2.]
    assert load_case(output / "sections/section_000").numerics.max_iters == 7
    assert len(list(output.glob("sections/*/profile.hug"))) == 2
    assert len(list(output.glob("sections/*/geometry.png"))) == 2
    assert not (output / "calculations").exists()
    stdout = capsys.readouterr().out
    assert "ICMS campaign" in stdout and "section 001:" in stdout and "check complete" in stdout


def test_cli_campaign_runs_the_gpu_api_by_default(prepared, monkeypatch, capsys):
    from types import SimpleNamespace
    from jaxrans.__main__ import main
    import jaxrans.campaign as campaign_module

    calls = []

    def prepare(campaign, output):
        assert campaign.directory == prepared.campaign.directory
        return prepared

    def run(prepared_arg, *, backend="gpu", progress=None):
        calls.append((prepared_arg, backend))
        assert progress is not None
        progress(2, ("converged", "max_iters"))
        return SimpleNamespace(directory=prepared.directory,
                               rows=[{"exit_reason": "converged"}, {"exit_reason": "max_iters"}])

    monkeypatch.setattr(campaign_module, "prepare_campaign", prepare)
    monkeypatch.setattr(campaign_module, "run_campaign", run)
    assert main([str(prepared.campaign.directory)]) == 0
    assert calls == [(prepared, "gpu")]
    stdout = capsys.readouterr().out
    assert "batch iteration" in stdout and "campaign complete: 2 cases" in stdout


@pytest.mark.parametrize("reason", ["non_finite", "stalled"])
def test_cli_campaign_quiet_and_failed_case_exit_code(prepared, monkeypatch, capsys, reason):
    from types import SimpleNamespace
    from jaxrans.__main__ import main
    import jaxrans.campaign as campaign_module

    monkeypatch.setattr(campaign_module, "prepare_campaign", lambda *args: prepared)

    def run(prepared_arg, *, progress):
        assert progress is None
        return SimpleNamespace(directory=prepared.directory, rows=[{"exit_reason": reason}])

    monkeypatch.setattr(campaign_module, "run_campaign", run)
    assert main([str(prepared.campaign.directory), "--quiet"]) == 1
    stdout = capsys.readouterr().out
    assert "section 000:" not in stdout and f"{reason}=1" in stdout


@pytest.mark.parametrize("arguments, message", [
    (["--devices", "2"], "--devices must be 1"),
    (["--set=case.sections.count=1"], "sections.count"),
    (["--set=case.campaign.mach=[]"], "campaign.mach"),
])
def test_cli_campaign_configuration_errors_are_reported(tmp_path, capsys, arguments, message):
    from jaxrans.__main__ import main

    write_campaign(tmp_path)
    assert main([str(tmp_path), *arguments]) == 2
    assert message in capsys.readouterr().err


def test_cli_malformed_json_is_reported_without_traceback(tmp_path, capsys):
    from jaxrans.__main__ import main

    (tmp_path / "case.json").write_text("{")
    assert main([str(tmp_path)]) == 2
    assert "error:" in capsys.readouterr().err
