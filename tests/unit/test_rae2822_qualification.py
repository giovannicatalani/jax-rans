"""Qualification planning, archive validation and reporting without a CFD solve."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jaxrans.case import Case, load_case


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("qualify_rae2822", ROOT / "scripts/qualify_rae2822.py")
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


@pytest.fixture
def case_dir(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    (directory / "case.json").write_text(json.dumps({
        "general": {"case_name": "witness", "precision": "fp64"},
        "domain": {"grid": "o_grid", "airfoil": "naca0012", "cells_airfoil": 16,
                   "cells_normal": 8, "wall_spacing": .001, "far_field": 10.},
        "freestream": {"mach": .729, "alpha_deg": 2.31, "reynolds": 6.5e6, "sweep_deg": 0.},
        "turbulence": {"model": "sa_noft2"}}))
    (directory / "numerical.json").write_text(json.dumps({
        "time_integration": {"solver": "implicit"},
        "convective_fluxes": {"limiter": "van_albada_smooth"},
        "implicit": {"max_relative_update": .2},
        "convergence": {"forces_tol": None, "block_iters": 20}}))
    return directory


def args(case_dir, tmp_path, *extra):
    return q.parser().parse_args(["--case", str(case_dir), "--out", str(tmp_path / "out"), *extra])


def test_defaults_are_one_bounded_job_and_do_not_build_mesh(case_dir, tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("planning freestream jobs must not build a grid or solve")
    monkeypatch.setattr(q, "run", forbidden)
    monkeypatch.setattr(Case, "grid", forbidden)
    manifest, _ = q.build_plan(args(case_dir, tmp_path))
    assert manifest["job_count"] == 1
    job = manifest["jobs"][0]
    assert job["id"] == "floor__baseline__fp32"
    assert job["config"]["flow"]["mach"] == .745
    assert job["config"]["flow"]["alpha_deg"] == -1.5
    assert job["criteria"]["max_iters"] == 200
    assert job["criteria"]["forces_tol"] == .0001
    assert job["config"]["numerics"]["diagnostics"] is True
    assert job["raw"]["numerical"]["convergence"]["diagnostics"] is True
    assert job["config"]["implicit"]["nonlinear_merit"] == "mass"
    assert job["config"]["implicit"]["backtracking_steps"] == 0
    assert job["initial"] == {"kind": "freestream", "continuation_reproduction": False}
    assert len(job["config_sha256"]) == 64
    assert manifest["source_sha256"]["scripts/qualify_rae2822.py"]
    assert not (tmp_path / "out").exists()


def test_matrix_is_explicit_and_variants_are_not_cumulative(case_dir, tmp_path):
    manifest, _ = q.build_plan(args(case_dir, tmp_path, "--witness", "all", "--variant", "all",
                                    "--precision", "fp32", "fp64"))
    assert manifest["job_count"] == len(q.WITNESSES) * len(q.VARIANTS) * 2
    assert {j["witness"] for j in manifest["jobs"]} == set(q.WITNESSES)
    baseline = manifest["jobs"][0]["config"]
    for job in manifest["jobs"]:
        assert job["config"]["flow"]["mach"] == q.WITNESSES[job["witness"]]["mach"]
        assert job["config"]["flow"]["alpha_deg"] == q.WITNESSES[job["witness"]]["alpha_deg"]
        actual = job["config"]["implicit"]
        expected = dict(baseline["implicit"])
        for key, value in q.VARIANTS[job["variant"]].items():
            if key.startswith("implicit."):
                expected[key.split(".")[1]] = value
        assert actual == expected
    groups = {}
    for job in manifest["jobs"]:
        groups.setdefault(job["witness"], set()).add(job["comparison_group"])
    assert all(len(group) == 1 for group in groups.values())
    assert len({next(iter(group)) for group in groups.values()}) == 5


def test_repeatable_set_preserves_plain_2d_and_common_criteria(case_dir, tmp_path):
    before = {p.name: p.read_bytes() for p in case_dir.iterdir()}
    opts = args(case_dir, tmp_path, "--set", "case.freestream.sweep_deg=null",
                "--set", "numerical.convergence.forces_tol=0.0002",
                "--set", "numerical.implicit.cfl_max=75", "--variant", "baseline", "minmod")
    manifest, cases = q.build_plan(opts)
    for job in manifest["jobs"]:
        assert cases[job["id"]].flow.sweep_deg is None
        assert job["criteria"]["forces_tol"] == .0002
        assert job["config"]["implicit"]["cfl_max"] == 75
    assert {p.name: p.read_bytes() for p in case_dir.iterdir()} == before
    other, _ = q.build_plan(args(case_dir, tmp_path))
    assert manifest["jobs"][0]["comparison_group"] != other["jobs"][0]["comparison_group"]


def test_common_merit_backtracking_and_equation_floors_are_traceable(case_dir, tmp_path):
    manifest, _ = q.build_plan(args(case_dir, tmp_path,
        "--set", "numerical.implicit.nonlinear_merit=equations",
        "--set", "numerical.implicit.backtracking_steps=3",
        "--set", 'numerical.convergence.equation_abs_tols={"sa":1e-7}',
        "--variant", "baseline", "nonlinear_growth_guard"))
    for job in manifest["jobs"]:
        assert job["config"]["implicit"]["nonlinear_merit"] == "equations"
        assert job["config"]["implicit"]["backtracking_steps"] == 3
        assert dict(job["criteria"]["equation_abs_tols"]) == {"sa": 1e-7}
    assert len({j["comparison_group"] for j in manifest["jobs"]}) == 1


@pytest.mark.parametrize("extra,match", [
    (("--max-iters", "0"), "positive"),
    (("--max-iters", "-1"), "positive"),
    (("--max-iters", "21"), "multiple"),
    (("--set", "numerical.implicit.no_such_knob=1"), "unknown key"),
    (("--set", "case.general.precision=fp16"), "precision"),
    (("--set", "numerical.time_integration.solver=explicit"), "implicit"),
    (("--set", "numerical.convergence.grid_sequencing=1"), "initial fields"),
])
def test_invalid_plan_inputs(case_dir, tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        q.build_plan(args(case_dir, tmp_path, *extra))


@pytest.mark.parametrize("extra", [("--witness", "unknown"), ("--variant", "unknown"),
                                    ("--precision", "fp16"), ("--max-iters", "no"),
                                    ("--dry-run", "--manifest-only")])
def test_cli_rejects_invalid_selections(case_dir, tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        args(case_dir, tmp_path, *extra)
    assert error.value.code == 2


def test_dry_run_and_manifest_only_never_execute_and_preserve_existing_files(case_dir, tmp_path,
                                                                           monkeypatch, capsys):
    monkeypatch.setattr(q, "run", lambda *a, **kw: pytest.fail("no CFD in a plan"))
    base = ["--case", str(case_dir), "--out", str(tmp_path / "out")]
    assert q.main([*base, "--dry-run"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["job_count"] == 1
    assert not (tmp_path / "out").exists()
    assert q.main([*base, "--manifest-only"]) == 0
    path = tmp_path / "out/manifest.json"
    original = path.read_bytes()
    assert list((tmp_path / "out").iterdir()) == [path]
    with pytest.raises(SystemExit):
        q.main([*base, "--manifest-only"])
    assert path.read_bytes() == original


class FakeGrid:
    """Small non-CFD physical mesh and a canvas with gaps/ghosts."""
    components = 2

    def __init__(self):
        nodes = np.stack(np.meshgrid(np.arange(3.), np.arange(3.), indexing="ij"))
        self.mesh = [SimpleNamespace(nodes=nodes)]

    def pack(self, blocks):
        canvas = np.zeros((blocks[0].shape[0], 4, 4), dtype=blocks[0].dtype)
        canvas[:, 1:3, 1:3] = blocks[0]
        return canvas


def archive(tmp_path, grid, mutate=None, name="historical_fields.npz"):
    U = np.zeros((5, 2, 2), np.float64)
    U[0], U[1], U[3], U[4] = 1., .1, 3., -1.e-6  # SA-neg is valid.
    values = {"n_blocks": np.array(1), "U_0": U,
              "x_node_0": grid.mesh[0].nodes[0].copy(),
              "y_node_0": grid.mesh[0].nodes[1].copy()}
    if mutate:
        mutate(values)
    path = tmp_path / name
    np.savez(path, **values)
    return path


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_restart_casts_physical_U_without_ghosts_controls_or_clipping(case_dir, tmp_path, precision):
    case = load_case(case_dir, ("case.freestream.sweep_deg=null", f"case.general.precision={precision}"))
    grid = FakeGrid()
    path = archive(tmp_path, grid)
    initial, record = q.load_initial(path, grid, case)
    assert initial.dtype == case.numerics.dtype
    assert np.all(initial[:, 0] == 0)
    assert np.all(initial[:, :, 0] == 0)
    np.testing.assert_array_equal(initial[4, 1:3, 1:3], np.full((2, 2), -1e-6, case.numerics.dtype))
    assert record["source_blocks"][0]["dtype"] == "float64"
    assert "solver controls and histories reset" in record["restart"]


@pytest.mark.parametrize("mutation,match", [
    (lambda v: v.update(n_blocks=np.array(2)), "n_blocks"),
    (lambda v: v.update(n_blocks=np.array([1])), "n_blocks"),
    (lambda v: v["x_node_0"].__setitem__((0, 0), 1e-15), "exact mesh"),
    (lambda v: v["U_0"].__setitem__((0, 0, 0), 0), "density"),
    (lambda v: v["U_0"].__setitem__((3, 0, 0), 0), "pressure"),
    (lambda v: v["U_0"].__setitem__((4, 0, 0), np.nan), "nonfinite"),
    (lambda v: v["U_0"].__setitem__((4, 0, 0), 1e300), "nonfinite"),
    (lambda v: v.update(U_0=np.ones((6, 2, 2))), "shape"),
    (lambda v: v.update(U_0=np.ones((5, 4, 4))), "physical U"),
    (lambda v: v.update(U_0=np.ones((5, 2, 2), dtype=int)), "floating"),
    (lambda v: v.pop("U_0"), "missing U_0"),
    (lambda v: v.pop("y_node_0"), "exact mesh"),
    (lambda v: v.update(cfl=np.array(10.)), "only Result.write"),
])
def test_restart_rejects_mismatched_or_nonphysical_archives(case_dir, tmp_path, mutation, match):
    case = load_case(case_dir, ("case.freestream.sweep_deg=null", "case.general.precision=fp32"))
    grid = FakeGrid()
    with pytest.raises(ValueError, match=match):
        q.load_initial(archive(tmp_path, grid, mutation), grid, case)


def test_historical_5var_archive_rejects_lifted_6var_case(case_dir, tmp_path):
    grid = FakeGrid()
    grid.components = 3
    with pytest.raises(ValueError, match="shape.*6"):
        q.load_initial(archive(tmp_path, grid), grid, load_case(case_dir))


def test_saved_initial_plan_tracks_file_and_same_initial_for_both_precisions(case_dir, tmp_path,
                                                                          monkeypatch):
    grid = FakeGrid()
    monkeypatch.setattr(Case, "grid", lambda self: grid)
    path = archive(tmp_path, grid)
    manifest, _ = q.build_plan(args(case_dir, tmp_path, "--initial", str(path), "--precision", "fp32", "fp64",
                                    "--set", "case.freestream.sweep_deg=null"))
    a, b = manifest["jobs"]
    assert a["comparison_group"] == b["comparison_group"]
    assert a["initial"]["physical_U_sha256"] == b["initial"]["physical_U_sha256"]
    assert a["initial"]["path"] == str(path.resolve())
    assert a["initial"]["sha256"] == q.file_record(path)["sha256"]
    assert a["initial"]["field_dtype"] == "float32"
    assert b["initial"]["field_dtype"] == "float64"


def pair():
    a = {"id": "baseline", "comparison_group": "same", "variant": "baseline", "precision": "fp32",
         "CL": 1., "CD": .01, "CM": -.1, "seconds": 2., "exit_reason": "max_iters"}
    b = dict(a, id="guard", variant="nonlinear_growth_guard", CL=1.2, seconds=3.)
    wall = {"x": np.array([0., 1.]), "y": np.zeros(2), "body": np.zeros(2),
            "cp": np.array([1., 2.]), "cf": np.array([.01, .02])}
    surfaces = {a["id"]: wall, b["id"]: dict(wall, cp=np.array([1.1, 2.2]))}
    return a, b, surfaces


def test_comparisons_report_differences_without_an_acceptance_threshold():
    a, b, surfaces = pair()
    comparison = q.compare(a, b, surfaces)
    assert comparison["delta"]["CL"] == pytest.approx(.2)
    assert comparison["surface_delta"]["cp"]["linf"] == pytest.approx(.2)
    assert comparison["surface_delta"]["cf"]["linf"] == 0
    assert comparison["exit_reasons"] == ["max_iters", "max_iters"]
    assert "passed" not in comparison
    b.update(variant="baseline", precision="fp64")
    assert q.compare(a, b, surfaces)["factor"] == "precision"


def test_comparisons_refuse_unlike_initials_criteria_multiple_factors_or_surface_order():
    a, b, surfaces = pair()
    with pytest.raises(ValueError, match="different initial"):
        q.compare(a, dict(b, comparison_group="other"), surfaces)
    with pytest.raises(ValueError, match="exactly one"):
        q.compare(a, dict(b, precision="fp64"), surfaces)
    surfaces[b["id"]]["x"] = np.array([1., 0.])
    with pytest.raises(ValueError, match="coordinates/order"):
        q.compare(a, b, surfaces)


def test_fake_execution_writes_every_result_and_summary(case_dir, tmp_path, monkeypatch):
    manifest, cases = q.build_plan(args(case_dir, tmp_path, "--variant", "baseline", "minmod"))
    out = tmp_path / "out"
    out.mkdir()
    grid = FakeGrid()
    calls = []
    monkeypatch.setattr(Case, "grid", lambda self: grid)

    def fake_run(case, *, grid, initial):
        assert initial is None  # Every variant starts independently, no warm start from the last result.
        wall = pair()[2]["baseline"]
        reason = "settled" if case.numerics.limiter == "minmod" else "max_iters"
        def write(directory, *, extra):
            calls.append(extra["qualification"]["id"])
            np.savetxt(directory / "witness_surface.dat", np.column_stack([wall["cp"], wall["cf"]]),
                       header="cp cf")
        return SimpleNamespace(
            seconds=1., timing=SimpleNamespace(record=lambda: {"total_s": 1., "compile_s": float("nan")}),
            forces=SimpleNamespace(cl=1., cd=.01, cm=-.1), U=np.ones((5, 2, 2), np.float32),
            history=SimpleNamespace(n_iters=20, reason=reason, block_diagnostics=({"accepted_count": 20},)),
            convergence_history=({"blocking_equations": ["sa"], "forces_settled": False,
                                  "settled": False},), surface=lambda: wall, write=write)
    monkeypatch.setattr(q, "run", fake_run)
    assert q.execute(manifest, cases) == 1  # A bounded nonconverged run is not labelled successful.
    assert calls == [j["id"] for j in manifest["jobs"]]
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["runs"]) == 2
    assert len(summary["comparisons"]) == 1
    assert summary["runs"][0]["blockers"] == ["sa", "forces"]
    assert summary["runs"][0]["timing"]["compile_s"] is None
    for job in manifest["jobs"]:
        directory = Path(job["directory"])
        assert (directory / "summary.json").is_file()
        assert (directory / "witness_surface.dat").is_file()
        assert json.loads((directory / "job.json").read_text())["config_sha256"] == job["config_sha256"]


def test_execution_failure_is_recorded_and_does_not_drop_next_job(case_dir, tmp_path, monkeypatch):
    manifest, cases = q.build_plan(args(case_dir, tmp_path, "--variant", "baseline", "minmod"))
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(Case, "grid", lambda self: FakeGrid())
    monkeypatch.setattr(q, "run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("test failure")))
    assert q.execute(manifest, cases) == 1
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert len(summary["runs"]) == 2
    assert all(r["blockers"] == ["test failure"] for r in summary["runs"])
    assert summary["comparisons"] == []


def test_replacing_initial_after_planning_is_refused(case_dir, tmp_path, monkeypatch):
    grid = FakeGrid()
    monkeypatch.setattr(Case, "grid", lambda self: grid)
    monkeypatch.setattr(q, "run", lambda *a, **kw: pytest.fail("changed initial must not reach CFD"))
    path = archive(tmp_path, grid)
    manifest, cases = q.build_plan(args(case_dir, tmp_path, "--initial", str(path),
                                        "--set", "case.freestream.sweep_deg=null"))
    archive(tmp_path, grid, lambda v: v["U_0"].__setitem__((1, 0, 0), .2))
    (tmp_path / "out").mkdir()
    assert q.execute(manifest, cases) == 1
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["runs"][0]["blockers"] == ["initial file changed since planning"]
