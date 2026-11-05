"""Qualification must not pass missing, unconverged or stale comparisons."""

from copy import deepcopy

import numpy as np
import pytest

from jaxrans.case import parse_case
from validation import sectional_qualification as q


def arrays():
    arc = np.array([.1, .3, .7, .9])
    return {"primitive": np.ones((6, 2, 2)), "x": np.array([.8, .2, .2, .8]),
            "y": np.array([-.02, -.04, .04, .02]), "cp": np.arange(4.) / 10,
            "cf": np.array([.01, .02, .03, .04]), "weight": np.ones(4), "arc": arc}


def put_run(output, job, *, reason="converged", state=None, source=None):
    folder = output / job["id"]
    folder.mkdir(parents=True)
    q.write_json(folder / "result.json", {"job": job, "source_sha256": source or q.source_fingerprint(),
                 "reason": reason, "finite": True, "forces": dict.fromkeys(q.COEFFICIENTS, .1)})
    np.savez(folder / "arrays.npz", **(arrays() if state is None else state))


def test_limit_comparison_uses_streamwise_coordinates_and_zero_spanwise_friction():
    ref, section = arrays(), arrays()
    section["x_streamwise"], section["x"] = section["x"], .8 * section["x"]
    section["cf_z"] = np.zeros(4)
    assert q.compare_limits(ref, section)["cp"]["linf"] == 0
    section["primitive"][5, 0, 0] += 1.
    assert max(q.compare_limits(ref, section)["field_relative_linf_by_row"]) == 1.


@pytest.mark.parametrize("failure", ["missing", "unconverged", "pressure", "sa_field", "coordinates"])
def test_equal_forces_are_insufficient_to_pass_a_limit(tmp_path, failure):
    jobs = q.plan("limits", ("sa",), (0.,))
    put_run(tmp_path, jobs[0])
    if failure != "missing":
        state = arrays()
        if failure == "pressure":
            state["cp"][0] += .01
        if failure == "sa_field":
            state["primitive"][5] += .01
        if failure == "coordinates":
            state["y"] += .01
        put_run(tmp_path, jobs[1], reason="max_iters" if failure == "unconverged" else "converged", state=state)
    if failure == "coordinates":
        with pytest.raises(AssertionError):
            q.report(jobs, tmp_path)
    else:
        assert not q.report(jobs, tmp_path)["limits"][0]["passed"]


def test_identical_converged_fields_and_forces_pass(tmp_path):
    jobs = q.plan("limits", ("sa",), (30.,))
    for job in jobs:
        put_run(tmp_path, job)
    assert q.report(jobs, tmp_path)["limits"][0]["passed"]


@pytest.mark.parametrize("wrong", ["source", "inputs"])
def test_reports_refuse_stale_results(tmp_path, wrong):
    jobs = q.plan("limits", ("sa",), (30.,))
    stored = deepcopy(jobs[0])
    if wrong == "inputs":
        stored["tol"] = 1e-5
    put_run(tmp_path, stored, source="old" if wrong == "source" else None)
    with pytest.raises(ValueError, match="do not match"):
        q.report(jobs, tmp_path)


@pytest.mark.parametrize("values", [(1., .9, .95), (1., .9, .8), (1., .9, .7)])
def test_richardson_does_not_extrapolate_nonshrinking_or_oscillating_changes(values):
    fit = q.richardson_summary(values)
    assert fit["extrapolated"] is None and fit["order"] is None


def test_richardson_recovers_a_known_second_order_limit():
    fit = q.richardson_summary((1. + .4, 1. + .4 / 4, 1. + .4 / 16))
    assert fit["order"] == pytest.approx(2.) and fit["extrapolated"] == pytest.approx(1.)


def test_wall_interpolation_does_not_bridge_the_leading_edge():
    coarse, fine = arrays(), arrays()
    coarse["cf"] = np.array([-1., -1., 1., 1.])
    fine["arc"] = np.array([.15, .25, .75, .85])
    fine["cf"] = np.array([-1., -1., 1., 1.])
    assert q.surface_change(coarse, fine)["cf"]["linf"] == 0


def test_meshes_are_nested_and_outer_extension_preserves_the_complete_inner_grid():
    jobs = {j["id"]: j for j in q.plan("mesh")}
    coarse_job = jobs["mesh_sa_l2"]
    case = parse_case(*q.documents(coarse_job))
    coarse = q.build_grid(case, coarse_job)
    fine = q.build_grid(case, jobs["mesh_sa_l1"])
    np.testing.assert_array_equal(coarse.reference.mesh[0].nodes, fine.reference.mesh[0].nodes[:, ::2, ::2])
    outer = q.build_grid(case, jobs["outer_sa_2"])
    original, extended = fine.reference.mesh[0].nodes, outer.reference.mesh[0].nodes
    np.testing.assert_array_equal(original, extended[..., :original.shape[-1]])
    assert extended.shape[-1] > original.shape[-1]
    from jaxrans.mesh.mesh import interface_node_mismatch
    for a, b in zip(coarse.reference.mesh, fine.reference.mesh):
        np.testing.assert_array_equal(a.nodes, b.nodes[:, ::2, ::2])
    for a, b in zip(fine.reference.mesh[1:], outer.reference.mesh[1:]):
        np.testing.assert_array_equal(a.nodes, b.nodes)
    assert interface_node_mismatch(outer.reference.mesh) < 1e-12
    assert all(np.asarray(b.volume).min() > 0 for b in outer.mesh)


def test_campaign_cases_parse_and_only_the_reference_zero_sweep_uses_native_2d():
    for job in q.plan():
        case = parse_case(*q.documents(job))
        if job["formulation"] == "legacy" and job["sweep"] == 0:
            assert case.flow.sweep_deg is None and case.grid_spec.sectional is None
        if job["formulation"] == "sectional":
            assert case.grid_spec.sectional.taper_rate == job["taper"]


def test_human_report_keeps_missing_and_failed_pairs_visible(tmp_path):
    from validation.sectional_report import markdown
    jobs = q.plan("limits", ("sa",), (0., 30.))
    put_run(tmp_path, jobs[0])
    put_run(tmp_path, jobs[1], reason="max_iters")
    text = markdown(q.report(jobs, tmp_path))
    assert "FAIL" in text and "missing" in text and "PASS" not in text


def test_blunt_base_is_not_interpolated_with_the_airfoil_branches():
    a = arrays()
    a.update(wall_region=np.array([0, 1, 2, 2]), region_arc=np.array([.5, .5, .25, .75]))
    b = deepcopy(a)
    b["cp"][2:] += .2
    assert q.surface_change(a, b)["cp"]["linf"] == pytest.approx(.2)


@pytest.mark.parametrize("sweep", [0., 30.])
@pytest.mark.parametrize("physics", q.MODELS)
def test_blunt_c_grid_zero_taper_residual_and_complete_archive(physics, sweep):
    from types import SimpleNamespace
    from jaxrans.numerics.residual import make_residual
    from validation.sectional_baseline import archive_arrays
    from validation.sectional_boundary_layer import SectionSampler

    jobs = q.plan("limits", (physics,), (sweep,))
    cases = [parse_case(*q.documents(job)) for job in jobs]
    grids = [q.build_grid(case, job) for case, job in zip(cases, jobs)]
    operators = [make_residual(grid, case.flow, case.numerics) for grid, case in zip(grids, cases)]
    rng = np.random.default_rng(7)
    U = operators[0].initial_state()
    # Include nonuniform lateral momentum and SA in all three blocks.
    U = U * (1 + .001 * rng.standard_normal(U.shape))
    if grids[0].components == 2:
        import jax.numpy as jnp
        lifted = jnp.concatenate([U[:3], jnp.zeros_like(U[:1]), U[3:]])
    else:
        lifted = U
    U3 = grids[1].expand(lifted)
    plain = np.asarray(operators[0].residual(U))[:, grids[0].interior]
    actual = np.asarray(grids[1].section_values(operators[1].residual(U3)) / grids[1].depth)
    actual = actual[:, grids[0].interior]
    if grids[0].components == 2:
        assert np.max(abs(actual[3])) < 1e-13
        actual = actual[[0, 1, 2, *range(4, len(actual))]]
    np.testing.assert_allclose(actual, plain, atol=1e-11, rtol=1e-10)
    grid, case = grids[1], cases[1]
    dummy = SimpleNamespace(grid=grid, case=case, U=U3,
                            history=SimpleNamespace(residual=np.array([1.])), surface=lambda: {})
    saved = archive_arrays(dummy)
    assert saved["primitive"].shape == (len(lifted), grid.cells)
    assert len(saved["block_shapes"]) == 3
    assert np.count_nonzero(saved["wall_region"] == 2) == grid.reference.mesh[1].ni
    assert {0, 1} == set(saved["wall_block"])
    changed = deepcopy(saved)
    changed["primitive"][:, -1] += .1
    # The final cell belongs to the wake: it must affect the limit check.
    saved.update(x=np.zeros(1), y=np.zeros(1), cp=np.zeros(1), weight=np.ones(1))
    changed.update(x=np.zeros(1), y=np.zeros(1), cp=np.zeros(1), weight=np.ones(1))
    assert max(q.compare_limits(saved, changed)["field_relative_linf_by_row"]) > 0
    if physics != "euler":
        sampler = SectionSampler(saved, grid.geometry)
        # Interface values are not spuriously forced to no-slip.
        interface = grid.reference.mesh[0].sides.j_min[0]
        assert np.linalg.norm(sampler.values[1:4, interface.start * (grid.mesh[0].nj + 1)]) > 0
