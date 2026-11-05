"""Known profile integrals, conical coordinates and isolated normal grids."""

import numpy as np
import pytest

from jaxrans.case import parse_case
from jaxrans.core.sectional import SectionalSpec
from jaxrans.geometry.conical import ConicalGeometry
from validation.sectional_boundary_layer import (
    SectionSampler, edge_metrics, first_gap_for_yplus, foot_points,
    layers_through_thickness, reference_speed,
    layer_design,
)
from validation.sectional_normal_refinement import normal_grid


@pytest.mark.parametrize("taper", [0., -.2, .2])
def test_inverse_conical_map_recovers_generator_feet(taper):
    geometry = ConicalGeometry(SectionalSpec(leading_edge_sweep_deg=30., taper_rate=taper), 1., .3)
    points = np.array([[.4, .7, .9], [.02, .1, -.04], [-.1, 0., .1]])
    physical = geometry.map_points(points, np.ones(3))
    np.testing.assert_allclose(foot_points(physical, geometry), points[:2], atol=1e-14)


def test_exponential_profile_recovers_known_edge_and_integrals():
    scale = .001
    d = np.r_[0., np.geomspace(1e-8, .03, 4000)]
    q = np.zeros((6, len(d)))
    q[0], q[4] = 1., 1 / 1.4
    q[1] = .5 * (1 - np.exp(-d / scale))
    out, _, _ = edge_metrics(d, q, np.array([0., 1., 0.]))
    assert out["status"] == "estimated"
    assert out["delta99_speed"] == pytest.approx(scale * np.log(100), rel=1e-5)
    assert out["delta_star_scalar"] == pytest.approx(.99 * scale, rel=1e-5)
    assert out["theta_scalar"] == pytest.approx(.49005 * scale, rel=2e-5)
    assert out["max_crossflow_over_ui_in_layer"] == 0


def test_edge_is_not_invented_if_the_field_does_not_reach_outer_flow():
    d = np.linspace(0, .02, 100)
    q = np.zeros((6, len(d)))
    q[0], q[4], q[1] = 1., 1 / 1.4, .2
    out, _, _ = edge_metrics(d, q, np.array([0., 1., 0.]))
    assert out["status"] == "edge_not_found" and "delta99_speed" not in out


def test_a_transient_crossing_is_rejected_if_outer_reference_is_not_recovered():
    d = np.linspace(0., .03, 3000)
    q = np.zeros((6, len(d)))
    q[0], q[4] = 1., 1 / 1.4
    q[1] = .5 * (1 - np.exp(-d / .001))
    q[1, d > .005] *= .8
    out, _, _ = edge_metrics(d, q, np.array([0., 1., 0.]))
    assert out["status"] == "reference_not_recovered" and "delta99_speed" not in out


def test_thickness_diagnostics_are_invariant_under_cartesian_rotation():
    d = np.r_[0., np.geomspace(1e-8, .03, 2000)]
    f = 1 - np.exp(-d / .001)
    q = np.zeros((6, len(d)))
    q[0], q[4], q[1], q[3] = 1., 1 / 1.4, .5 * f, .08 * f * (1 - f)
    n = np.array([0., 1., 0.])
    axis = np.array([1., 2., 3.]) / np.sqrt(14)
    x, y, z = axis
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    rotation = np.eye(3) + np.sin(.7) * k + (1 - np.cos(.7)) * k @ k
    rotated = q.copy()
    rotated[1:4] = rotation @ q[1:4]
    a, _, _ = edge_metrics(d, q, n)
    b, _, _ = edge_metrics(d, rotated, rotation @ n)
    assert a["status"] == b["status"] == "estimated"
    assert a["max_crossflow_over_ui_in_layer"] > .01
    for key in ("delta99_speed", "delta_vector_1pct", "delta_star_scalar", "theta_scalar"):
        assert a[key] == pytest.approx(b[key], rel=1e-11)


def test_unphysical_reference_pressure_stays_invalid():
    assert np.isnan(reference_speed(np.array([2.]), np.array([0.]))[0])


def test_first_gap_accounts_for_reynolds_and_the_cell_centre_factor_two():
    gap = first_gap_for_yplus(1e6, .005)
    recovered = .5 * gap * 1e6 * np.sqrt(.005 / 2)
    assert recovered == pytest.approx(.5)
    assert first_gap_for_yplus(2e6, .005) == pytest.approx(gap / 2)
    n = layers_through_thickness(.02, gap)
    assert gap * (1.15**n - 1) / .15 >= .02
    assert gap * (1.15**(n - 1) - 1) / .15 < .02


def test_sizing_refuses_nonphysical_inputs():
    with pytest.raises(ValueError):
        first_gap_for_yplus(0, .005)
    with pytest.raises(ValueError):
        layers_through_thickness(.02, 1e-5, .9)


def test_layer_design_enforces_both_wall_units_and_thinnest_layer_resolution():
    d = layer_design(1e6, .015, .00065, .04)
    assert d["first_gap"] <= d["wall_unit_gap_limit"]
    assert d["first_gap"] * (d["growth"]**30 - 1) / (d["growth"] - 1) <= .00065 * (1 + 1e-12)
    n = d["layers_to_cover_delta_max"]
    assert d["first_gap"] * (d["growth"]**n - 1) / (d["growth"] - 1) >= .04


def case():
    return parse_case({"general": {"case_name": "normal_test"},
                       "domain": {"grid": "c_grid", "airfoil": "naca0012_blunt", "cells_airfoil": 128, "cells_wake": 32,
                                  "cells_normal": 96, "wall_spacing": 1e-5, "far_field": 100.,
                                  "sectional": {"model": "conical", "leading_edge_sweep_deg": 30., "taper_rate": .2}},
                       "freestream": {"mach": .5, "reynolds": 1e6}, "turbulence": {"model": "sa_noft2"}},
                      {"sectional": {"depth": .1}})


def test_normal_refinement_keeps_surface_and_outer_nodes_exactly():
    grids = [normal_grid(case(), stride) for stride in (4, 2, 1)]
    for grid in grids:
        np.testing.assert_array_equal(grid.mesh[0].nodes[:, :, 0], grids[0].mesh[0].nodes[:, :, 0])
        np.testing.assert_array_equal(grid.mesh[0].nodes[:, :, -1], grids[0].mesh[0].nodes[:, :, -1])
    from jaxrans.mesh.mesh import interface_node_mismatch
    for grid in grids:
        assert len(grid.mesh) == 3
        assert interface_node_mismatch(grid.reference.mesh) < 1e-12
        for a, b in zip(grid.mesh[1:], grids[0].mesh[1:]):
            np.testing.assert_array_equal(a.nodes, b.nodes)
    assert [g.mesh[0].shape[:2] for g in grids] == [(144, 24), (144, 48), (144, 96)]


def test_structured_interpolation_recovers_an_affine_scalar_on_a_generator():
    grid = case().grid()
    nodes = np.asarray(grid.mesh[0].nodes)
    centres = np.asarray(grid.mesh[0].centres)[..., 0]
    q = np.zeros((6,) + centres.shape[1:])
    q[0] = 1 + .01 * centres[0] + .02 * centres[1]
    q[4] = 1 / 1.4
    sampler = SectionSampler({"nodes": nodes, "primitive": q}, grid.geometry)
    point = .3 * centres[:, 3, 4] + .3 * centres[:, 4, 4] + .4 * centres[:, 3, 5]
    physical = grid.geometry.map_points(np.array([[point[0]], [point[1]], [.05]]), np.ones(1))
    sampled = sampler.sample(physical)
    assert sampled[0, 0] == pytest.approx(1 + .01 * point[0] + .02 * point[1], abs=1e-13)


@pytest.mark.parametrize("taper", [0., .2])
def test_diagnostic_uses_actual_c_grid_wall_columns_and_supports_zero_taper(tmp_path, taper):
    from types import SimpleNamespace
    from validation.sectional_baseline import archive_arrays
    from validation.sectional_boundary_layer import diagnose
    from validation.sectional_qualification import write_json
    from jaxrans.numerics.residual import make_residual
    from jaxrans.post.forces import surface

    original = case()
    doc = original.raw["case"]
    doc["domain"]["sectional"]["taper_rate"] = taper
    c = parse_case(doc, original.raw["numerical"])
    grid = c.grid(coarsen=2)
    U = make_residual(grid, c.flow, c.numerics).initial_state()
    result = SimpleNamespace(grid=grid, case=c, U=U,
                             history=SimpleNamespace(residual=np.array([1.])),
                             surface=lambda: surface(U, grid, c.flow))
    saved = archive_arrays(result)
    write_json(tmp_path / "case.json", doc)
    write_json(tmp_path / "numerical.json", original.raw["numerical"])
    np.savez_compressed(tmp_path / "arrays.npz", **saved)
    report = diagnose(tmp_path, tmp_path / "profiles", stations=(.1, .5, .9))
    assert len(report["profiles"]) == 6
    extracted = [p for p in report["profiles"] if "wall_face" in p]
    assert len(extracted) >= 4
    for p in extracted:
        face = p["wall_face"]
        assert saved["wall_block"][face] == 0
        assert saved["wall_region"][face] != 2
        assert p["actual_x"] == saved["x_streamwise"][face]
