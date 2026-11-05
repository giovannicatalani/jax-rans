"""Independent planform checks and input limits for finite-taper Euler sections."""

from dataclasses import replace

import numpy as np
import pytest

from jaxrans.case import CaseError, parse_case
from jaxrans.core.sectional import SectionalSpec
from jaxrans.geometry.conical import ConicalGeometry, relaxation_weight
from jaxrans.mesh.mesh import wall_faces
from tests.unit.test_sectional import document


@pytest.mark.parametrize("taper", [-0.2, 0.0, 0.2])
def test_both_edge_sweeps_follow_the_input_planform(taper):
    spec = SectionalSpec(leading_edge_sweep_deg=30.0, taper_rate=taper)
    geom = ConicalGeometry(spec, chord=2.0, leading_edge_x=0.0)
    angle = geom.angle
    # Actual leading/trailing edge intersections with the normal plane.
    end = 2.0 / (np.cos(angle) + np.sin(angle) * np.tan(np.deg2rad(spec.trailing_edge_sweep_deg)))
    x = np.array([0.0, end])
    points = np.stack([x, np.zeros(2), np.full(2, 0.05)])
    shifted = geom.map_points(points, np.ones(2))
    dx, dz = shifted[0] - x, shifted[2]
    global_dx = np.cos(angle) * dx + np.sin(angle) * dz
    global_ds = -np.sin(angle) * dx + np.cos(angle) * dz
    np.testing.assert_allclose(global_dx / global_ds,
                               np.tan(np.deg2rad([30.0, spec.trailing_edge_sweep_deg])), atol=1e-13)


def test_streamwise_profile_is_recovered_by_the_inverse_wall_mapping():
    spec = SectionalSpec(leading_edge_sweep_deg=30.0, taper_rate=0.2)
    doc = document(taper_rate=0.2)
    case = parse_case(doc, {"sectional": {"depth": 0.1}})
    mesh = case.grid_spec.build(case.directory)
    from jaxrans.mesh.mesh import assign_conditions
    mesh = assign_conditions(mesh, {"wall": "slip_wall", "farfield": "far_field"})
    geometry = ConicalGeometry.of(mesh, spec, "streamwise")
    normal = geometry.normal_section(mesh, 1.0, 5.0)
    original, transformed = wall_faces(mesh), wall_faces(normal)
    # Wall ordering is geometric; compare the point sets rather than assuming
    # that transforming the profile preserves which TE point starts the loop.
    restored = geometry.streamwise_points(transformed.a.T).T
    expected = original.a
    np.testing.assert_allclose(restored[np.argsort(restored[:, 1])],
                               expected[np.argsort(expected[:, 1])], atol=1e-12)


def test_relaxation_has_exact_plateaus_and_a_smooth_monotone_transition():
    x = np.linspace(0.0, 7.0, 1001)
    w = relaxation_weight(x, 1.0, 5.0)
    assert np.all(w[x <= 1] == 1) and np.all(w[x >= 5] == 0)
    assert np.all(np.diff(w) <= 1e-15)
    eps = 1e-4
    assert abs(float(relaxation_weight(np.array(1.0 + eps), 1, 5)) - 1) < 1e-11
    assert abs(float(relaxation_weight(np.array(5.0 - eps), 1, 5))) < 1e-11


@pytest.mark.parametrize("start,end", [(-1, 5), (2, 2), (3, 1), (0, float("inf"))])
def test_invalid_relaxation_is_refused(start, end):
    with pytest.raises(ValueError, match="relaxation"):
        parse_case(document(taper_rate=0.2), {"sectional": {"relaxation_start": start, "relaxation_end": end}})


def test_finite_taper_uses_the_mean_frame_and_preserves_its_input_specification():
    case = parse_case(document(taper_rate=0.2), {"sectional": {"depth": 0.1}})
    assert case.flow.sweep_deg == case.grid_spec.sectional.reference_sweep_deg
    assert case.flow.sweep_deg != 30.0
    atlas = case.grid()
    assert atlas.geometry.chord == pytest.approx(1.0)
    assert atlas.spec.leading_edge_sweep_deg == 30.0
    assert all(b.shape[-1] == 1 for b in atlas.mesh)


def test_viscous_finite_taper_is_reserved_for_stage_c():
    doc = document(taper_rate=0.2)
    doc["freestream"]["reynolds"] = 1e6
    with pytest.raises(CaseError, match="Euler only"):
        parse_case(doc, {})
    case = parse_case(document(taper_rate=0.2), {"sectional": {"depth": 0.1}})
    from jaxrans.numerics.residual import make_residual
    with pytest.raises(ValueError, match="Euler only"):
        make_residual(case.grid(), replace(case.flow, reynolds=1e6), case.numerics)


def test_finite_taper_rejects_the_unextended_point_vortex_model():
    doc = document(taper_rate=0.2)
    doc["boundary_conditions"] = {"farfield": {"type": "far_field", "model": "point_vortex"}}
    with pytest.raises(CaseError, match="uniform characteristic"):
        parse_case(doc, {})


def test_an_extrusion_crossing_the_apex_is_refused():
    with pytest.raises(ValueError, match="apex"):
        parse_case(document(taper_rate=0.2), {"sectional": {"depth": 20.0}}).grid()


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_a_finite_taper_euler_case_runs_and_writes_physical_section_outputs(tmp_path, solver):
    import json
    from jaxrans import run

    case = parse_case(document(taper_rate=0.2), {
        "sectional": {"depth": 0.1}, "time_integration": {"solver": solver},
        "convergence": {"max_iters": 3, "block_iters": 1, "forces_tol": None},
        "implicit": {"krylov_subspace": 3, "krylov_restarts": 1},
    }, tmp_path)
    result = run(case)
    assert result.history.n_iters == 3
    assert np.isfinite(np.asarray(result.U)).all() and np.isfinite(result.forces.cl)
    assert "y_streamwise" in result.surface()
    out = result.write()
    metadata = json.loads((out / "sectional_metadata.json").read_text())["extra"]["sectional"]
    assert metadata["streamwise_chord"] == pytest.approx(1.0)
    assert metadata["relaxation_law"] == "quintic_wall_distance"
