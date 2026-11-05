"""Planform checks, input limits and runnable finite-taper sections."""

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


@pytest.mark.parametrize("sweep,taper", [(30., .4), (30., -.4), (-20., .2)])
def test_streamwise_traction_scale_matches_a_differentiated_conical_field(sweep, taper):
    geom = ConicalGeometry(SectionalSpec(leading_edge_sweep_deg=sweep, taper_rate=taper), 2., .3)
    xy = np.array([geom.normal_leading_edge_x + .7, .1])
    stream = geom.streamwise_points(xy)
    delta = stream[0] - geom.leading_edge_x
    central = np.r_[xy, 0.]
    target = np.array([geom.normal_leading_edge_x + np.cos(geom.angle) * delta,
                       stream[1], np.sin(geom.angle) * delta])
    p, mu = geom.coefficients

    def scalar(P):
        x = (P[0] - (p + mu * geom.normal_leading_edge_x) * P[2]) / (1 - mu * P[2])
        y = P[1] / (1 - mu * P[2])
        return np.sin(x) * np.cos(y)

    def derivative(P):
        eps = 1e-5
        return np.array([(scalar(P + eps * e) - scalar(P - eps * e)) / (2 * eps) for e in np.eye(3)])

    assert scalar(target) == pytest.approx(scalar(central), abs=1e-14)
    np.testing.assert_allclose(derivative(target),
                               geom.streamwise_gradient_scale(xy) * derivative(central),
                               rtol=1e-8, atol=1e-10)


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


@pytest.mark.parametrize("model", [None, "sa_noft2", "sa_noft2_qcr"])
def test_viscous_finite_taper_builds_with_the_physical_wall_distance(model):
    doc = document(taper_rate=0.2)
    doc["freestream"]["reynolds"] = 1e6
    if model:
        doc["turbulence"] = {"model": model}
    case = parse_case(doc, {"sectional": {"depth": 0.1}})
    from jaxrans.numerics.residual import make_residual
    op = make_residual(case.grid(), case.flow, case.numerics)
    assert op.viscosity is not None
    if model:
        assert np.all(np.asarray(op.distance)[op.atlas.interior] > 0)


def test_finite_taper_rejects_the_unextended_point_vortex_model():
    doc = document(taper_rate=0.2)
    doc["boundary_conditions"] = {"farfield": {"type": "far_field", "model": "point_vortex"}}
    with pytest.raises(CaseError, match="uniform characteristic"):
        parse_case(doc, {})


def test_an_extrusion_crossing_the_apex_is_refused():
    with pytest.raises(ValueError, match="apex"):
        parse_case(document(taper_rate=0.2), {"sectional": {"depth": 20.0}}).grid()


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
@pytest.mark.parametrize("physics", ["euler", "laminar", "sa", "qcr"])
def test_a_finite_taper_case_runs_and_writes_physical_section_outputs(tmp_path, solver, physics):
    import json
    from jaxrans import run

    doc = document(taper_rate=0.2)
    if physics != "euler":
        doc["freestream"]["reynolds"] = 1e6
    if physics in ("sa", "qcr"):
        doc["turbulence"] = {"model": "sa_noft2_qcr" if physics == "qcr" else "sa_noft2"}
    case = parse_case(doc, {
        "sectional": {"depth": 0.1}, "time_integration": {"solver": solver},
        "convergence": {"max_iters": 3, "block_iters": 1, "forces_tol": None},
        "implicit": {"krylov_subspace": 3, "krylov_restarts": 1},
    }, tmp_path)
    result = run(case)
    assert result.history.n_iters == 3
    assert np.isfinite(np.asarray(result.U)).all() and np.isfinite(result.forces.cl)
    assert "y_streamwise" in result.surface()
    if physics != "euler":
        assert np.isfinite(result.forces.cd_friction)
        assert np.isfinite(result.surface()["cf_z"]).all()
    out = result.write()
    metadata = json.loads((out / "sectional_metadata.json").read_text())["extra"]["sectional"]
    assert metadata["streamwise_chord"] == pytest.approx(1.0)
    assert metadata["relaxation_law"] == "quintic_wall_distance"
