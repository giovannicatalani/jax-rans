"""Direct geometric fairing: objective gradient, displacement and shape bounds."""

from pathlib import Path

import numpy as np
import pytest

from jaxrans import load_case
from jaxrans.geometry.airfoil import read_airfoil
from jaxrans.geometry.curvature import _curvature_energy
from jaxrans.geometry.regularize import regularize_airfoil

ROOT = Path(__file__).resolve().parents[2]


def reference():
    return read_airfoil(ROOT / "validation/references/rae2822/rae2822.dat").unit_chord()


def test_geometric_energy_gradient_matches_independent_finite_differences():
    rng = np.random.default_rng(42)
    fields = [rng.normal(size=(4, 2)) for _ in range(3)]
    fields[0][:, 0] += 3
    weights = np.array([.2, .4, .1, .3])
    _, partials = _curvature_energy(*fields, weights)
    for field in range(3):
        for row in range(4):
            for axis in range(2):
                h = 1e-6
                fields[field][row, axis] += h
                upper = _curvature_energy(*fields, weights)[0]
                fields[field][row, axis] -= 2 * h
                lower = _curvature_energy(*fields, weights)[0]
                fields[field][row, axis] += h
                assert partials[field][row, axis] == pytest.approx((upper - lower) / (2 * h), rel=1e-5, abs=1e-8)


def test_curvature_objective_is_geometric_not_parametric():
    rng = np.random.default_rng(7)
    v, a, j = [rng.normal(size=(5, 2)) for _ in range(3)]
    v[:, 0] += 4
    weights = np.ones(5)
    original = _curvature_energy(v, a, j, weights)[0]
    rescaled_parameter = _curvature_energy(2 * v, 4 * a, 8 * j, weights / 2)[0]
    assert rescaled_parameter == pytest.approx(original, rel=1e-13)
    t = np.linspace(0, 2 * np.pi, 100)
    v = np.column_stack([-np.sin(t), np.cos(t)])
    a = np.column_stack([-np.cos(t), -np.sin(t)])
    assert _curvature_energy(v, a, -v, np.ones(100))[0] < 1e-22


def test_global_fairing_reduces_noise_with_certified_fidelity_and_radius_bounds():
    a = reference()
    b, report = regularize_airfoil(a, 5e-4, method="curvature", control_points=80)
    assert len(b.curves[0].c) == 80
    assert b.curves[0].k == 5
    s = np.unique(np.r_[np.linspace(0, a.length, 30001), a.s])
    delta = np.stack([f(s) - g(s) for f, g in zip(a.spline(), b.spline())])
    assert np.max(np.linalg.norm(delta, axis=0)) <= 5e-4
    assert report["deviation_bound"] <= 5e-4
    assert report["after"]["curvature_variation"] < .5 * report["before"]["curvature_variation"]
    assert report["after"]["aft_curvature_ranges"]["upper"][1] < 0
    assert abs(report["after"]["leading_edge_radius"] / report["before"]["leading_edge_radius"] - 1) <= .1000001
    assert report["curvature_energy_after"] < report["curvature_energy_reference"]
    assert abs(report["after"]["max_thickness"] - report["before"]["max_thickness"]) <= .001
    assert abs(report["after"]["max_abs_camber"] - report["before"]["max_abs_camber"]) <= .0005
    speed = np.hypot(*(f(s, 1) for f in b.spline()))
    assert np.min(speed) > .24
    for f in b.spline():
        knots = f.t[6:-6]
        assert len(np.unique(knots)) == len(knots)
        for d in (1, 2):
            df = f.derivative(d)
            left, right = df(np.nextafter(knots, -np.inf)), df(np.nextafter(knots, np.inf))
            assert np.max(abs(left - right) / (1 + np.maximum(abs(left), abs(right)))) < 1e-7
    for f, g in zip(a.spline(), b.spline()):
        for t in [0, a.length, a.s[a.leading_edge]]:
            assert f(t) == pytest.approx(g(t), abs=1e-10)
        for t in (0, a.length):
            assert f(t, 1) == pytest.approx(g(t, 1), abs=1e-8, rel=1e-8)
    from jaxrans.mesh.generate import wake

    def wake_direction(foil):
        return wake(foil, float(foil.x[0]), float(foil.y[0]), 8, .002, 1.)[-1]

    np.testing.assert_allclose(wake_direction(a), wake_direction(b), atol=1e-9)
    assert report["trailing_edge_tangents_preserved"]
    # Cached scientific results must not be shared mutable state.
    report["after"]["leading_edge_radius"] = -1
    b.curves[0].c[:] = 100
    fresh, fresh_report = regularize_airfoil(a, 5e-4, method="curvature", control_points=80)
    assert fresh_report["after"]["leading_edge_radius"] > 0
    assert fresh.curves[0](0) == pytest.approx(1., abs=1e-10)


@pytest.mark.parametrize("setting", ["airfoil_smoothing_points=6", "airfoil_smoothing_radius_tol=-0.1",
                                      "airfoil_smoothing_radius_tol=1"])
def test_invalid_input_bounds_are_rejected(setting):
    with pytest.raises(ValueError):
        load_case(ROOT / "examples/rae2822_sensitivity", (f"case.domain.{setting}",))


def test_zero_budget_keeps_original_curve_even_with_new_method():
    a = reference()
    assert regularize_airfoil(a, 0, method="curvature")[0] is a


def test_curvature_settings_reach_the_actual_mesh_contour():
    case = load_case(ROOT / "examples/rae2822_sensitivity", (
        "case.domain.airfoil_smoothing_method=curvature", "case.domain.airfoil_smoothing=0.0005",
        "case.domain.airfoil_smoothing_points=80", "case.domain.airfoil_smoothing_radius_tol=0.1"))
    fitted, notes = case.grid_spec.airfoil_shape(case.directory)
    assert len(fitted.curves[0].c) == 80
    assert any("80 control points" in note and "relative bound 0.1" in note for note in notes)
