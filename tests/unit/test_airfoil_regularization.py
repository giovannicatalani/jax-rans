"""Geometry bounds and invariants, independently checked on dense contours."""

from pathlib import Path

import numpy as np
import pytest

from jaxrans import load_case
from jaxrans.geometry.airfoil import naca4, read_airfoil
from jaxrans.geometry.regularize import regularize_airfoil

ROOT = Path(__file__).resolve().parents[2]


def reference():
    return read_airfoil(ROOT / "validation/references/rae2822/rae2822.dat").unit_chord()


def test_disabled_is_exact_noop():
    a = naca4("0012")
    assert regularize_airfoil(a, 0)[0] is a


@pytest.mark.parametrize("budget", [1e-5, 5e-5, 1e-4])
def test_rae_continuous_contour_bound_and_edge_invariants(budget):
    a = reference()
    b, report = regularize_airfoil(a, budget)
    indices = [0, a.leading_edge, len(a.x) - 1]
    np.testing.assert_array_equal(b.x[indices], a.x[indices])
    np.testing.assert_array_equal(b.y[indices], a.y[indices])
    fraction = np.linspace(0, 1, 51)
    sa = a.s[:-1, None] + np.diff(a.s)[:, None] * fraction
    sb = b.s[:-1, None] + np.diff(b.s)[:, None] * fraction
    delta = np.stack([fa(sa) - fb(sb) for fa, fb in zip(a.spline(), b.spline())])
    assert np.max(np.linalg.norm(delta, axis=0)) <= budget * (1 + 1e-8)
    assert report["after"]["curvature_variation"] < report["before"]["curvature_variation"]
    assert abs(report["after"]["leading_edge_radius"] / report["before"]["leading_edge_radius"] - 1) < .001
    assert abs(report["after"]["max_thickness"] - report["before"]["max_thickness"]) < 1e-6


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        regularize_airfoil(reference(), budget)


def test_case_override_reaches_the_meshed_geometry():
    case = load_case(ROOT / "examples/rae2822_sensitivity", ("case.domain.airfoil_smoothing=0.0001",))
    a, notes = case.grid_spec.airfoil_shape(case.directory)
    expected, _ = regularize_airfoil(reference(), 1e-4)
    np.testing.assert_array_equal(a.y, expected.y)
    assert any("regularization" in note for note in notes)


@pytest.mark.parametrize("method", ["parametric", "bspline"])
def test_regularization_respects_translation_and_scale(method):
    from jaxrans.geometry.airfoil import Airfoil

    a = reference()
    b, _ = regularize_airfoil(a, 5e-5, method=method)
    shifted = Airfoil(a.name, 3 * a.x + 2, 3 * a.y - 4)
    fitted, report = regularize_airfoil(shifted, 5e-5, method=method)
    np.testing.assert_allclose(fitted.x, 3 * b.x + 2, atol=1e-11)
    np.testing.assert_allclose(fitted.y, 3 * b.y - 4, atol=1e-11)
    assert report["deviation_bound"] <= 5e-5 * (1 + 1e-8)
    if method == "bspline":
        normalized = fitted.unit_chord()
        assert normalized.spline()[0].k == 5
        np.testing.assert_allclose(normalized.y, b.y, atol=1e-11)


@pytest.mark.parametrize("budget", [1e-5, 1e-4])
def test_bspline_bound_and_c4_continuity_in_actual_mesh_curve(budget):
    from scipy.interpolate import BSpline

    a = reference()
    b, report = regularize_airfoil(a, budget, method="bspline")
    assert all(isinstance(f, BSpline) and f.k == 5 for f in b.spline())
    s = np.unique(np.r_[np.linspace(0, a.length, 30001), a.s])
    delta = np.stack([fa(s) - fb(s) for fa, fb in zip(a.spline(), b.spline())])
    assert np.max(np.linalg.norm(delta, axis=0)) <= budget * (1 + 1e-8)
    for fa, fb in zip(a.spline(), b.spline()):
        for t, d in [(a.s[0], 0), (a.s[-1], 0), (a.s[a.leading_edge], 0),
                     (a.s[0], 1), (a.s[-1], 1), (a.s[a.leading_edge], 1), (a.s[a.leading_edge], 2)]:
            np.testing.assert_allclose(fb(t, d), fa(t, d), rtol=1e-8, atol=1e-8)
        interior = fb.t[6:-6]
        assert len(np.unique(interior)) == len(interior)
        for derivative in range(5):
            df = fb.derivative(derivative)
            left = df(np.nextafter(interior, -np.inf))
            right = df(np.nextafter(interior, np.inf))
            assert np.max(abs(left - right) / (1 + np.maximum(abs(left), abs(right)))) < 1e-7
    assert report["after"]["curvature_variation"] < report["before"]["curvature_variation"]
    assert report["after"]["leading_edge_radius"] == pytest.approx(report["before"]["leading_edge_radius"], rel=1e-8)


def test_bspline_case_option_preserves_fitted_representation():
    case = load_case(ROOT / "examples/rae2822_sensitivity", (
        "case.domain.airfoil_smoothing=0.0001", "case.domain.airfoil_smoothing_method=bspline"))
    a, _ = case.grid_spec.airfoil_shape(case.directory)
    assert a.spline()[0].k == 5
