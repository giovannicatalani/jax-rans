"""Bounded fairing of coordinate contours before meshing.

The legacy quadratic penalty is a resolution-weighted third parametric
derivative. The B-spline variant uses its arc-integrated square. Neither is
a curvature monotonicity constraint. The global curvature method directly
minimizes arc-length curvature variation under shape bounds. Displacement is bounded on the entire
piecewise polynomial contour using the convex hull of Bernstein controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.interpolate import BSpline, CubicSpline, PPoly
from scipy.linalg import null_space, svd

from .airfoil import Airfoil


def _difference_bound(a: Airfoil, b: Airfoil) -> float:
    # Match each knot interval by its local fraction, allowing the fitted
    # contour to be reparameterized by its new polygonal arc length.
    def polynomial(curve):
        return PPoly.from_spline(curve) if isinstance(curve, BSpline) else curve

    pa, pb = [polynomial(f) for f in a.spline()], [polynomial(f) for f in b.spline()]
    index = np.arange(len(a.s), dtype=float)
    if len(a.s) != len(b.s):
        raise ValueError("contour correspondence needs matching parameter landmarks")
    # Common partition in landmark-index coordinates also handles refined
    # B-spline knots and the legacy contour's different arc parameter.
    splits = np.unique(np.concatenate([index, *[
        np.interp(p.x, obj.s, index) for obj, curves in ((a, pa), (b, pb)) for p in curves]]))
    degree = max(p.c.shape[0] - 1 for p in (*pa, *pb))

    def coefficients(curves, obj):
        left = np.interp(splits[:-1], index, obj.s)
        width = np.interp(splits[1:], index, obj.s) - left
        result = []
        for p in curves:
            idx = np.clip(np.searchsorted(p.x, left + width / 2, side="right") - 1, 0, len(p.x) - 2)
            offset = left - p.x[idx]
            raw = p.c[::-1, idx]
            result.append(np.stack([
                sum(comb(j, k) * raw[j] * offset**(j - k) for j in range(k, len(raw))) * width**k
                if k < len(raw) else np.zeros_like(width) for k in range(degree + 1)]))
        return np.stack(result, axis=-1)

    delta = coefficients(pa, a) - coefficients(pb, b)
    controls = np.stack([sum(comb(i, j) / comb(degree, j) * delta[j]
                             for j in range(i + 1)) for i in range(degree + 1)])
    return float(np.max(np.linalg.norm(controls, axis=-1)))


def _sample(a: Airfoil, count: int = 4097):
    s = np.linspace(0, a.length, count)
    fx, fy = a.spline()
    xy = np.column_stack([fx(s), fy(s)])
    dx, dy = fx(s, 1), fy(s, 1)
    curvature = (dx * fy(s, 2) - dy * fx(s, 2)) / (dx * dx + dy * dy) ** 1.5
    return s, xy, curvature


def _shape_metrics(a: Airfoil) -> dict:
    s, xy, k = _sample(a)
    le = a.s[a.leading_edge]
    grid = np.linspace(0, 1, 2001)
    lower = s <= le
    upper = s >= le
    if np.any(np.diff(xy[lower, 0]) > 1e-7 * a.chord) or np.any(np.diff(xy[upper, 0]) < -1e-7 * a.chord):
        raise ValueError("regularization requires a contour monotone in x on each surface")
    xgrid = a.x[a.leading_edge] + grid * a.chord
    yl = np.interp(xgrid, xy[lower, 0][::-1], xy[lower, 1][::-1])
    yu = np.interp(xgrid, xy[upper, 0], xy[upper, 1])
    if np.min(yu[1:-1] - yl[1:-1]) <= 0:
        raise ValueError("regularization produced crossed surfaces")
    fx, fy = a.spline()
    speed2 = fx(le, 1)**2 + fy(le, 1)**2
    kle = (fx(le, 1) * fy(le, 2) - fy(le, 1) * fx(le, 2)) / speed2**1.5
    aft = (xy[:, 0] > a.x[a.leading_edge] + .6 * a.chord) & (xy[:, 0] < a.x[a.leading_edge] + .85 * a.chord)
    ranges = {name: [float(np.min(k[aft & mask]) * a.chord),
                     float(np.max(k[aft & mask]) * a.chord)]
              for name, mask in (("lower", lower), ("upper", upper)) if np.any(aft & mask)}
    return {"max_thickness": float(np.max(yu - yl) / a.chord),
            "max_abs_camber": float(np.max(np.abs((yu + yl) / 2 - a.y[a.leading_edge])) / a.chord),
            "leading_edge_radius": float(1 / abs(kle) / a.chord),
            "curvature_variation": float(np.sum(np.abs(np.diff(k)))),
            "aft_curvature_ranges": ranges}


def regularize_airfoil(a: Airfoil, max_deviation: float, *, method="parametric",
                       control_points=96, radius_tolerance=.1) -> tuple[Airfoil, dict]:
    """Fair a contour within ``max_deviation`` chords; zero is an exact no-op.

    Legacy methods fix edge derivatives. Global curvature fitting instead
    bounds nose radius and fixes the LE tangent and edge positions. After
    reparameterization, actual radius and shape changes are reported rather
    than assumed zero. Only contours monotone in x per surface are supported.
    """
    if not np.isfinite(max_deviation) or max_deviation < 0:
        raise ValueError("airfoil_smoothing must be finite and nonnegative")
    if method not in ("parametric", "bspline", "curvature"):
        raise ValueError("unknown airfoil smoothing method")
    if max_deviation == 0:
        return a, {"enabled": False, "deviation_bound": 0.0}
    if method == "curvature":
        from .curvature import fit_curvature

        return fit_curvature(a, max_deviation, control_points=control_points,
                             radius_tolerance=radius_tolerance)
    if method == "bspline":
        return _regularize_bspline(a, max_deviation)
    before = _shape_metrics(a)
    s, n, le = a.s, len(a.x), a.leading_edge
    basis = CubicSpline(s, np.eye(n))
    constraints = np.stack([basis(t, d) for t, d in (
        (s[0], 0), (s[-1], 0), (s[le], 0), (s[0], 1),
        (s[-1], 1), (s[le], 1), (s[le], 2))])
    constraints /= np.linalg.norm(constraints, axis=1)[:, None]
    free = null_space(constraints)
    if free.shape[1] == 0:
        raise ValueError("not enough free contour points for regularization")
    # h^4 weights the integrated squared third derivative: avoid allowing
    # tiny clustered nose intervals to dominate fairing of the whole contour.
    h = np.diff(s)
    penalty = basis((s[1:] + s[:-1]) / 2, 3) * h[:, None]**2.5
    u, sigma, vt = svd(penalty @ free, full_matrices=False)
    scale = sigma[0]
    sigma /= scale
    xy = np.column_stack([a.x, a.y])
    rhs = u.T @ (penalty @ xy / scale)

    def candidate(strength):
        factors = strength * sigma / (1 + strength * sigma**2)
        moved = xy - free @ (vt.T @ (factors[:, None] * rhs))
        moved[[0, le, -1]] = xy[[0, le, -1]]
        return Airfoil(a.name, moved[:, 0], moved[:, 1])

    budget = max_deviation * a.chord
    low, high = 0., 1.
    while high < 1e16 and _difference_bound(a, candidate(high)) <= budget:
        low, high = high, high * 10
    # Keep the accepted side of the bracket; all returned contours satisfy
    # the continuous displacement bound, even if the search is conservative.
    for _ in range(48):
        mid = (low + high) / 2
        if _difference_bound(a, candidate(mid)) <= budget:
            low = mid
        else:
            high = mid
    fitted = candidate(low)
    after = _shape_metrics(fitted)
    return fitted, {"enabled": True, "method": "constrained_parametric_fairing",
                    "max_deviation": float(max_deviation),
                    "deviation_bound": _difference_bound(a, fitted) / a.chord,
                    "strength": low, "before": before, "after": after}


@dataclass(frozen=True)
class SplineAirfoil(Airfoil):
    """Keep the fitted B-spline itself all the way to the mesh generator.

    Parameter values retain the original polygonal arc-length convention;
    no cubic reinterpolation or post-fit reparameterization is performed.
    """

    parameter: np.ndarray
    curves: tuple[BSpline, BSpline]

    @property
    def s(self):
        return self.parameter

    def spline(self):
        return self.curves

    def unit_chord(self):
        le, c = self.leading_edge, self.chord
        curves = tuple(BSpline(f.t / c, (f.c - v) / c, f.k)
                       for f, v in zip(self.curves, (self.x[le], self.y[le])))
        return SplineAirfoil(self.name, (self.x - self.x[le]) / c,
                             (self.y - self.y[le]) / c, self.s / c, curves)

    def closed(self):
        if self.sharp:
            return self
        raise ValueError("close the trailing edge before B-spline regularization")


def _regularize_bspline(a, max_deviation, refinement=0):
    """Constrained quintic B-spline fit with simple interior knots (C4).

    Minimize arc-weighted squared distance plus an integrated third-derivative
    penalty. Select the largest smoothing strength with a certified maximum
    displacement. This is a fidelity/smoothness tradeoff, not a global
    minimum-distance solution under an independently prescribed curvature.
    """
    before = _shape_metrics(a)
    s = a.s
    degree = 5
    breaks = s.copy()
    for _ in range(refinement):
        breaks = np.sort(np.r_[breaks, (breaks[:-1] + breaks[1:]) / 2])
    knots = np.r_[[s[0]] * (degree + 1), breaks[1:-1], [s[-1]] * (degree + 1)]
    count = len(knots) - degree - 1
    basis = BSpline(knots, np.eye(count), degree)
    ref = a.spline()
    le = s[a.leading_edge]
    conditions = [(s[0], 0), (s[-1], 0), (le, 0), (s[0], 1),
                  (s[-1], 1), (le, 1), (le, 2)]
    constraints = np.stack([basis(t, nu=d) for t, d in conditions])
    targets = np.array([[f(t, d) for f in ref] for t, d in conditions])
    norms = np.linalg.norm(constraints, axis=1)
    constraints /= norms[:, None]
    targets /= norms[:, None]
    particular = np.linalg.lstsq(constraints, targets, rcond=None)[0]
    free = null_space(constraints)
    # Eight-point Gauss integration resolves the polynomial distance objective
    # exactly on each original interval (degree <= 10 after squaring).
    nodes, weights = np.polynomial.legendre.leggauss(8)
    h = np.diff(breaks)
    sites = ((breaks[:-1] + breaks[1:])[:, None] / 2 + h[:, None] * nodes / 2).ravel()
    w = np.sqrt((h[:, None] * weights / 2).ravel())
    design = basis(sites) * w[:, None]
    values = np.column_stack([f(sites) for f in ref]) * w[:, None]
    q, r = np.linalg.qr(design @ free, mode="reduced")
    base = particular + free @ np.linalg.solve(r, q.T @ (values - design @ particular))
    whitened = np.linalg.solve(r.T, free.T).T
    penalty = basis(sites, nu=3) * w[:, None]
    u, sigma, vt = svd(penalty @ whitened, full_matrices=False)
    scale = sigma[0]
    sigma /= scale
    rhs = u.T @ (penalty @ base / scale)

    def candidate(strength):
        factors = strength * sigma / (1 + strength * sigma**2)
        coeff = base - whitened @ (vt.T @ (factors[:, None] * rhs))
        curves = tuple(BSpline(knots, coeff[:, axis], degree) for axis in range(2))
        return SplineAirfoil(a.name, curves[0](s), curves[1](s), s.copy(), curves)

    budget = max_deviation * a.chord
    fitted = candidate(0.)
    minimum_bound = _difference_bound(a, fitted)
    if minimum_bound > budget:
        if refinement < 3:
            return _regularize_bspline(a, max_deviation, refinement + 1)
        raise ValueError(f"B-spline approximation needs a larger budget or more knots: "
                         f"bound {minimum_bound / a.chord:.3g} c > {max_deviation:g} c")
    low, high = 0., 1.
    while high < 1e16 and _difference_bound(a, candidate(high)) <= budget:
        low, high = high, high * 10
    for _ in range(48):
        mid = (low + high) / 2
        if _difference_bound(a, candidate(mid)) <= budget:
            low = mid
        else:
            high = mid
    fitted = candidate(low)
    speed = np.hypot(*(f(np.linspace(s[0], s[-1], 16385), 1) for f in fitted.curves))
    if np.min(speed) < 1e-6:
        raise ValueError("B-spline has a near-singular tangent")
    final_values = np.column_stack([f(sites) for f in fitted.curves]) * w[:, None]
    rms = float(np.sqrt(np.sum((final_values - values)**2) / (s[-1] - s[0])) / a.chord)
    return fitted, {"enabled": True, "method": "bspline", "degree": degree,
                    "control_points": count, "knot_refinements": refinement,
                    "interior_continuity": "C4 (simple knots); trailing-edge corner retained",
                    "max_deviation": max_deviation,
                    "deviation_bound": _difference_bound(a, fitted) / a.chord,
                    "minimum_fit_bound": minimum_bound / a.chord,
                    "rms_deviation": rms,
                    "strength": low, "before": before, "after": _shape_metrics(fitted)}
