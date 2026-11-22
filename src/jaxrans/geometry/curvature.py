"""Global C4 contour fitting with a geometric curvature objective."""

from copy import deepcopy
from functools import lru_cache

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import null_space
from scipy.optimize import linprog, minimize
from scipy.special import comb, factorial

from .airfoil import Airfoil
from .regularize import SplineAirfoil, _difference_bound, _shape_metrics


def _curvature_energy(v, acceleration, jerk, weights):
    """Integral (d curvature / d arc)^2 d arc and coordinate partials."""
    q = np.maximum(np.sum(v * v, axis=1), 1e-20)
    root = np.sqrt(q)
    cross = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    c, d = cross(v, acceleration), cross(v, jerk)
    e = np.sum(v * acceleration, axis=1)
    f = d / q**2 - 3 * c * e / q**3
    dq = 2 * v
    dc = np.column_stack([acceleration[:, 1], -acceleration[:, 0]])
    dd = np.column_stack([jerk[:, 1], -jerk[:, 0]])
    rotate = np.column_stack([-v[:, 1], v[:, 0]])
    fv = dd / q[:, None]**2 - 2 * (d / q**3)[:, None] * dq
    fv -= 3 * ((e[:, None] * dc + c[:, None] * acceleration) / q[:, None]**3
               - (3 * c * e / q**4)[:, None] * dq)
    fa = -3 * (e[:, None] * rotate + c[:, None] * v) / q[:, None]**3
    fj = rotate / q[:, None]**2
    multiplier = (2 * f * root)[:, None]
    partials = (multiplier * fv + (f * f / root)[:, None] * v,
                multiplier * fa, multiplier * fj)
    return float(np.sum(weights * f * f * root)), [weights[:, None] * p for p in partials]


def fit_curvature(original, budget, *, control_points=96, radius_tolerance=.1):
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("curvature fitting needs a positive finite displacement budget")
    if not isinstance(control_points, (int, np.integer)) or isinstance(control_points, bool) or control_points < 12:
        raise ValueError("airfoil_smoothing_points must be an integer >= 12")
    if not np.isfinite(radius_tolerance) or not 0 <= radius_tolerance < 1:
        raise ValueError("airfoil_smoothing_radius_tol must lie in [0, 1)")
    if type(original) is Airfoil:
        fitted, report = _cached_fit(original.name, np.asarray(original.x, dtype=np.float64).tobytes(),
                                     np.asarray(original.y, dtype=np.float64).tobytes(),
                                     float(budget), int(control_points), float(radius_tolerance))
        curves = tuple(BSpline(f.t.copy(), f.c.copy(), f.k) for f in fitted.curves)
        # Callers can inspect/edit their arrays without corrupting later fits.
        return SplineAirfoil(fitted.name, fitted.x.copy(), fitted.y.copy(), fitted.s.copy(), curves), deepcopy(report)
    return _fit_curvature(original, budget, control_points=control_points, radius_tolerance=radius_tolerance)


@lru_cache(maxsize=8)
def _cached_fit(name, x, y, budget, points, radius):
    original = Airfoil(name, np.frombuffer(x, dtype=np.float64), np.frombuffer(y, dtype=np.float64))
    return _fit_curvature(original, budget, control_points=points, radius_tolerance=radius)


def _fit_curvature(original, budget, *, control_points=96, radius_tolerance=.1):
    """Minimize geometric curvature variation with certified contour bounds.

    Coordinates and parameter are normalized internally for scale invariance.
    A second solve minimizes distance while allowing at most 1% more curvature
    energy. No knot insertion or budget increase hides an infeasible request.
    """
    if not isinstance(control_points, (int, np.integer)) or isinstance(control_points, bool) or control_points < 12:
        raise ValueError("airfoil_smoothing_points must be an integer >= 12")
    if not np.isfinite(radius_tolerance) or not 0 <= radius_tolerance < 1:
        raise ValueError("airfoil_smoothing_radius_tol must lie in [0, 1)")
    before = _shape_metrics(original)
    chord = original.chord
    origin = np.array([original.x[original.leading_edge], original.y[original.leading_edge]])
    a = original.unit_chord()
    s, le = a.s, a.s[a.leading_edge]
    ref = a.spline()
    n = control_points
    internal = np.interp(np.linspace(0, len(s) - 1, n - 4)[1:-1], np.arange(len(s)), s)
    knots = np.r_[[s[0]] * 6, internal, [s[-1]] * 6]
    basis = BSpline(knots, np.eye(n), 5)
    # The generator orients the wake with the bisector of the two TE
    # tangents. Fix both vectors, not just endpoint positions, so fairing
    # cannot silently rotate the entire wake grid.
    conditions = [(s[0], 0), (s[-1], 0), (le, 0), (le, 1),
                  (s[0], 1), (s[-1], 1)]
    c = np.stack([basis(t, nu=d) for t, d in conditions])
    target = np.array([[f(t, d) for f in ref] for t, d in conditions])
    scale = np.linalg.norm(c, axis=1)
    c, target = c / scale[:, None], target / scale[:, None]
    particular = np.linalg.lstsq(c, target, rcond=None)[0]
    free = null_space(c)
    # Common polynomial partition is also used for the independent certificate.
    breaks = np.unique(np.concatenate([s, internal, *[f.t for f in ref if isinstance(f, BSpline)]]))
    h = np.diff(breaks)
    nodes, gw = np.polynomial.legendre.leggauss(12)
    sites = ((breaks[:-1] + breaks[1:])[:, None] / 2 + h[:, None] * nodes / 2).ravel()
    weights = (h[:, None] * gw / 2).ravel()
    matrices = [basis(sites, nu=d) for d in range(4)]
    values = np.column_stack([f(sites) for f in ref])
    w = np.sqrt(weights)
    qfit, rfit = np.linalg.qr((matrices[0] @ free) * w[:, None], mode="reduced")
    z = np.linalg.solve(rfit, qfit.T @ ((values - matrices[0] @ particular) * w[:, None]))
    base = particular + free @ z
    # Whiten the distance Hessian for the nonlinear optimizer.
    transform = np.linalg.solve(rfit.T, free.T).T * budget
    maps = [b @ transform for b in matrices]
    constants = [b @ base for b in matrices]
    dim = free.shape[1]

    def fields(x):
        z = x.reshape(dim, 2)
        return [v + b @ z for v, b in zip(constants, maps)]

    def energy(x):
        _, v, acceleration, jerk = fields(x)
        value, partials = _curvature_energy(v, acceleration, jerk, weights)
        gradient = sum(b.T @ p for b, p in zip(maps[1:], partials))
        return value, gradient.ravel()

    x0 = np.zeros(dim * 2)
    e0 = max(energy(x0)[0], 1e-20)

    def objective(x):
        value, jac = energy(x)
        return value / e0, jac / e0

    # Linear maps to Bernstein controls on every common interval: bounding
    # their Euclidean displacement certifies the entire polynomial contour.
    powers = [basis(breaks[:-1], nu=d) * h[:, None]**d / factorial(d) for d in range(6)]
    original_powers = [np.column_stack([f(breaks[:-1], d) for f in ref]) * h[:, None]**d / factorial(d)
                       for d in range(6)]
    bern = np.stack([sum(comb(i, j) / comb(5, j) * powers[j] for j in range(i + 1)) for i in range(6)], axis=1).reshape(-1, n)
    reference = np.stack([sum(comb(i, j) / comb(5, j) * original_powers[j] for j in range(i + 1)) for i in range(6)], axis=1).reshape(-1, 2)
    displacement_base = (bern @ base - reference) / budget
    displacement_map = bern @ transform / budget

    def bound(x):
        d = displacement_base + displacement_map @ x.reshape(dim, 2)
        jac = (-2 * d[:, None, :] * displacement_map[:, :, None]).reshape(len(d), -1)
        return (1 - 1e-6)**2 - np.sum(d * d, axis=1), jac

    # Projection on the original tangent keeps speed nonzero on quadrature
    # sites during optimization. The final contour is checked more densely.
    tangent = np.column_stack([f(sites, 1) for f in ref])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    speed_base = np.sum(constants[1] * tangent, axis=1) - .25
    speed_map = (maps[1][:, :, None] * tangent[:, None, :]).reshape(len(sites), -1)
    # With the nose tangent fixed, its signed curvature is linear in r''.
    v = np.array([f(le, 1) for f in ref])
    normal = np.array([-v[1], v[0]]) / np.linalg.norm(v)**3
    kref = float(sum(normal[i] * ref[i](le, 2) for i in range(2)))
    if abs(kref) < 1e-10:
        raise ValueError("curvature fitting needs a nonzero leading-edge curvature")
    radius_base = float((basis(le, nu=2) @ base) @ normal / kref)
    radius_map = ((basis(le, nu=2) @ transform)[:, None] * normal[None, :] / kref).ravel()
    lower, upper = 1 / (1 + radius_tolerance), 1 / (1 - radius_tolerance)
    constraints = [
        {"type": "ineq", "fun": lambda x: bound(x)[0], "jac": lambda x: bound(x)[1]},
        {"type": "ineq", "fun": lambda x: speed_base + speed_map @ x, "jac": lambda x: speed_map},
        {"type": "ineq", "fun": lambda x: np.array([radius_base + radius_map @ x - lower, upper - radius_base - radius_map @ x]),
         "jac": lambda x: np.stack([radius_map, -radius_map])},
    ]
    def distance(x):
        delta = fields(x)[0] - values
        value = np.sum(weights[:, None] * delta * delta) / (budget**2 * a.length)
        jac = 2 * maps[0].T @ (weights[:, None] * delta) / (budget**2 * a.length)
        return value, jac.ravel()

    if radius_tolerance == 0:
        constraints[-1] = {"type": "eq", "fun": lambda x: radius_base + radius_map @ x - 1,
                           "jac": lambda x: radius_map}
    feasible = minimize(distance, x0, jac=True, method="SLSQP", constraints=constraints,
                        options={"maxiter": 2000, "ftol": 1e-7})
    if not feasible.success or np.min(bound(feasible.x)[0]) < -1e-7:
        # A feasible warm start avoids fragile SQP subproblems when the
        # unconstrained compact fit lies far outside a strict geometry bound.
        # An inscribed octagon is conservative relative to the Euclidean ball.
        angles = np.arange(8) * np.pi / 4
        directions = np.column_stack([np.cos(angles), np.sin(angles)])
        linear = (displacement_map[:, None, :, None] * directions[None, :, None, :]).reshape(-1, dim * 2)
        rhs = (.999999 * np.cos(np.pi / 8) - displacement_base @ directions.T).ravel()
        equality = radius_map[None, :] if radius_tolerance == 0 else None
        equality_rhs = np.array([1 - radius_base]) if radius_tolerance == 0 else None
        linear = np.vstack([linear, -speed_map, radius_map, -radius_map])
        rhs = np.r_[rhs, speed_base, upper - radius_base, radius_base - lower]
        seed = linprog(np.zeros(dim * 2), A_ub=linear, b_ub=rhs,
                       A_eq=equality, b_eq=equality_rhs,
                       bounds=[(None, None)] * (dim * 2), method="highs-ds", options={"time_limit": 5.})
        if seed.success:
            feasible = minimize(distance, seed.x, jac=True, method="SLSQP", constraints=constraints,
                                options={"maxiter": 1000, "ftol": 1e-9})
    if not feasible.success or np.min(bound(feasible.x)[0]) < -1e-7:
        raise ValueError(f"no feasible curvature fit ({feasible.message}, bound residual {np.min(bound(feasible.x)[0]):.3g}, "
                         f"speed residual {np.min(speed_base + speed_map @ feasible.x):.3g}, "
                         f"radius ratio {radius_base + radius_map @ feasible.x:.3g}); review the displacement, "
                         "control-point count and radius bounds")
    result = minimize(objective, feasible.x, jac=True, method="SLSQP", constraints=constraints,
                      options={"maxiter": 2000, "ftol": 1e-7})
    if not result.success or np.min(bound(result.x)[0]) < -1e-7:
        raise ValueError(f"curvature fit failed ({result.message}); increase the explicit displacement "
                         "budget or control-point count, or review the radius bound")
    best_energy = energy(result.x)[0]

    quality = {"type": "ineq", "fun": lambda x: 1.01 * best_energy / e0 - objective(x)[0],
               "jac": lambda x: -objective(x)[1]}
    closer = minimize(distance, result.x, jac=True, method="SLSQP", constraints=[*constraints, quality],
                      options={"maxiter": 400, "ftol": 1e-8})
    refined = (closer.success and np.min(bound(closer.x)[0]) >= -1e-8
               and quality["fun"](closer.x) >= -1e-9
               and distance(closer.x)[0] <= distance(result.x)[0] * (1 + 1e-8))
    chosen = closer.x if refined else result.x
    coeff = base + transform @ chosen.reshape(dim, 2)
    curves = tuple(BSpline(knots * chord, coeff[:, i] * chord + origin[i], 5) for i in range(2))
    fitted = SplineAirfoil(original.name, curves[0](original.s), curves[1](original.s), original.s.copy(), curves)
    original_tangents = np.array([[f(t, 1) for f in original.spline()]
                                 for t in (original.s[0], original.s[-1])])
    fitted_tangents = np.array([[f(t, 1) for f in fitted.spline()]
                               for t in (fitted.s[0], fitted.s[-1])])
    if not np.allclose(original_tangents, fitted_tangents, rtol=1e-8, atol=1e-8):
        raise ValueError("curvature fit failed its independent trailing-edge tangent check")
    certificate = _difference_bound(original, fitted) / chord
    # Do not silently clip or scale a failed optimizer result into the budget.
    if certificate > budget * (1 + 1e-7):
        raise ValueError(f"curvature fit failed its independent displacement certificate: {certificate:g} > {budget:g}; constraint {np.min(bound(chosen)[0]):g}")
    check_sites = np.linspace(original.s[0], original.s[-1], 16385)
    speed = np.hypot(*(f(check_sites, 1) for f in fitted.curves))
    if np.min(speed) < .24:
        raise ValueError("curvature fit failed its dense tangent regularity check")
    after = _shape_metrics(fitted)
    if abs(after["leading_edge_radius"] / before["leading_edge_radius"] - 1) > radius_tolerance + 1e-7:
        raise ValueError("curvature fit failed its independent radius check")
    if (abs(after["max_thickness"] - before["max_thickness"]) > 2 * budget
            or abs(after["max_abs_camber"] - before["max_abs_camber"]) > budget):
        raise ValueError("curvature fit failed the sampled thickness/camber safeguards")
    reference_energy = _curvature_energy(
        np.column_stack([f(sites, 1) for f in ref]),
        np.column_stack([f(sites, 2) for f in ref]),
        np.column_stack([f(sites, 3) for f in ref]), weights)[0]
    final_energy = energy(chosen)[0]
    if final_energy > reference_energy * (1 + 1e-6):
        raise ValueError("fit does not reduce reference curvature energy within the requested bounds")
    return fitted, {"enabled": True, "method": "curvature", "degree": 5, "control_points": n,
                    "trailing_edge_tangents_preserved": True,
                    "max_deviation": budget, "deviation_bound": certificate,
                    "radius_tolerance": radius_tolerance, "rms_deviation": float(np.sqrt(distance(chosen)[0]) * budget),
                    "shape_bounds": {"max_thickness_change": 2 * budget,
                                     "max_abs_camber_change": budget},
                    "curvature_energy_before_fit": e0,
                    "curvature_energy_reference": reference_energy,
                    "curvature_energy_after": final_energy,
                    "optimizer_iterations": result.nit, "distance_refinement_success": bool(refined),
                    "distance_refinement_status": str(closer.message),
                    "before": before, "after": after}
