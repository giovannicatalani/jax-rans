"""Operational boundary-layer diagnostics from saved sectional fields.

Samples true 3D wall-normal rays, using the prescribed conical copy closure
to interpolate the independent section. The edge reference is an ideal-gas
isentropic pressure reconstruction from freestream stagnation conditions.
This is an explicit engineering diagnostic, not a general 3D displacement
thickness definition or the exact algorithm of Griffin et al. (2021).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def first_gap_for_yplus(reynolds, cf_estimate, target=.5):
    """Preliminary full first-cell height/c, rho_w=rho_inf, mu_w=mu_inf.

    The FV centre is assumed halfway through the gap. This sizing estimate
    requires a positive skin-friction estimate and an a-posteriori y+ check.
    """
    if not all(np.isfinite(x) and x > 0 for x in (reynolds, cf_estimate, target)):
        raise ValueError("Reynolds, friction estimate and target y+ must be positive and finite")
    return 2 * target / (reynolds * np.sqrt(cf_estimate / 2))


def layers_through_thickness(thickness, first_gap, growth=1.15):
    if not (np.isfinite(thickness) and np.isfinite(first_gap) and np.isfinite(growth)
            and thickness > 0 and first_gap > 0 and growth >= 1):
        raise ValueError("positive finite thickness/gap and growth >= 1 required")
    if growth == 1:
        return int(np.ceil(thickness / first_gap))
    return int(np.ceil(np.log1p(thickness * (growth - 1) / first_gap) / np.log(growth)))


def layer_design(reynolds, cf_estimate, delta_min, delta_max, target=.5, minimum_cells=30, growth=1.1):
    """Couple wall-unit and minimum-through-layer constraints, before meshing."""
    wall_gap = first_gap_for_yplus(reynolds, cf_estimate, target)
    if (not isinstance(minimum_cells, int) or isinstance(minimum_cells, bool) or minimum_cells < 1
            or not np.isfinite(delta_min) or not np.isfinite(delta_max)
            or not np.isfinite(growth) or not 0 < delta_min <= delta_max or growth < 1):
        raise ValueError("positive ordered thickness range, cell count and growth >= 1 required")
    count_gap = delta_min / minimum_cells if growth == 1 else delta_min * (growth - 1) / (growth**minimum_cells - 1)
    gap = min(wall_gap, count_gap)
    return {"first_gap": gap, "growth": growth,
            "layers_to_cover_delta_max": layers_through_thickness(delta_max, gap, growth),
            "wall_unit_gap_limit": wall_gap, "cell_count_gap_limit": count_gap,
            "minimum_cells_in_delta_min": minimum_cells}


def foot_points(points, geometry):
    """Inverse of the FULL-conicity generator map, away from its apex."""
    p, mu = geometry.coefficients
    factor = 1 - mu * points[2]
    if np.any(factor <= 0):
        raise ValueError("normal ray crosses the conical apex")
    return np.stack([(points[0] - (p + mu * geometry.normal_leading_edge_x) * points[2]) / factor,
                     points[1] / factor])


def reference_speed(pressure, normal_velocity, mach=.5, gamma=1.4):
    """Tangential speed from freestream entropy and stagnation enthalpy.

    Units: rho_inf=1, p_inf=1/gamma, a_inf=1. Measured normal velocity
    is subtracted from reconstructed total speed. Invalid states stay NaN.
    """
    t0 = 1 + .5 * (gamma - 1) * mach**2
    tis = np.where(pressure > 0, (gamma * pressure)**((gamma - 1) / gamma), np.nan)
    square = 2 / (gamma - 1) * (t0 - tis) - normal_velocity**2
    return np.sqrt(np.where(square > 0, square, np.nan))


def edge_metrics(distance, primitive, normal, mach=.5, gamma=1.4):
    """Speed-based delta99 and locally planar scalar integral estimates.

    Acceptance checks ensure that the reference is recovered at twice the
    detected thickness. Crossflow is reported separately and never hidden.
    """
    velocity = primitive[1:4]
    vn = normal @ velocity
    tangent = velocity - normal[:, None] * vn
    speed = np.linalg.norm(tangent, axis=0)
    ui = reference_speed(primitive[4], vn, mach, gamma)
    ratio = speed / ui
    candidates = np.flatnonzero((ratio >= .99) & np.isfinite(ratio))
    candidates = candidates[candidates > 0]
    if not len(candidates):
        return {"status": "edge_not_found"}, ratio, ui
    k = int(candidates[0])
    previous = k - 1
    fraction = (.99 - ratio[previous]) / (ratio[k] - ratio[previous])
    delta = float(distance[previous] + fraction * (distance[k] - distance[previous]))
    if not np.isfinite(delta) or delta <= 0 or 2 * delta > distance[-1]:
        return {"status": "insufficient_outer_extent"}, ratio, ui
    check = float(np.interp(2 * delta, distance, ratio))
    if not .985 <= check <= 1.015:
        return {"status": "reference_not_recovered", "outer_speed_ratio": check}, ratio, ui
    edge_velocity = np.array([np.interp(2 * delta, distance, v) for v in tangent])
    direction = edge_velocity / np.linalg.norm(edge_velocity)
    cross = np.cross(normal, direction)
    u, w = direction @ velocity, cross @ velocity
    selected = distance < delta
    d = np.r_[distance[selected], delta]
    rho = np.interp(d, distance, primitive[0])
    pressure = np.interp(d, distance, primitive[4])
    u = np.interp(d, distance, u)
    uis = np.interp(d, distance, ui)
    rhois = (gamma * pressure)**(1 / gamma)
    mass = rho * u / (rhois * uis)
    displacement = float(np.trapezoid(1 - mass, d))
    momentum = float(np.trapezoid(mass * (1 - u / uis), d))
    result = {"status": "estimated", "delta99_speed": delta,
              "delta_star_scalar": displacement, "theta_scalar": momentum,
              "shape_factor_scalar": displacement / momentum if momentum > 0 else None,
              "outer_speed_ratio": check,
              "crossflow_at_edge_over_ui": float(np.interp(delta, distance, w / ui)),
              "max_crossflow_over_ui_in_layer": float(np.max(abs((w / ui)[selected])))}
    defect = np.linalg.norm(tangent - direction[:, None] * ui, axis=0) / ui
    hits = np.flatnonzero((defect <= .01) & (distance > 0))
    result["delta_vector_1pct"] = None
    if len(hits) and hits[0] > 0:
        j = int(hits[0])
        f = (.01 - defect[j - 1]) / (defect[j] - defect[j - 1])
        result["delta_vector_1pct"] = float(distance[j - 1] + f * (distance[j] - distance[j - 1]))
    return result, ratio, ui


class SectionSampler:
    """Piecewise-linear interpolation on an explicitly connected dual mesh.

    Triangles follow the main C-block without joining its downstream ends.
    Only actual wall faces receive no-slip/SA wall values; base/wake
    interfaces receive the adjacent block values. Rays outside this block
    are refused; profiles on the blunt base are not extracted.
    """

    def __init__(self, arrays, geometry):
        from matplotlib.tri import Triangulation
        self.geometry = geometry
        nodes = arrays.get("nodes_b0", arrays["nodes"])
        q = arrays.get("primitive_b0", arrays["primitive"])
        central = nodes.mean(axis=-1)
        centres = .25 * (central[:, :-1, :-1] + central[:, 1:, :-1]
                         + central[:, :-1, 1:] + central[:, 1:, 1:])
        self.wall = .5 * (central[:, :-1, 0] + central[:, 1:, 0])
        along = central[:, 1:, 0] - central[:, :-1, 0]
        generators = .5 * (nodes[:, :-1, 0, 1] + nodes[:, 1:, 0, 1]
                            - nodes[:, :-1, 0, 0] - nodes[:, 1:, 0, 0])
        normal = np.cross(along.T, generators.T).T
        normal /= np.linalg.norm(normal, axis=0)
        normal *= np.sign(np.sum((centres[:, :, 0] - self.wall) * normal, axis=0))
        self.normal, self.centres = normal, centres
        points = np.concatenate([self.wall[:, :, None], centres], axis=2)
        wall_q = q[:, :, :1].copy()
        if "topology" in arrays:
            from jaxrans.mesh.atlas import interior_cell
            topology = json.loads(str(arrays["topology"]))
            for segment in topology[0]["j_min"]:
                cells = slice(segment["start"], segment["stop"])
                if "block" in segment:
                    donor = arrays[f"primitive_b{segment['block']}"]
                    partner = segment["partner_start"] + np.arange(segment["stop"] - segment["start"])
                    if segment["reversed"]:
                        partner = partner[::-1]
                    i, j = interior_cell(segment["side"], partner, 0, *donor.shape[1:])
                    wall_q[:, cells, 0] = .5 * (q[:, cells, 0] + donor[:, i, j])
                elif segment["kind"] == "no_slip_wall":
                    wall_q[1:4, cells] = 0
                    if len(q) > 5:
                        wall_q[5:, cells] = 0
        else:
            # Legacy single-block archives retained for historical audits.
            wall_q[1:4] = 0
            if len(q) > 5:
                wall_q[5:] = 0
        values = np.concatenate([wall_q, q], axis=2)
        ni, nj = points.shape[1:]
        i, j = np.meshgrid(np.arange(ni - 1), np.arange(nj - 1), indexing="ij")
        a, b = i * nj + j, (i + 1) * nj + j
        triangles = np.concatenate([np.stack([a, b, a + 1], -1).reshape(-1, 3),
                                     np.stack([b, b + 1, a + 1], -1).reshape(-1, 3)])
        self.points, self.values = points[:2].reshape(2, -1), values.reshape(len(q), -1)
        triangulation = Triangulation(*self.points, triangles)
        self.triangles, self.find = triangulation.triangles, triangulation.get_trifinder()

    def sample(self, points):
        xy = foot_points(points, self.geometry)
        index = self.find(*xy)
        if np.any(index < 0):
            raise ValueError("normal ray leaves the saved field's interpolation domain")
        vertices = self.triangles[index]
        a, b, c = (self.points[:, vertices[:, k]] for k in range(3))
        determinant = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        t = ((xy[0] - a[0]) * (c[1] - a[1]) - (xy[1] - a[1]) * (c[0] - a[0])) / determinant
        u = ((b[0] - a[0]) * (xy[1] - a[1]) - (b[1] - a[1]) * (xy[0] - a[0])) / determinant
        return (self.values[:, vertices[:, 0]] * (1 - t - u)
                + self.values[:, vertices[:, 1]] * t + self.values[:, vertices[:, 2]] * u)

    def contains(self, points):
        return self.find(*foot_points(points, self.geometry)) >= 0


def diagnose(folder, output, stations=(.02, .1, .5, .9, .98), height=.12):
    from jaxrans.case import load_case
    from jaxrans.geometry.conical import ConicalGeometry
    from jaxrans.physics.navier_stokes import Viscosity
    from jaxrans.physics.spalart_allmaras import SAModel
    from validation.sectional_qualification import write_json

    folder, output = Path(folder), Path(output)
    case = load_case(folder)
    if case.flow.inviscid or case.grid_spec.sectional is None:
        raise ValueError("boundary-layer diagnostics require a viscous sectional case")
    # Qualification's source is a unit-chord, unshifted NACA 0012.
    if case.grid_spec.airfoil not in ("naca0012", "naca0012_blunt") or case.grid_spec.section != "streamwise":
        raise ValueError("this diagnostic currently requires the unit-chord streamwise NACA 0012 case")
    if height >= case.numerics.sectional_relaxation_start:
        raise ValueError("rays must stay inside the full-conicity zone")
    spec = case.grid_spec.sectional
    geometry = ConicalGeometry(spec, 1., 0.)
    with np.load(folder / "arrays.npz") as archive:
        arrays = dict(archive)
    sampler = SectionSampler(arrays, geometry)
    y_streamwise = arrays.get("y_streamwise", arrays["y"])
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for side in (-1, 1):
        eligible = side * y_streamwise > 0
        if "wall_block" in arrays:
            from jaxrans.mesh.block import SIDES
            eligible &= (arrays["wall_block"] == 0) & (arrays["wall_side"] == SIDES.index("j_min"))
        available = np.flatnonzero(eligible)
        for x in stations:
            i = int(available[np.argmin(abs(arrays["x_streamwise"][available] - x))])
            column = int(arrays["wall_cell"][i]) if "wall_cell" in arrays else i
            origin, normal = sampler.wall[:, column], sampler.normal[:, column]
            first = float((sampler.centres[:, column, 0] - origin) @ normal)
            # Dense sampling locates crossings; it DOES NOT add mesh resolution.
            d = np.r_[0., np.geomspace(max(first / 4, 1e-9), height, 800)]
            ray = origin[:, None] + normal[:, None] * d
            outside = np.flatnonzero(~sampler.contains(ray))
            if len(outside):
                stop = int(outside[0])
                if stop < 3:
                    summaries.append({"side": "upper" if side > 0 else "lower", "requested_x": x,
                                      "status": "outside_interpolation_domain"})
                    continue
                d, ray = d[:stop], ray[:, :stop]
            q = sampler.sample(ray)
            q[1:4, 0] = 0
            metrics, ratio, ui = edge_metrics(d, q, normal, case.flow.mach, case.flow.gamma)
            if metrics["status"] == "estimated":
                delta = metrics["delta99_speed"]
                native_d = np.sum((sampler.centres[:, column] - origin[:, None]) * normal[:, None], axis=0)
                metrics["native_centres_below_delta99"] = int(np.count_nonzero((native_d > 0) & (native_d < delta)))
            # Surface friction is transported to the streamwise cut by the
            # postprocessor. Undo its gradient scale for this normal-cut ray.
            scale = geometry.streamwise_gradient_scale(sampler.wall[:2, column])
            stress = .5 * case.flow.mach**2 * np.hypot(arrays["cf"][i], arrays["cf_z"][i]) / scale
            mu = np.broadcast_to(np.asarray(Viscosity.of(case.flow)(case.flow.gamma * q[4] / q[0])), q[0].shape)
            metrics["yplus_first_centre_plane"] = float(first * np.sqrt(q[0, 0] * stress) / mu[0])
            model = SAModel.of(case.flow)
            eddy = np.zeros_like(q[0]) if model is None else np.asarray(model.eddy_viscosity(q[0], q[5], mu)) / mu
            label = f"{'upper' if side > 0 else 'lower'}_{x:g}"
            np.savez_compressed(output / f"{label}.npz", distance=d, primitive=q, speed_ratio=ratio,
                                reference_speed=ui, normal=normal, eddy_viscosity_ratio=eddy)
            summaries.append({"side": "upper" if side > 0 else "lower", "requested_x": x,
                              "actual_x": float(arrays["x_streamwise"][i]), "wall_face": i,
                              "available_normal_extent": float(d[-1]),
                              "interpolation_domain_truncated": bool(len(outside)),
                              "first_centre_normal_distance": first, **metrics})
    result = {"source": str(folder.resolve()), "method": "isentropic_pressure_reference_tangential_speed",
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "baseline": str(arrays.get("baseline", "historical_o_grid")),
              "height_limit": height, "profiles": summaries,
              "limitations": ["Scalar operational estimates; not a general 3D thickness tensor.",
                               "Nearest wall face; actual x is reported.",
                               "Native-cell count uses the corresponding grid column; dense interpolation adds no cells.",
                               "Vector indicator uses the outer direction at twice the speed-based thickness.",
                               "Normal rays use the prescribed full-conicity copy closure."]}
    write_json(output / "boundary_layer.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stations", nargs="+", type=float, default=(.02, .1, .5, .9, .98))
    args = parser.parse_args()
    result = diagnose(args.source, args.out, args.stations)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
