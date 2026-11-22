"""Compare the normal-only study and render its boundary-layer evidence.

    python -m validation.sectional_boundary_report --source results/sectional_c_blunt_normal \
      --out reports/sectional_c_blunt_boundary_layer
"""

import argparse
import json
from pathlib import Path

import numpy as np

from validation.sectional_boundary_layer import diagnose, first_gap_for_yplus, layer_design, layers_through_thickness
from validation.sectional_qualification import write_json
from validation.sectional_baseline import BASELINE, require_baseline


def render(source, output):
    from jaxrans.case import load_case
    source, output = Path(source), Path(output)
    shapes = ("normal_s4", "normal_s2", "normal_s1")
    runs, diagnostics = {}, {}
    for shape in shapes:
        runs[shape] = json.loads((source / shape / "result.json").read_text())
        if runs[shape].get("baseline") != BASELINE:
            raise ValueError(f"{shape}: this report requires the current blunt C-grid baseline")
        if runs[shape]["reason"] != "converged" or not runs[shape]["finite"]:
            raise ValueError(f"{shape}: boundary-layer comparisons require a converged finite solve")
        diagnostics[shape] = diagnose(source / shape, output / shape)
    rows = []
    case = load_case(source / shapes[0])
    require_baseline(case)
    flow, spec = case.flow, case.grid_spec.sectional
    lines = ["# Normal-only boundary-layer refinement", "",
             f"Conical blunt-TE NACA 0012, C-grid, M={flow.mach:g}, alpha={flow.alpha_deg:g} deg, Re={flow.reynolds:g}, "
             f"LE sweep={spec.leading_edge_sweep_deg:g} deg, taper rate={spec.taper_rate:g}.", "",
             "All grids share their surface nodes and outer boundary; base and wake blocks are fixed.",
             "All three solves start independently from freestream; achieved residual reductions appear below.", "",
             "| grid | CL | CD | friction CD | iterations | orders |",
             "|---|---|---|---|---|---|"]
    for shape, r in runs.items():
        f = r["forces"]
        lines.append(f"| {shape} | {f['cl']:.10f} | {f['cd']:.10f} | {f['cd_friction']:.10f} | {r['iterations']} | {r['orders_dropped']:.2f} |")
    changes = {}
    for a, b in zip(shapes, shapes[1:]):
        changes[f"{a}_to_{b}"] = {k: 100 * abs(runs[a]["forces"][k] / runs[b]["forces"][k] - 1)
                                    for k in ("cl", "cd", "cd_friction")}
    lines += ["", "## Profiles and operational thickness estimates", "",
              "True 3D normal rays are mapped back to the independent section through the full-conicity closure.",
              "Piecewise-linear interpolation uses the mesh's dual connectivity, within the main C-block, using actual wall/interface connectivity.",
              "Dense interpolation locates the edge; it does not add physical grid cells.", "",
              "The speed reference follows an ideal-gas isentropic pressure reconstruction from the freestream",
              "stagnation state, subtracting measured normal velocity. The speed-based edge is |V_t|/U_I=.99.",
              "The reference must also recover within 1.5% at twice that thickness; otherwise no estimate is accepted.",
              "Scalar displacement/momentum deficits use rho_I=(gamma*p)^(1/gamma) and a direction fixed at 2*delta99.",
              "These are operational locally planar indicators, not a general 3D displacement tensor.", "",
              "| grid | side | requested x/c | actual x/c | delta99/c | delta*/c | theta/c | native centres below delta99 | first-centre y+ (plane) |",
              "|---|---|---|---|---|---|---|---|---|"]
    keys = ("delta99_speed", "delta_star_scalar", "theta_scalar")
    for shape, diag in diagnostics.items():
        for p in diag["profiles"]:
            if p["status"] != "estimated":
                lines.append(f"| {shape} | {p['side']} | {p['requested_x']} | — | {p['status']} | — | — | — | — |")
                continue
            lines.append(f"| {shape} | {p['side']} | {p['requested_x']:.2f} | {p['actual_x']:.4f} | "
                         f"{p['delta99_speed']:.6f} | {p['delta_star_scalar']:.6f} | {p['theta_scalar']:.6f} | "
                         f"{p['native_centres_below_delta99']} | {p['yplus_first_centre_plane']:.3f} |")
    for old, fine in zip(diagnostics["normal_s2"]["profiles"], diagnostics["normal_s1"]["profiles"]):
        if old["status"] == fine["status"] == "estimated":
            assert old["actual_x"] == fine["actual_x"]
            label = f"{fine['side']}_{fine['requested_x']:g}.npz"
            with np.load(output / "normal_s2" / label) as a, np.load(output / "normal_s1" / label) as b:
                d = b["distance"]
                mask = d <= fine["delta99_speed"]
                v_old = np.stack([np.interp(d, a["distance"], u) for u in a["primitive"][1:4]])
                v = b["primitive"][1:4]
                velocity_error = 100 * np.sqrt(np.trapezoid(np.sum((v_old - v)**2, axis=0)[mask], d[mask])
                                               / np.trapezoid(np.sum(v * v, axis=0)[mask], d[mask]))
                eddy = b["eddy_viscosity_ratio"]
                eddy_old = np.interp(d, a["distance"], a["eddy_viscosity_ratio"])
                denominator = np.trapezoid(eddy[mask]**2, d[mask])
                eddy_error = 100 * np.sqrt(np.trapezoid((eddy_old[mask] - eddy[mask])**2, d[mask]) / denominator) if denominator > 0 else None
            rows.append({"side": fine["side"], "x": fine["requested_x"],
                         **{key: 100 * abs(old[key] / fine[key] - 1) for key in keys},
                         "velocity_profile_l2_percent": float(velocity_error),
                         "eddy_viscosity_profile_l2_percent": float(eddy_error) if eddy_error is not None else None})
    maxima = {k: max((r[k] for r in rows), default=None) for k in keys}
    profile_maxima = {k: max((r[k] for r in rows if r[k] is not None), default=None)
                      for k in ("velocity_profile_l2_percent", "eddy_viscosity_profile_l2_percent")}
    thicknesses = [p["delta99_speed"] for p in diagnostics["normal_s1"]["profiles"] if p["status"] == "estimated"]
    design = layer_design(flow.reynolds, .015, min(thicknesses), 1.5 * max(thicknesses)) if thicknesses else None
    wall_changes = {}
    with np.load(source / "normal_s2" / "arrays.npz") as old, np.load(source / "normal_s1" / "arrays.npz") as fine:
        for key in ("cp", "cf", "cf_z"):
            delta = old[key] - fine[key]
            denominator = np.average(fine[key]**2, weights=fine["weight"])
            wall_changes[key] = {"linf_absolute": float(abs(delta).max()),
                                 "relative_l2_percent": float(100 * np.sqrt(np.average(delta**2, weights=fine["weight"]) / denominator)) if denominator > 0 else None}
    if len(rows) != 10:
        lines += ["", "Some profiles are unavailable: the thickness comparison is incomplete."]
    lines += ["", "## Two finest normal resolutions", "",
              f"Relative coefficient changes (%): `{changes['normal_s2_to_normal_s1']}`.",
              f"Maximum relative thickness changes over available stations (%): `{maxima}`.", "",
              f"Maximum physical-profile L2 changes inside the finest estimated layer (%): `{profile_maxima}`.", "",
              f"Surface changes: `{wall_changes}`.", "",
              "A vector-defect thickness is also saved: 1% mismatch relative to the direction at 2*delta99.",
              "Its difference from the speed-based edge measures sensitivity to crossflow/edge definition,",
              "rather than an independent proof of the physical layer edge.", "",
              "This verifies numerical sensitivity on this model and these stations. It does not validate",
              "the conical closure against a 3D wing, certify all Reynolds numbers, or establish tangential-grid independence.", "",
              "## Reynolds-aware initial sizing", "",
              "For rho_w/rho_inf=mu_w/mu_inf=1 and a cell centre at half the first gap:",
              "`first_gap/c = 2*yplus_target / (Re_c*sqrt(Cf_estimate/2))`.",
              "Cf must be estimated conservatively; verify actual wall units after solving.",
              "To cover an estimated layer thickness with geometric growth r, choose N so",
              "`first_gap*(r**N-1)/(r-1) >= layer_thickness`.",
              "An example at Re=1e6, Cf_estimate=.015, yplus_target=.5, layer_thickness=.04 and r=1.15 gives",
              f"a first gap {first_gap_for_yplus(1e6,.015):.4e} chords and {layers_through_thickness(.04,first_gap_for_yplus(1e6,.015))} layers.",
              "This is preliminary sizing, not automatic Reynolds-aware adaptation in the solver.", "",
              "Coupling the first-gap wall-unit limit with at least 30 cells in the thinnest sampled layer",
              "(growth 1.1, Cf estimate .015, outer layer target 1.5 times the thickest sampled layer) gives:",
              f"`{design}`", "",
              "These estimates cover only sampled locations and need compressibility/geometry and a-posteriori checks.", "",
              "Reference for the local-reconstruction principle (the exact published algorithm differs):",
              "[Griffin, Fu and Moin (2021)](https://doi.org/10.1103/PhysRevFluids.6.024608)."]
    summary = {"runs": runs, "diagnostics": diagnostics, "coefficient_change_percent": changes,
               "thickness_change_percent_two_finest": rows, "max_thickness_change_percent": maxima,
               "surface_change_two_finest": wall_changes, "preliminary_layer_design": design}
    summary["max_profile_change_percent"] = profile_maxima
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return summary


def plots(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, stations, crossflow in (("velocity_profiles", (.1, .5, .9), False),
                                      ("crossflow_profiles", (.1, .5, .9), True),
                                      ("eddy_viscosity_profiles", (.1, .5, .9), False),
                                      ("leading_trailing_profiles", (.02, .98), False)):
        fig, axes = plt.subplots(2, len(stations), figsize=(4 * len(stations), 7), layout="constrained")
        for row, side in enumerate(("lower", "upper")):
            for col, x in enumerate(stations):
                ax = axes[row, col]
                fine = json.loads((output / "normal_s1" / "boundary_layer.json").read_text())
                profile = next(p for p in fine["profiles"] if p["side"] == side and p["requested_x"] == x)
                height = 1.5 * profile.get("delta99_speed", .03)
                for shape in ("normal_s4", "normal_s2", "normal_s1"):
                    path = output / shape / f"{side}_{x:g}.npz"
                    if not path.exists():
                        continue
                    with np.load(path) as a:
                        d, q, normal, ui = a["distance"], a["primitive"], a["normal"], a["reference_speed"]
                        value = a["speed_ratio"]
                        if name == "eddy_viscosity_profiles":
                            value = a["eddy_viscosity_ratio"]
                        if crossflow:
                            v = q[1:4] - normal[:, None] * (normal @ q[1:4])
                            own = json.loads((output / shape / "boundary_layer.json").read_text())
                            own_profile = next(p for p in own["profiles"] if p["side"] == side and p["requested_x"] == x)
                            edge = np.array([np.interp(2 * own_profile.get("delta99_speed", .02), d, u) for u in v])
                            edge /= np.linalg.norm(edge)
                            value = np.cross(normal, edge) @ v / ui
                        mask = d <= height
                        ax.plot(value[mask], d[mask], label=shape)
                if not crossflow and name != "eddy_viscosity_profiles":
                    ax.axvline(.99, color="k", ls=":", lw=.7)
                label = "mu_t / mu" if name == "eddy_viscosity_profiles" else ("W / UI" if crossflow else "|Vt| / UI")
                ax.set(title=f"{side}, x/c ≈ {x:g}", xlabel=label, ylabel="n/c")
                ax.grid(alpha=.25)
                ax.legend()
        fig.suptitle("Blunt-TE C-grid: fixed surface/base/wake; normal field profiles")
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/sectional_c_blunt_normal"))
    parser.add_argument("--out", type=Path, default=Path("reports/sectional_c_blunt_boundary_layer"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    render(args.source, args.out)
    if args.plot:
        plots(args.out)


if __name__ == "__main__":
    main()
