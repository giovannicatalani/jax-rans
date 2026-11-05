"""Render a sectional qualification summary and optional mesh-refinement plots.

    python -m validation.sectional_report results/sectional_d_qualification --plot
"""

import argparse
import json
from pathlib import Path

import numpy as np


def markdown(summary):
    lines = ["# Sectional qualification: converged limits and mesh sensitivity", "",
             "Numerical verification on NACA 0012; no finite-wing aerodynamic validation.", "",
             "## Converged limits", "",
             "Each pair starts independently from freestream. Stopping uses the residual, not force settling.", "",
             "| model / sweep | result | max coefficient difference | Cp Linf | Cf Linf | field relative Linf |",
             "|---|---|---|---|---|---|"]
    for row in summary["limits"]:
        name = row["reference"].removeprefix("limit_").removesuffix("_legacy")
        if row.get("missing"):
            lines.append(f"| {name} | missing | — | — | — | — |")
            continue
        force = max(row["force_absolute_differences"].values())
        cf = max(row.get(k, {"linf": 0.})["linf"] for k in ("cf", "cf_z"))
        field = max(row["field_relative_linf_by_row"])
        lines.append(f"| {name} | {'PASS' if row['passed'] else 'FAIL'} | {force:.3e} | "
                     f"{row['cp']['linf']:.3e} | {cf:.3e} | {field:.3e} |")
    lines += ["", "Tolerances (absolute for coefficients and wall distributions):", "",
              "```json", json.dumps(summary["tolerances"], indent=2), "```", "",
              "The field comparison includes density, three velocities, pressure and the SA variable where present.", ""]
    for name, group in summary["mesh"].items():
        lines += [f"## {name.capitalize()} study", "",
                  f"All requested solves converged: **{group['converged']}**.", "",
                  "| case | section cells | CL | CD | friction CD | CM | y+ p95 | outer distance | orders |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for case_id in group["cases"]:
            record = summary["runs"].get(case_id)
            if not record:
                lines.append(f"| {case_id} | missing | — | — | — | — | — | — | — |")
                continue
            f = record["forces"]
            shape = " x ".join(map(str, record["shape"][:2]))
            yplus = record.get("yplus_normal_cut", {}).get("p95")
            lines.append(f"| {case_id} | {shape} | {f['cl']:.10f} | {f['cd']:.10f} | "
                         f"{f['cd_friction']:.10f} | {f['cm']:.10f} | "
                         f"{yplus:.3f} | {record['quality']['far_field']:.2f} | {record['orders_dropped']:.2f} |")
        lines += ["", "| change | delta CL | delta CD | relative CL | relative CD | Cp L2 | Cf L2 |", "|---|---|---|---|---|---|---|"]
        for change in group["changes"]:
            f, s = change["force_differences"], change["surface"]
            target = summary["runs"][change["to"]]["forces"]
            relative = [f"{100 * abs(f[k] / target[k]):.4f}%" if target[k] else "n/a" for k in ("cl", "cd")]
            lines.append(f"| {change['from']} → {change['to']} | {f['cl']:+.3e} | {f['cd']:+.3e} | "
                         f"{relative[0]} | {relative[1]} | "
                         f"{s['cp']['l2']:.3e} | {s.get('cf', {'l2': 0.})['l2']:.3e} |")
        if "richardson" in group:
            lines += ["", "Richardson fits from the three finest levels (only monotone, shrinking differences are extrapolated):", "",
                      "| coefficient | observed order | extrapolated value |", "|---|---|---|"]
            for key, fit in group["richardson"].items():
                if fit["order"] is None:
                    lines.append(f"| {key} | unavailable | no extrapolation |")
                else:
                    lines.append(f"| {key} | {fit['order']:.3f} | {fit['extrapolated']:.10f} |")
            lines += ["", "A fit alone does not establish an asymptotic range or certify mesh independence."]
        lines.append("")
    if summary["missing"]:
        lines += ["## Incomplete cases", "", *[f"- {i}" for i in summary["missing"]], ""]
    return "\n".join(lines)


def plots(summary, source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "global" not in summary["mesh"]:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    sizes, cl, cd = [], [], []
    for case_id in summary["mesh"]["global"]["cases"]:
        record = summary["runs"].get(case_id)
        if record is None or record["reason"] != "converged":
            continue
        ni, nj = record["shape"][:2]
        with np.load(source / case_id / "arrays.npz") as arrays:
            x = arrays["x_streamwise"]
            for axis, key in zip(axes[0], ("cp", "cf")):
                line, = axis.plot(x[arrays["arc"] < .5], arrays[key][arrays["arc"] < .5], "--", lw=1)
                axis.plot(x[arrays["arc"] >= .5], arrays[key][arrays["arc"] >= .5],
                          color=line.get_color(), label=f"{ni} × {nj}", lw=1)
        sizes.append(1 / np.sqrt(ni * nj))
        cl.append(record["forces"]["cl"])
        cd.append(record["forces"]["cd"])
    axes[0, 0].invert_yaxis()
    for axis, label in zip(axes[0], ("Cp", "Cf")):
        axis.set(xlabel="Streamwise x/c", ylabel=label)
        axis.legend()
        axis.grid(alpha=.25)
    for axis, values, label in zip(axes[1], (cl, cd), ("CL", "CD")):
        axis.plot(sizes, values, "o-")
        axis.set(xlabel="1 / sqrt(section cells)", ylabel=label)
        axis.grid(alpha=.25)
    fig.suptitle("NACA 0012 conical SA: M=0.5, α=2°, Re=10⁶, LE sweep=30°, taper rate=0.2")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"mesh_refinement.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    output = args.out or args.source
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.source / "summary.json").read_text())
    (output / "summary.md").write_text(markdown(summary))
    if args.plot:
        plots(summary, args.source, output)


if __name__ == "__main__":
    main()
