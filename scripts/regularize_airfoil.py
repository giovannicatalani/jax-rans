"""Compare bounded geometry regularization budgets without running CFD."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from jaxrans.geometry.airfoil import read_airfoil
from jaxrans.geometry.regularize import regularize_airfoil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("airfoil", type=Path)
    parser.add_argument("--budgets", type=float, nargs="+", default=[1e-5, 5e-5, 1e-4])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--method", choices=("parametric", "bspline", "curvature", "both", "all"), default="parametric")
    parser.add_argument("--control-points", type=int, default=96)
    parser.add_argument("--radius-tolerance", type=float, default=.1)
    args = parser.parse_args()
    original = read_airfoil(args.airfoil).unit_chord()
    args.out.mkdir(parents=True, exist_ok=True)
    variants = []
    reports = []
    methods = (("parametric", "bspline", "curvature") if args.method == "all" else
               ("parametric", "bspline") if args.method == "both" else (args.method,))
    for method in methods:
        for budget in args.budgets:
            fitted, report = regularize_airfoil(original, budget, method=method,
                                                control_points=args.control_points,
                                                radius_tolerance=args.radius_tolerance)
            stem = "regularized" if method == "parametric" else method
            path = args.out / f"{stem}_{budget:g}.dat"
            np.savetxt(path, np.column_stack([fitted.x, fitted.y]),
                       header=f"{original.name}; {method}; budget {budget:g} c", fmt="%.16e")
            report["coordinates"] = path.name
            if hasattr(fitted, "curves") and budget:
                representation = path.with_suffix(".npz")
                np.savez(representation, parameter=fitted.s, knots=fitted.curves[0].t,
                         coefficients=np.column_stack([f.c for f in fitted.curves]),
                         degree=fitted.curves[0].k)
                report["bspline_representation"] = representation.name
                report["coordinates_note"] = "sampled landmarks; NPZ retains the exact C4 curve"
            variants.append((f"{method} {budget:g} c", fitted))
            reports.append(report)
    payload = {"source": str(args.airfoil.resolve()),
               "source_sha256": hashlib.sha256(args.airfoil.read_bytes()).hexdigest(),
               "parameters": {"method": args.method, "budgets": args.budgets,
                              "control_points": args.control_points,
                              "radius_tolerance": args.radius_tolerance},
               "normalized_to_unit_chord": True, "variants": reports}
    (args.out / "comparison.json").write_text(json.dumps(payload, indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for label, fitted in [("original", original), *variants]:
            sa = np.linspace(0, original.length, 12001)
            sb = np.interp(sa, original.s, fitted.s)
            fx, fy = fitted.spline()
            ax, ay = original.spline()
            x, y = fx(sb), fy(sb)
            dx, dy = fx(sb, 1), fy(sb, 1)
            k = (dx * fy(sb, 2) - dy * fx(sb, 2)) / (dx**2 + dy**2)**1.5
            speed = np.hypot(ax(sa, 1), ay(sa, 1))
            normal = (-(x - ax(sa)) * ay(sa, 1) + (y - ay(sa)) * ax(sa, 1)) / speed
            for row, mask in enumerate([sa <= original.s[original.leading_edge],
                                        sa >= original.s[original.leading_edge]]):
                axes[row, 0].plot(x[mask], k[mask], label=label)
                axes[row, 1].plot(x[mask], normal[mask], label=label)
        for row, side in enumerate(["Intrados", "Extrados"]):
            axes[row, 0].set(ylabel=f"{side}: courbure (1/c)", ylim=(-4, 2))
            axes[row, 1].set(ylabel=f"{side}: deplacement normal / c")
            for ax in axes[row]:
                ax.set(xlabel="x/c", xlim=(0, 1))
                ax.grid(alpha=.3)
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / "comparison.png", dpi=170)
        plt.close(fig)
    print(args.out / "comparison.json")


if __name__ == "__main__":
    main()
