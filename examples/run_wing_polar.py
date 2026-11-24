"""Polar of a 3D wing by continuation, one chain per Mach number.

    python examples/run_wing_polar.py CASE_DIR --out DIR --machs 0.70 0.78 --alphas 0 1 2 3 [--set file.section.key=value]

The first alpha of each Mach starts from the free stream; every next one
restarts from the previous converged field on the same grid (``run(...,
initial=U)``). A chain stops at its first point that neither converged nor
settled. ``jaxrans.campaign`` is 2D only, hence this script.
Results: DIR/mach_<M>/alpha_<A>/ (fields, surface, vts) and DIR/polar.csv,
appended after every point.
"""
import argparse
import csv
import sys
from pathlib import Path

import jax

from jaxrans import load_case, run

COLUMNS = ["mach", "alpha_deg", "cl", "cd", "cd_friction", "cm", "exit_reason", "iterations", "seconds"]
OK = ("converged", "settled")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--machs", type=float, nargs="+", required=True)
    p.add_argument("--alphas", type=float, nargs="+", required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="FILE.SECTION.KEY=VALUE")
    p.add_argument("--devices", type=int, default=1)
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    devices = jax.devices()[:args.devices] if args.devices > 1 else None

    grid = None
    args.out.mkdir(parents=True, exist_ok=True)
    table = args.out / "polar.csv"
    new = not table.exists()
    status = 0
    with table.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(COLUMNS)
        for mach in args.machs:
            U = None
            for alpha in args.alphas:
                case = load_case(args.case, tuple(args.overrides) + (
                    f"case.freestream.mach={mach!r}", f"case.freestream.alpha_deg={alpha!r}"))
                if grid is None:
                    grid = case.grid()
                    print(f"{case.name}: {grid.cells} cells", flush=True)

                def progress(it, res, orders, forces):
                    print(f"M {mach:g} a {alpha:+g} {it:7d} res {res:.3e} ({orders:5.2f} orders) "
                          f"{forces.summary().splitlines()[0]}", flush=True)

                result = run(case, grid, progress=progress, initial=U, devices=devices)
                result.write(args.out / f"mach_{mach:g}" / f"alpha_{alpha:+g}")
                fc, reason = result.forces, result.history.reason
                writer.writerow([mach, alpha, fc.cl, fc.cd, fc.cd_friction, fc.cm, reason,
                                 result.history.n_iters, f"{result.seconds:.1f}"])
                f.flush()
                print(f"M {mach:g} a {alpha:+g}: {reason}, CL {fc.cl:+.5f} CD {fc.cd:+.5f}", flush=True)
                if reason not in OK:
                    print(f"chain M {mach:g} stopped at alpha {alpha:+g}", file=sys.stderr)
                    status = 1
                    break
                U = result.U
    return status


if __name__ == "__main__":
    raise SystemExit(main())
