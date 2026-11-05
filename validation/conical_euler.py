"""Depth sensitivity of a converged conical section (not wing validation).

    JAX_PLATFORMS=cpu python -m validation.conical_euler --out /tmp/conical.json
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from jaxrans import load_case, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, nargs="?", default=Path("examples/naca0012_conical_euler"))
    parser.add_argument("--depths", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    parser.add_argument("--set", action="append", default=[], help="case/numerical override, repeatable")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--progress", action="store_true", help="write iteration progress to stderr")
    args = parser.parse_args()
    records = []
    for depth in args.depths:
        case = load_case(args.case, tuple(args.set) + (f"numerical.sectional.depth={depth}",))
        if case.grid_spec.sectional is None:
            parser.error("the depth study requires a sectional case")
        def progress(iterations, residual, orders, forces):
            print(f"depth={depth:g}  iter={iterations}  residual={residual:.5e}  "
                  f"drop={orders:.2f}  CL={forces.cl:.9f}", file=sys.stderr, flush=True)

        result = run(case, progress=progress if args.progress else None)
        record = {"depth": depth, "taper_rate": case.grid_spec.sectional.taper_rate,
                  "relaxation_start": case.numerics.sectional_relaxation_start,
                  "relaxation_end": case.numerics.sectional_relaxation_end,
                  "reynolds": case.flow.reynolds, "turbulence": case.flow.turbulence,
                  "cl": result.forces.cl, "cd": result.forces.cd, "cy": result.forces.cy,
                  "cd_friction": result.forces.cd_friction,
                  "iterations": result.history.n_iters, "reason": result.history.reason,
                  "orders_dropped": float(result.history.orders_dropped), "seconds": result.seconds,
                  "finite": bool(np.isfinite(np.asarray(result.U)).all())}
        records.append(record)
        print(json.dumps(record), flush=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(records, indent=2) + "\n")
    return 0 if all(row["finite"] and row["reason"] == "converged" for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
