"""Depth sensitivity of a converged conical Euler section (not wing validation).

    JAX_PLATFORMS=cpu python -m validation.conical_euler --out /tmp/conical.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from jaxrans import load_case, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, nargs="?", default=Path("examples/naca0012_conical_euler"))
    parser.add_argument("--depths", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    parser.add_argument("--set", action="append", default=[], help="case/numerical override, repeatable")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    records = []
    for depth in args.depths:
        case = load_case(args.case, tuple(args.set) + (f"numerical.sectional.depth={depth}",))
        if case.grid_spec.sectional is None or not case.flow.inviscid:
            parser.error("the depth study requires a sectional Euler case")
        result = run(case)
        record = {"depth": depth, "taper_rate": case.grid_spec.sectional.taper_rate,
                  "relaxation_start": case.numerics.sectional_relaxation_start,
                  "relaxation_end": case.numerics.sectional_relaxation_end,
                  "cl": result.forces.cl, "cd": result.forces.cd, "cy": result.forces.cy,
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
