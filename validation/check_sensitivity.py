"""A case's sensitivities against central finite differences.

    python validation/check_sensitivity.py CASE_DIR [--parameters alpha mach]
                                           [--set file.section.key=value ...] [--out FILE]

Converges the case to round-off (``tol`` 1e-12, no forces stop), takes the
derivatives of CL, CD and CM, then converges it again at each parameter plus
and minus its step (alpha 0.01 deg, Mach 1e-4) and compares. Prints the table
and the costs; ``--out`` writes them as JSON.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from jaxrans import load_case, run, sensitivity
from jaxrans.derivatives import PARAMETERS

OUTPUTS = ("cl", "cd", "cm")
STEPS = {"alpha": 0.01, "mach": 1e-4}
FIELDS = {"alpha": "alpha_deg", "mach": "mach"}
ROUND_OFF = ("numerical.convergence.tol=1e-12", "numerical.convergence.forces_tol=null")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path)
    parser.add_argument("--parameters", nargs="+", default=["alpha"], choices=PARAMETERS)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    case = load_case(args.case, ROUND_OFF + tuple(args.overrides))
    grid = case.grid()
    base = run(case, grid)
    print(f"{case.name}: {grid.cells} cells, {base.history.summary()}, {base.seconds:.1f} s")
    s = sensitivity(base, OUTPUTS, tuple(args.parameters))
    print(s.summary())

    rows = []
    for name in args.parameters:
        field = FIELDS[name]
        value = getattr(case.flow, field)
        plus, minus = (run(dataclasses.replace(case, flow=dataclasses.replace(
            case.flow, **{field: value + sign * STEPS[name]})), grid) for sign in (1, -1))
        for out in OUTPUTS:
            fd = (getattr(plus.forces, out) - getattr(minus.forces, out)) / (2 * STEPS[name])
            t = s.derivatives[out][name]
            rows.append({"output": out, "parameter": name, "tangent": t, "finite_difference": fd,
                         "relative_difference": abs(t - fd) / abs(fd),
                         "orders_dropped": [plus.history.orders_dropped, minus.history.orders_dropped]})
    print(f"{'':12s}{'tangent':>17s}{'central FD':>17s}{'|t-fd|/|fd|':>13s}")
    for r in rows:
        print(f"d{r['output']}/d{r['parameter']:6s}{r['tangent']:+17.9e}{r['finite_difference']:+17.9e}"
              f"{r['relative_difference']:13.1e}")
    if args.out:
        args.out.write_text(json.dumps({
            "case": case.name, "cells": grid.cells, "overrides": list(ROUND_OFF + tuple(args.overrides)),
            "forward": {"seconds": base.seconds, "iterations": base.history.n_iters,
                        "orders_dropped": base.history.orders_dropped},
            "sensitivity": {"seconds": s.seconds, "compile_seconds": s.compile_seconds,
                            "linear_residuals": s.linear_residuals},
            "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
