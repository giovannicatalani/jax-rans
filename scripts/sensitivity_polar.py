"""A lift and drag polar with its derivatives: each angle converged, then
dCL/dalpha and dCD/dalpha from ``jaxrans.sensitivity``, drawn as tangent
lines through the polar's points and checked against its finite slopes.

    python scripts/sensitivity_polar.py CASE_DIR --alpha START STOP STEP
                                        [--set file.section.key=value ...] [--out DIR]

Output: ``sensitivity_polar.dat`` (alpha, CL, CD, dCL/dalpha, dCD/dalpha,
orders dropped, linear residual) and ``sensitivity_polar.png`` in ``--out``
(default ``<save_path>/<case_name>_sensitivity``). The case should converge
to round-off (``forces_tol`` null) with a smooth limiter (docs/gradients.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

from jaxrans import load_case, run, sensitivity
from jaxrans.plot import SERIES, TEXT_SECONDARY, _figure, _save

COLUMNS = "alpha_deg cl cd dcl_dalpha dcd_dalpha orders_dropped linear_residual"


def polar(case, alphas: np.ndarray) -> np.ndarray:
    """One row of ``COLUMNS`` per angle."""
    grid = case.grid()
    rows = []
    for alpha in alphas:
        result = run(dataclasses.replace(case, flow=dataclasses.replace(case.flow, alpha_deg=float(alpha))), grid)
        s = sensitivity(result, ("cl", "cd"), ("alpha",))
        rows.append([alpha, result.forces.cl, result.forces.cd, s.derivatives["cl"]["alpha"],
                     s.derivatives["cd"]["alpha"], result.history.orders_dropped, s.linear_residuals["alpha"]])
        print(f"alpha {alpha:+.2f}: CL {rows[-1][1]:+.6f}  CD {rows[-1][2]:.6f}  dCL/dalpha {rows[-1][3]:+.6f}  "
              f"dCD/dalpha {rows[-1][4]:+.3e}  ({rows[-1][5]:.1f} orders, {result.seconds + s.seconds:.0f} s)")
    return np.array(rows)


def finite_slopes(table: np.ndarray, column: int) -> np.ndarray:
    """Central differences of a column over the interior angles."""
    a, y = table[:, 0], table[:, column]
    return (y[2:] - y[:-2]) / (a[2:] - a[:-2])


def draw(table: np.ndarray, title: str, out: Path) -> None:
    """CL and CD against alpha, each point with its tangent line."""
    fig, axes = _figure(ncols=2, figsize=(9.0, 3.6))
    half = 0.4 * np.min(np.diff(table[:, 0])) if len(table) > 1 else 0.5
    for ax, (value, slope, label) in zip(axes, ((1, 3, "CL"), (2, 4, "CD"))):
        for alpha, y, dy in table[:, [0, value, slope]]:
            ax.plot([alpha - half, alpha + half], [y - half * dy, y + half * dy], color=SERIES[1], lw=2.0)
        ax.plot(table[:, 0], table[:, value], "o", color=SERIES[0], ms=6)
        ax.set_xlabel("α (deg)")
        ax.set_ylabel(label)
    axes[0].set_title(title, loc="left")
    axes[1].set_title("points: converged runs; lines: their derivatives", loc="left",
                      fontweight="normal", color=TEXT_SECONDARY)
    _save(fig, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path)
    parser.add_argument("--alpha", type=float, nargs=3, required=True, metavar=("START", "STOP", "STEP"))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    case = load_case(args.case, tuple(args.overrides))
    start, stop, step = args.alpha
    table = polar(case, np.round(np.arange(start, stop + 0.5 * step, step), 10))
    out = args.out or case.output / f"{case.name}_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "sensitivity_polar.dat", table, header=COLUMNS,
               fmt=["%.4f", "%.8f", "%.8f", "%.8e", "%.8e", "%.2f", "%.1e"])
    if len(table) > 2:
        for column, name in ((3, "dCL/dalpha"), (4, "dCD/dalpha")):
            print(f"{name}: derivative vs the polar's central slope, largest relative difference "
                  f"{np.max(np.abs(table[1:-1, column] / finite_slopes(table, column - 2) - 1)):.1e}")
    f = case.flow
    draw(table, f"{case.name}: M {f.mach:g}, Re {f.reynolds / 1e6:g} million", out / "sensitivity_polar.png")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
