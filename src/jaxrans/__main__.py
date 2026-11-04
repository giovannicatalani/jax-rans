"""Command line: run a case or an ICMS campaign and write its results.

    python -m jaxrans CASE_DIR [--set file.section.key=value ...] [--out DIR] [--quiet] [--check]

The case directory holds ``case.json`` and ``numerical.json`` (format:
``docs/guide.md``). ``--set`` changes one key for this run; the resolved
files are saved with the result. ICMS campaigns are detected by their
``campaign``/``sections`` blocks. ``--check`` prepares grids and diagnostic
figures, then stops before solving.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import jax
import numpy as np

from . import CaseError, load_case, run
from .core.state import first_bad_cell
from .mesh.quality import quality


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jaxrans", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path, help="case directory (case.json + numerical.json)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="FILE.SECTION.KEY=VALUE", help="override one key; repeatable")
    parser.add_argument("--out", type=Path, help="output directory (default: <save_path>/<case_name>)")
    parser.add_argument("--quiet", action="store_true", help="no progress lines")
    parser.add_argument("--devices", type=int, default=1,
                        help="split a single case over this many GPUs; campaigns currently use one GPU")
    parser.add_argument("--check", action="store_true",
                        help="prepare grids and diagnostic figures without solving")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # readable while a long run is going

    try:
        case_path = args.case / "case.json" if args.case.is_dir() else args.case
        document = json.loads(case_path.read_text())
        if isinstance(document, dict) and ("campaign" in document or "sections" in document):
            return campaign_main(args)
        case = load_case(args.case, tuple(args.overrides))
        grid = case.grid()
        if case.sequencing:
            case.grid(coarsen=case.sequencing)       # fail now, not after the first levels
        devices = jax.devices()[:args.devices] if args.devices > 1 else None
        if devices is not None and len(devices) < args.devices:
            raise ValueError(f"--devices {args.devices}, but only {len(devices)} device(s) visible")
    except (CaseError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"{case.name}: {case.description}")
    print(f"grid {case.grid_spec.describe(case.directory)}: {len(grid.mesh)} block(s), {grid.cells} cells, "
          f"{grid.walls.faces.bodies} body(ies); {case.numerics.solver}, {case.numerics.flux}, "
          f"limiter {case.numerics.limiter}")
    print(quality(grid.mesh).summary())
    if args.check:
        return check(case, grid, args.out)

    def progress(iteration, residual, orders, forces):
        print(f"{iteration:8d}  res {residual:.4e} ({orders:5.2f} orders)  "
              f"{forces.summary().splitlines()[0]}")

    def on_level(level, atlas):
        name = f"coarse level {level} (1/{2 ** level} of the nodes in each direction)" if level else "the case's grid"
        print(f"grid sequencing: {name}, {atlas.cells} cells")

    result = run(case, grid, progress=None if args.quiet else progress, on_level=on_level, devices=devices)
    directory = result.write(args.out)
    for k, level in enumerate(result.levels):
        print(f"level {len(result.levels) - k}: {level['cells']} cells, {level['exit_reason']} after "
              f"{level['iterations']} iterations, {level['seconds']:.1f} s")
    print(f"{result.history.summary()}\n{result.forces.summary()}\n{result.timing.summary()}\n"
          f"written to {directory}")
    physical, cell = first_bad_cell(result.U, case.flow.gamma, grid.components)
    if not physical:
        at = ", ".join(f"{float(c):.4f}" for c in grid.grid.centres[(slice(None),) + cell])
        print(f"non-physical state, first at canvas cell {cell}: ({at})", file=sys.stderr)
    return 0 if physical and result.history.reason != "non_finite" else 1


def campaign_main(args) -> int:
    """The same entry point and overrides, with campaign-specific preparation."""
    from .campaign import load_campaign, prepare_campaign, run_campaign

    try:
        if args.devices != 1:
            raise CaseError("ICMS campaigns currently use one GPU; --devices must be 1")
        campaign = load_campaign(args.case, tuple(args.overrides))
        count = campaign.sections["count"] * len(campaign.machs) * len(campaign.alphas)
        print(f"{campaign.template.name}: ICMS campaign, {campaign.sections['count']} sections, "
              f"{len(campaign.machs)} Mach x {len(campaign.alphas)} alpha = {count} cases; "
              f"batch size {campaign.batch_size}")
        prepared = prepare_campaign(campaign, args.out)
        if not args.quiet:
            for section, grid in zip(prepared.sections, prepared.grids):
                print(f"section {section.index:03d}: chord {section.chord:.6g} m, "
                      f"TE-up pitch {section.chord_pitch_deg:.4f} deg, sweep {section.sweep_deg:.4f} deg; "
                      f"{grid.cells} cells")
                print(quality(grid.mesh).summary())
        print(f"preprocessing written to {prepared.directory}")
        if args.check:
            print("check complete: HUG files, geometric reports and figures generated; no CFD calculations")
            return 0

        def progress(iteration, reasons):
            status = ", ".join(f"{reason}={n}" for reason, n in Counter(reasons).items())
            print(f"batch iteration {iteration:8d}: {status}")

        result = run_campaign(prepared, progress=None if args.quiet else progress)
        status = ", ".join(f"{reason}={n}" for reason, n in Counter(
            row["exit_reason"] for row in result.rows).items())
        print(f"campaign complete: {len(result.rows)} cases; {status}\n"
              f"written to {result.directory / 'campaign.csv'}")
        return 1 if any(row["exit_reason"] == "non_finite" for row in result.rows) else 0
    except (CaseError, OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def check(case, grid, out: Path | None) -> int:
    """Patches with their conditions, bodies with their extent, and a picture."""
    faces: dict[str, list] = {}
    for block in grid.mesh:
        for _, seg in block.sides.boundaries():
            n = int(np.prod(np.subtract(seg.stop, seg.start)))      # a segment, or a 3D rectangle
            faces.setdefault(seg.patch, [seg.kind, 0])[1] += n
    for patch, (kind, n) in faces.items():
        print(f"patch {patch!r}: {kind}, {n} faces")
    walls = grid.walls.faces
    if grid.dim == 3:
        lo, hi = walls.corners.min(axis=(0, 1)), walls.corners.max(axis=(0, 1))
        print(f"wall: {len(walls.mid)} faces, x {lo[0]:.4f} to {hi[0]:.4f}, y {lo[1]:.4f} to {hi[1]:.4f}, "
              f"z {lo[2]:.4f} to {hi[2]:.4f}")
        print("no grid picture: the picture is 2D only")
        return 0
    for body in range(walls.bodies):
        x = walls.a[walls.body == body]
        print(f"body {body}: x {x[:, 0].min():.4f} to {x[:, 0].max():.4f}, "
              f"y {x[:, 1].min():.4f} to {x[:, 1].max():.4f}, {len(x)} wall faces")
    if importlib.util.find_spec("matplotlib") is None:
        print("no grid picture: pip install matplotlib")
        return 0
    from .plot import plot_grid

    path = (out if out is not None else case.output / case.name) / "grid.png"
    plot_grid(grid, out=path)
    print(f"grid drawn in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
