"""Refine only the wall-normal direction, preserving surface nodes exactly.

    python -m validation.sectional_normal_refinement --out results/sectional_c_blunt_normal

Uses the blunt-TE three-block C-grid source. Tangential nodes and both
auxiliary blocks stay fixed; only the main C-block's normal direction changes.
All three solves start independently from freestream.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from validation.sectional_baseline import BASELINE, DEFAULT_OUTPUT, archive_arrays, normal_grid, require_baseline


def worker(source, folder, stride):
    import jax
    from jaxrans import run
    from jaxrans.case import load_case
    from validation.sectional_qualification import source_fingerprint, write_json

    source, folder = Path(source), Path(folder)
    case = load_case(source)
    require_baseline(case)
    if case.grid_spec.sectional is None or case.grid_spec.sectional.taper_rate == 0:
        raise ValueError("this study requires a finite-taper sectional source C-grid")
    grid = normal_grid(case, stride)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "case.json", case.raw["case"])
    write_json(folder / "numerical.json", case.raw["numerical"])
    write_json(folder / "normal_job.json", {"source": str(source.resolve()), "tangential_stride": 2,
                                          "normal_stride": stride})

    def progress(it, residual, orders, forces):
        print(f"{grid.mesh[0].shape}: {it} iterations, {orders:.2f} orders, CL={forces.cl:.9f}", flush=True)

    result = run(case, grid, progress=progress)
    np.savez_compressed(folder / "arrays.npz", **archive_arrays(result))
    write_json(folder / "result.json", {"baseline": BASELINE, "shape": list(grid.mesh[0].shape),
               "block_shapes": [list(b.shape) for b in grid.mesh], "forces": asdict(result.forces),
               "reason": result.history.reason, "orders_dropped": float(result.history.orders_dropped),
               "iterations": result.history.n_iters, "seconds": result.seconds,
               "finite": bool(np.isfinite(np.asarray(result.U)).all()), "dtype": str(result.U.dtype),
               "jax": jax.__version__, "source_sha256": source_fingerprint()})


def main():
    from validation.sectional_qualification import ROOT, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(DEFAULT_OUTPUT) / "mesh_sa_l0")
    parser.add_argument("--out", type=Path, default=Path("results/sectional_c_blunt_normal"))
    parser.add_argument("--worker", type=int, choices=(1, 2, 4), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.source, args.out, args.worker)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for stride in (4, 2, 1):
        folder = args.out.resolve() / f"normal_s{stride}"
        if (folder / "result.json").exists():
            parser.error("results already exist; select a new --out directory")
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "run.log").open("w") as log:
            process = subprocess.run([sys.executable, "-u", "-m", "validation.sectional_normal_refinement",
                                      "--source", str(args.source.resolve()), "--out", str(folder),
                                      "--worker", str(stride)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            print((folder / "run.log").read_text()[-4000:], file=sys.stderr)
            return 1
        row = json.loads((folder / "result.json").read_text())
        rows.append({"folder": str(folder), **row})
        write_json(args.out / "normal_summary.json", rows)
        print(folder.name, row["reason"], row["forces"], flush=True)
    return int(not all(r["reason"] == "converged" and r["finite"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
