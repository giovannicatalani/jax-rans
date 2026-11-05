"""Converged 2D/2.5D limits and separate conical mesh sensitivity studies.

    python -m validation.sectional_qualification --study all --out results/sectional_d
    python -m validation.sectional_qualification --study all --plan

Each solve runs in its own process, saves its inputs, convergence and arrays,
and can be resumed. Mesh sensitivity is measured, never silently called mesh
independence. No finite-wing aerodynamic reference is assumed here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("euler", "laminar", "sa", "qcr")
COEFFICIENTS = ("cl", "cd", "cm", "cd_friction", "cy")
TOLERANCES = {"coefficient_abs": 1e-6, "cp_linf": 1e-5, "cf_linf": 1e-6, "field_relative": 1e-5}
PROTOCOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # NaN/inf must not turn a failed run into a usable cached result.
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        return None if isinstance(x, float) and not np.isfinite(x) else x
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plan(study="all", physics=MODELS, sweeps=(0., 30.), max_iters=3000, tol=1e-10):
    jobs = []
    if study in ("all", "limits"):
        for model in physics:
            for sweep in sweeps:
                for formulation in ("legacy", "sectional"):
                    jobs.append(dict(id=f"limit_{model}_{sweep:g}_{formulation}", study="limits",
                                     physics=model, sweep=sweep, taper=0., formulation=formulation,
                                     level=1, family="limit", wall_factor=1., outer_factor=1.))
    if study in ("all", "mesh"):
        # A nested family: subsampling one 256x192 mesh by factors 8,4,2,1.
        # The same medium case anchors both one-parameter sensitivity studies.
        for level in (3, 2, 1, 0):
            jobs.append(dict(id=f"mesh_sa_l{level}", study="mesh", physics="sa", sweep=30., taper=.2,
                             formulation="sectional", family="mesh", level=level,
                             wall_factor=1., outer_factor=1.))
        for factor in (.5, .25):
            jobs.append(dict(id=f"wall_sa_{factor:g}", study="mesh", physics="sa", sweep=30., taper=.2,
                             formulation="sectional", family="wall", level=2,
                             wall_factor=factor, outer_factor=1.))
        for factor in (2., 4.):
            jobs.append(dict(id=f"outer_sa_{factor:g}", study="mesh", physics="sa", sweep=30., taper=.2,
                             formulation="sectional", family="outer", level=2,
                             wall_factor=1., outer_factor=factor))
    for job in jobs:
        job.update(max_iters=max_iters, tol=tol)
    return jobs


def documents(job):
    model = job["physics"]
    # Limits on a 32x32 grid, sampled from the same 64x64 source on both paths.
    limits = job["study"] == "limits"
    first = {"euler": 5e-4, "laminar": 1e-4, "sa": 1e-5, "qcr": 1e-5}[model]
    if not limits:
        first *= .5
    doc = {"general": {"case_name": job["id"]},
           "domain": {"grid": "o_grid", "airfoil": "naca0012", "cells_airfoil": 64 if limits else 256,
                      "cells_normal": 64 if limits else 192, "wall_spacing": first * job["wall_factor"],
                      "far_field": 50., "leading_edge_spacing": .001 if limits else .0005,
                      "trailing_edge_spacing": .001 if limits else .0005},
           "freestream": {"mach": .5, "alpha_deg": 2.}}
    if model != "euler":
        doc["freestream"]["reynolds"] = 5000. if model == "laminar" else 1e6
    if model in ("sa", "qcr"):
        doc["turbulence"] = {"model": "sa_noft2_qcr" if model == "qcr" else "sa_noft2"}
    numerical = {"convective_fluxes": {"limiter": "none"}, "time_integration": {"solver": "implicit"},
                 "implicit": {"max_relative_update": .2},
                 "convergence": {"max_iters": job["max_iters"], "tol": job["tol"],
                                 "forces_tol": None, "block_iters": 50}}
    if job["formulation"] == "sectional":
        doc["domain"]["sectional"] = {"model": "conical", "leading_edge_sweep_deg": job["sweep"],
                                      "taper_rate": job["taper"]}
        numerical["sectional"] = {"depth": .1, "relaxation_start": 1., "relaxation_end": 5.}
    elif job["sweep"] != 0.:
        doc["freestream"]["sweep_deg"] = job["sweep"]
    return doc, numerical


def extend_outer(block, factor):
    """Extend a normal-plane O-grid, retaining EVERY original node exactly."""
    from jaxrans.geometry.conical import distance_to_section
    from jaxrans.mesh.block import BlockSides, Boundary, Grid, Interface
    from jaxrans.mesh.generate import march
    from jaxrans.mesh.mesh import wall_faces

    if factor <= 1:
        return block
    xy = np.asarray(block.nodes)
    walls = wall_faces((block,))
    target = factor * float(distance_to_section(xy[:, :, -1], walls).min())
    line = xy[:, :-1, -1].T
    step = float(np.median(np.linalg.norm(xy[:, :-1, -1] - xy[:, :-1, -2], axis=0)))
    added = []
    for _ in range(100):
        steps = step * 1.2 ** np.arange(1, 3)
        layers = march(line, steps, periodic=True)
        added.append(np.concatenate([layers[:, 1:], layers[:1, 1:]], axis=0).transpose(2, 0, 1))
        line, step = layers[:, -1], float(steps[-1])
        if distance_to_section(line.T, walls).min() >= target:
            break
    else:
        raise ValueError("outer extension did not reach the requested boundary")
    nodes = np.concatenate([xy, *added], axis=-1)
    ni, nj = nodes.shape[1] - 1, nodes.shape[2] - 1
    sides = BlockSides(ni, nj, (Boundary("wall", 0, ni, "no_slip_wall"),),
                       (Boundary("farfield", 0, ni, "far_field"),),
                       (Interface(0, nj, 0, "i_max", 0, False),),
                       (Interface(0, nj, 0, "i_min", 0, False),))
    return Grid.from_node_array(nodes, sides)


def build_grid(case, job):
    grid = case.grid(coarsen=job["level"])
    if job["outer_factor"] == 1.:
        return grid
    from jaxrans.mesh.sectional import build_sectional_atlas
    if len(grid.reference.mesh) != 1:
        raise ValueError("this outer-boundary study expects a single O-grid block")
    mesh = extend_outer(grid.reference.mesh[0], job["outer_factor"])
    return build_sectional_atlas(mesh, grid.spec, grid.g, grid.depth,
                                 grid.relaxation_start, grid.relaxation_end, grid.geometry)


def source_fingerprint():
    digest = hashlib.sha256()
    digest.update(PROTOCOL_SHA256.encode())
    for path in sorted((ROOT / "src" / "jaxrans").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_job(job, folder):
    import jax
    import jax.numpy as jnp
    from jaxrans import run
    from jaxrans.case import parse_case
    from jaxrans.core.state import Rows
    from jaxrans.mesh.quality import quality
    from jaxrans.numerics.residual import make_residual, primitive_from_conservative
    from jaxrans.post.forces import wall_loads

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    doc, numerical = documents(job)
    write_json(folder / "case.json", doc)
    write_json(folder / "numerical.json", numerical)
    case = parse_case(doc, numerical, folder)
    grid = build_grid(case, job)
    print(f"{job['id']}: {grid.cells} cells on {jax.devices()}", flush=True)

    def progress(it, residual, orders, forces):
        print(f"  {it:5d}  {orders:6.2f} orders  CL={forces.cl:.10f} CD={forces.cd:.10f}", flush=True)

    result = run(case, grid, progress=progress)
    surface = result.surface()
    # Canonical primitives: rho,u,v,w,p[,nu_hat], even for native 2D.
    primitive = np.asarray(grid.unpack(primitive_from_conservative(result.U, case.flow.gamma, grid.components))[0])
    if grid.dim == 3:
        primitive = primitive[..., 0]
    if grid.components == 2:
        primitive = np.concatenate([primitive[:3], np.zeros_like(primitive[:1]), primitive[3:]])
    faces = grid.reference.walls.faces if hasattr(grid, "reference") else grid.walls.faces
    a, b = faces.a.T, faces.b.T
    if getattr(grid, "geometry", None) is not None:
        a, b = (grid.geometry.streamwise_points(p) for p in (a, b))
    elif case.flow.swept:
        a, b = a / np.array([[np.cos(case.flow.sweep_rad)], [1.]]), b / np.array([[np.cos(case.flow.sweep_rad)], [1.]])
    ds = np.linalg.norm(b - a, axis=0)
    arc = (np.cumsum(ds) - .5 * ds) / ds.sum()
    op = make_residual(grid, case.flow, case.numerics)
    residual = np.asarray(op.residual(result.U) / op.grid.volume)[:, grid.interior]
    record = {"job": job, "source_sha256": source_fingerprint(), "jax": jax.__version__,
              "devices": [str(d) for d in jax.devices()], "dtype": str(result.U.dtype),
              "cells": grid.cells, "shape": list(grid.mesh[0].shape),
              "quality": quality(grid.reference.mesh if hasattr(grid, "reference") else grid.mesh).record(),
              "forces": asdict(result.forces), "reason": result.history.reason,
              "orders_dropped": float(result.history.orders_dropped), "iterations": result.history.n_iters,
              "seconds": result.seconds, "finite": bool(np.isfinite(np.asarray(result.U)).all()),
              "equation_residual_rms_per_volume": np.sqrt(np.mean(residual**2, axis=1)).tolist()}
    if not case.flow.inviscid:
        _, area, integrated = wall_loads(result.U, grid, case.flow)
        area, traction = np.asarray(area), np.asarray(integrated) / np.linalg.norm(area, axis=0)
        normal = area / np.linalg.norm(area, axis=0)
        traction -= normal * np.sum(traction * normal, axis=0)
        q = primitive_from_conservative(result.U, case.flow.gamma, grid.components)
        cell = grid.walls.cell
        rho = np.asarray(q[0]).ravel()[cell]
        temp = np.asarray(case.flow.gamma * q[Rows(grid.components).pressure] / q[0])
        mu = np.broadcast_to(np.asarray(op.viscosity(jnp.asarray(temp))), grid.shape).ravel()[cell]
        if op.distance is not None:
            distance = np.asarray(op.distance).ravel()[cell]
        else:
            from jaxrans.mesh.mesh import wall_distance
            distance = np.asarray(grid.pack(wall_distance(grid.mesh), 1.)).ravel()[cell]
        yplus = distance * np.sqrt(rho * np.linalg.norm(traction, axis=0)) / mu
        record["yplus_normal_cut"] = {"max": float(yplus.max()), "p95": float(np.percentile(yplus, 95))}
    np.savez_compressed(folder / "arrays.npz", primitive=primitive, arc=arc, weight=ds,
                        nodes=np.asarray(grid.mesh[0].nodes),
                        residual_history=result.history.residual, **surface)
    write_json(folder / "result.json", record)
    print(json.dumps({k: record[k] for k in ("reason", "orders_dropped", "iterations", "forces")}), flush=True)
    return record


def compare_limits(reference, sectional):
    r, s = reference, sectional
    if r["primitive"].shape != s["primitive"].shape:
        raise ValueError("limit fields have different shapes")
    # Compare the same physical wall faces, without interpolation.
    for key in ("x", "y"):
        rk = "x_streamwise" if key == "x" and "x_streamwise" in r else key
        sk = "x_streamwise" if key == "x" and "x_streamwise" in s else key
        np.testing.assert_allclose(r[rk], s[sk], rtol=1e-12, atol=1e-12)
    scale = np.maximum(1., np.max(abs(r["primitive"]), axis=(1, 2)))
    field = np.max(abs(s["primitive"] - r["primitive"]), axis=(1, 2)) / scale
    out = {"field_relative_linf_by_row": field.tolist()}
    for key in ("cp", "cf", "cf_z"):
        if key not in r and key not in s:
            continue
        delta = s.get(key, np.zeros_like(s["cp"])) - r.get(key, np.zeros_like(r["cp"]))
        out[key] = {"linf": float(np.max(abs(delta))), "l2": float(np.sqrt(np.average(delta**2, weights=r["weight"])))}
    return out


def surface_change(coarse, fine):
    """Interpolate along wall arc, keeping upper and lower branches separate."""
    out = {}
    for key in ("cp", "cf", "cf_z"):
        if key in coarse and key in fine:
            deltas, weights = [], []
            # The symmetric NACA 0012 has its LE at half the wall arc.
            for lo, hi in ((0., .5), (.5, 1.)):
                source = (coarse["arc"] >= lo) & (coarse["arc"] < hi)
                arc = coarse["arc"][source]
                mask = (fine["arc"] >= arc[0]) & (fine["arc"] <= arc[-1])
                deltas.append(fine[key][mask] - np.interp(fine["arc"][mask], arc, coarse[key][source]))
                weights.append(fine["weight"][mask])
            delta, weight = np.concatenate(deltas), np.concatenate(weights)
            out[key] = {"linf": float(abs(delta).max()),
                        "l2": float(np.sqrt(np.average(delta**2, weights=weight)))}
    return out


def richardson_summary(values):
    from validation.validate import richardson
    d1, d2 = values[1] - values[0], values[2] - values[1]
    if not (d1 * d2 > 0 and abs(d2) < abs(d1)):
        return {"order": None, "extrapolated": None, "monotone": d1 * d2 > 0,
                "differences": [d1, d2], "shrinking": False}
    return {**asdict(richardson(values)), "shrinking": True}


def report(jobs, output):
    records, arrays = {}, {}
    for job in jobs:
        path = output / job["id"]
        if (path / "result.json").exists():
            record = json.loads((path / "result.json").read_text())
            if record["job"] != job or record["source_sha256"] != source_fingerprint():
                raise ValueError(f"{job['id']}: stored inputs/source do not match this campaign")
            records[job["id"]] = record
            with np.load(path / "arrays.npz") as archive:
                arrays[job["id"]] = dict(archive)
    limits = []
    for job in jobs:
        if job["study"] != "limits" or job["formulation"] != "legacy":
            continue
        rid, sid = job["id"], job["id"].replace("_legacy", "_sectional")
        if rid not in records or sid not in records:
            limits.append({"reference": rid, "sectional": sid, "passed": False, "missing": True})
            continue
        diffs = compare_limits(arrays[rid], arrays[sid])
        forces = {k: abs(records[sid]["forces"][k] - records[rid]["forces"][k]) for k in COEFFICIENTS}
        converged = all(records[k]["reason"] == "converged" and records[k]["finite"] for k in (rid, sid))
        passed = converged and max(forces.values()) <= TOLERANCES["coefficient_abs"]
        passed &= max(diffs["field_relative_linf_by_row"]) <= TOLERANCES["field_relative"]
        passed &= all(v["linf"] <= TOLERANCES["cp_linf" if k == "cp" else "cf_linf"]
                      for k, v in diffs.items() if k in ("cp", "cf", "cf_z"))
        limits.append({"reference": rid, "sectional": sid, "passed": bool(passed),
                       "converged": converged, "force_absolute_differences": forces, **diffs})
    mesh = {}
    groups = {"global": ["mesh_sa_l3", "mesh_sa_l2", "mesh_sa_l1", "mesh_sa_l0"],
              "wall": ["mesh_sa_l2", "wall_sa_0.5", "wall_sa_0.25"],
              "outer": ["mesh_sa_l2", "outer_sa_2", "outer_sa_4"]}
    for group, ids in groups.items():
        if not any(j["study"] == "mesh" for j in jobs):
            break
        present = [i for i in ids if i in records]
        item = {"cases": ids, "complete": len(present) == len(ids),
                "converged": len(present) == len(ids) and all(records[i]["reason"] == "converged" and records[i]["finite"] for i in present),
                "changes": []}
        for a, b in zip(ids, ids[1:]):
            if a in records and b in records:
                item["changes"].append({"from": a, "to": b,
                    "force_differences": {k: records[b]["forces"][k] - records[a]["forces"][k] for k in COEFFICIENTS},
                    "surface": surface_change(arrays[a], arrays[b])})
        if group == "global" and item["converged"]:
            item["richardson"] = {k: richardson_summary(tuple(records[i]["forces"][k] for i in ids[-3:])) for k in ("cl", "cd", "cm", "cd_friction")}
        mesh[group] = item
    result = {"tolerances": TOLERANCES, "limits": limits, "mesh": mesh,
              "runs": records, "missing": [j["id"] for j in jobs if j["id"] not in records]}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("all", "limits", "mesh"), default="all")
    parser.add_argument("--physics", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--sweeps", type=float, nargs="+", default=(0., 30.))
    parser.add_argument("--out", type=Path, default=Path("results/sectional_d"))
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--only", nargs="+", help="run only these IDs; report still covers the full plan")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    jobs = plan(args.study, args.physics, args.sweeps, args.max_iters, args.tol)
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return 0
    if args.worker:
        job = json.loads(Path(args.worker).read_text())
        run_job(job, args.out)
        return 0
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.only and set(args.only) - {j["id"] for j in jobs}:
        parser.error("--only contains an ID outside the selected plan")
    if not args.resume and not args.report_only and any((output / j["id"] / "result.json").exists() for j in jobs if not args.only or j["id"] in args.only):
        parser.error("results already exist; use --resume or a new --out directory")
    write_json(output / "plan.json", {"jobs": jobs, "source_sha256": source_fingerprint()})
    if not args.report_only:
        for job in jobs:
            if args.only and job["id"] not in args.only:
                continue
            folder = output / job["id"]
            existing = folder / "result.json"
            if args.resume and existing.exists():
                cached = json.loads(existing.read_text())
                if cached["job"] != job or cached["source_sha256"] != source_fingerprint():
                    raise ValueError(f"{job['id']}: cached inputs/source differ; choose a new output directory")
                if cached["reason"] == "converged" and cached["finite"]:
                    continue
            folder.mkdir(exist_ok=True)
            write_json(folder / "job.json", job)
            with (folder / "run.log").open("w") as log:
                completed = subprocess.run([sys.executable, "-u", "-m", "validation.sectional_qualification",
                                            "--worker", str(folder / "job.json"), "--out", str(folder)],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            print(f"{job['id']}: process exit={completed.returncode}; {folder / 'run.log'}", flush=True)
            if completed.returncode:
                print((folder / "run.log").read_text()[-4000:], file=sys.stderr)
                return 1
    summary = report(jobs, output)
    return int(bool(summary["missing"]) or not all(r["passed"] for r in summary["limits"])
               or not all(r["converged"] for r in summary["mesh"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
