"""Batch compatible profiles, continuing each profile through its own alpha list.

    python -m jaxrans.campaign campaign.json --out results/campaign
    python -m jaxrans.campaign campaign.json --out results/campaign --resume

Topology and solver configuration are static within a batch; geometry is a
dynamic argument of the vmapped step. Angles advance in synchronized rounds.
Failed chains stop independently; finished entries are frozen between blocks.
Each profile may specify ``machs``: a list of freestream Mach numbers, each
run as a separate continuation in a directory named ``<name>_mach_<value>``.
If omitted, the Mach number from the case (or its overrides) is used.
Resume reuses saved successful points and retries the first unsuccessful point;
profiles without saved results start from scratch. Results are written at the
end of the campaign, so an interrupted run cannot resume its unsaved progress.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .case import Case, load_case
from .core.state import freestream_primitive
from .mesh.atlas import Atlas
from .mesh.coarsen import prolong
from .numerics.residual import make_residual
from .polar import ContinuationTiming, Polar, alpha_range, at_alpha, validate_alphas
from .post.forces import force_coefficients, integrate_forces
from .simulation import Result, Timing, _march, cl_spread, make_stepper
from .solvers.driver import ConvergenceHistory, ExitReason, run_to_steady

SUCCESS = (ExitReason.CONVERGED, ExitReason.SETTLED)


@dataclass(frozen=True)
class PolarRequest:
    name: str
    case: Case
    alphas: Sequence[float]
    grid: Atlas | None = None
    initial: object | None = None


@dataclass(frozen=True)
class Campaign:
    polars: dict[str, Polar]
    batches: tuple[tuple[str, ...], ...]
    seconds: float

    @property
    def successful(self) -> bool:
        return all(len(p.results) == len(p.requested_alphas)
                   and all(r.history.reason in SUCCESS for r in p.results) for p in self.polars.values())

    def summary(self) -> str:
        """Human-readable calculation and polar statistics, including failed points."""
        lines = [f"Campaign: {'complete' if self.successful else 'incomplete'}; "
                 f"{len(self.polars)} profile(s), {len(self.batches)} batch(es), {self.seconds:.1f} s wall time"]
        for name, polar in self.polars.items():
            results = polar.results
            count = sum(r.history.reason in SUCCESS for r in results)
            lines.append(f"\n{name}: {len(results)}/{len(polar.requested_alphas)} angles calculated, "
                         f"{count} converged or settled; {polar.timing.iterations} case iterations, "
                         f"{polar.timing.total:.1f} s profile time")
            if not results:
                continue
            lines.append(f"  Mesh: {results[0].cells} cells; devices: {polar.timing.devices}")
            lines.append("  alpha (deg)        CL        CD        CM   iterations     residual  status")
            for alpha, result in zip(polar.alphas, results):
                residual = result.history.residual[-1] if len(result.history.residual) else float('nan')
                lines.append(f"  {alpha:10.3f} {result.forces.cl:9.5f} {result.forces.cd:9.5f} "
                             f"{result.forces.cm:9.5f} {result.history.n_iters:12d} "
                             f"{residual:12.3e}  {result.history.reason}")
            valid = [(float(a), r) for a, r in zip(polar.alphas, results)
                     if r.history.reason in SUCCESS and np.isfinite(r.forces.cl) and np.isfinite(r.forces.cd)]
            if valid:
                a_min, r_min = min(valid, key=lambda x: x[1].forces.cd)
                a_max, r_max = max(valid, key=lambda x: x[1].forces.cl)
                lines.append(f"  Converged points: maximum CL {r_max.forces.cl:.5f} at {a_max:+g} deg; "
                             f"minimum CD {r_min.forces.cd:.5f} at {a_min:+g} deg")
                positive_drag = [(a, r) for a, r in valid if r.forces.cd > 0]
                if positive_drag:
                    a_best, r_best = max(positive_drag, key=lambda x: x[1].forces.cl / x[1].forces.cd)
                    lines.append(f"  Maximum CL/CD: {r_best.forces.cl / r_best.forces.cd:.3f} at {a_best:+g} deg")
        return "\n".join(lines)

    def write(self, directory: str | Path, *, preserve_prefix: dict[str, int] | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, polar in self.polars.items():
            polar.write(directory / name, preserve_prefix=(preserve_prefix or {}).get(name, 0))
        (directory / "summary.txt").write_text(self.summary() + "\n")
        (directory / "campaign_metadata.json").write_text(json.dumps({
            "mode": "continuation", "seconds": self.seconds, "batches": self.batches,
            "successful": self.successful,
            "profiles": {name: {"requested_alphas": list(p.requested_alphas),
                                 "calculated_alphas": p.alphas.tolist(),
                                 "exit_reasons": [r.history.reason for r in p.results]}
                         for name, p in self.polars.items()},
        }, indent=2))
        return directory


def _geometry(atlas: Atlas, distance) -> dict:
    """Only geometry varies: all integer connectivity stays on the template."""
    g = atlas.grid
    return {"canvas": (g.nodes, g.centres, g.volume, g.faces), "centres": atlas.centres,
            "conditions": tuple(tuple((normal, midpoint) for _, _, normal, midpoint in s.conditions)
                                for s in atlas.stages),
            "quarter_chord": jnp.asarray(atlas.quarter_chord),
            "wall_nodes": (atlas.walls.faces.a, atlas.walls.faces.b),
            "distance": distance}


def _with_geometry(template: Atlas, geometry: dict) -> Atlas:
    nodes, centres, volume, faces = geometry["canvas"]
    stages = tuple(dataclasses.replace(s, conditions=tuple(
        (kind, source, normal, midpoint)
        for (kind, source, _, _), (normal, midpoint) in zip(s.conditions, values)))
        for s, values in zip(template.stages, geometry["conditions"]))
    a, b = geometry["wall_nodes"]
    return dataclasses.replace(
        template, grid=dataclasses.replace(template.grid, nodes=nodes, centres=centres, volume=volume, faces=faces),
        centres=geometry["centres"], stages=stages, quarter_chord=geometry["quarter_chord"],
        walls=dataclasses.replace(template.walls, faces=dataclasses.replace(template.walls.faces, a=a, b=b)))


def _signature(case: Case, atlas: Atlas, geometry: dict) -> tuple:
    """Conservative grouping: equal connectivity, configuration and array shapes.

    Mach/Re/model parameters currently form separate groups when different.
    This avoids accidentally using one profile's physics for another.
    """
    digest = hashlib.sha256()

    def add(array):
        a = np.asarray(array)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())

    for field in (atlas.interior, atlas.neighbours,
                  *(getattr(atlas.lines, f.name) for f in dataclasses.fields(atlas.lines)),
                  atlas.walls.cell, atlas.walls.family, atlas.walls.face, atlas.walls.into_fluid,
                  atlas.walls.faces.block, atlas.walls.faces.side, atlas.walls.faces.cell, atlas.walls.faces.body):
        add(field)
    kinds = []
    for stage in atlas.stages:
        add(stage.index)
        kinds.append(tuple(kind for kind, *_ in stage.conditions))
        for _, source, _, _ in stage.conditions:
            add(source)
    leaves, tree = jax.tree_util.tree_flatten(geometry)
    shapes = tuple((np.shape(a), np.asarray(a).dtype.str) for a in leaves)
    return (dataclasses.replace(case.flow, alpha_deg=0.0), case.numerics, case.implicit,
            case.rk_scheme, case.block_iters, case.grid_spec.force_reference(), atlas.g,
            repr((atlas.origins, atlas.windows)), tuple(kinds), digest.digest(), tree, shapes)


def _coarse_start(case: Case, grid: Atlas):
    """Optional grid sequencing is startup work, before the final-grid batch."""
    start, levels = None, []
    for level in range(case.sequencing, 0, -1):
        coarse = case.grid(coarsen=level)
        result = _march(case, coarse, None, start)
        levels.append({"cells": coarse.cells, "iterations": result.history.n_iters,
                       "seconds": result.seconds, "exit_reason": result.history.reason,
                       "cl": result.forces.cl, "cd": result.forces.cd})
        start = (result.U, coarse)
    if start is None:
        return None, ()
    default = make_stepper(case, grid)[1].U
    return prolong(*start, grid, default), tuple(levels)


def _run_group(requests, grids, geometries, devices, progress, progress_with_residual) -> list[Polar]:
    case, template = requests[0].case, grids[0]
    use = jax.devices()[:min(devices or len(jax.devices()), len(requests))]
    n = len(requests)
    indices = list(range(n)) + [n - 1] * (-n % len(use))
    geometry = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *(geometries[k] for k in indices))

    def initial(q, g):
        return make_stepper(case, _with_geometry(template, g), freestream=q, distance=g["distance"])[1]

    initialize = jax.jit(jax.vmap(initial))

    def step(carry):
        states, q, g = carry

        def one(s, f, geom):
            atlas = _with_geometry(template, geom)
            return make_stepper(case, atlas, freestream=f, distance=geom["distance"])[0](s)

        states, residual = jax.vmap(one)(states, q, g)
        return (states, q, g), residual

    reference = case.grid_spec.force_reference()

    def coefficients(U, q, g):
        c = force_coefficients(U, _with_geometry(template, g), case.flow, reference, freestream=q)
        return jnp.stack([c["cl"], c["cd"], c["cm"]])

    monitor = jax.jit(jax.vmap(coefficients))
    results = [[] for _ in requests]
    alive = np.ones(n, dtype=bool)
    previous = None
    levels = [() for _ in requests]
    elapsed = np.zeros(n)
    used_devices = len(use)
    for point in range(max(len(r.alphas) for r in requests)):
        active = np.array([alive[k] and point < len(r.alphas) for k, r in enumerate(requests)])
        if not active.any():
            break
        angles = [r.alphas[min(point, len(r.alphas) - 1)] for r in requests]
        q = jnp.asarray([freestream_primitive(at_alpha(requests[k].case, angles[k]).flow, template.components)
                         for k in indices], dtype=case.numerics.dtype)
        states = initialize(q, geometry)   # CFL and residual control are reset at every angle.
        if previous is not None:
            states = states._replace(U=jnp.where(template.interior, previous, states.U))
        elif any(request.initial is not None for request in requests):
            if any(request.initial is None for request in requests):
                raise ValueError("cannot batch resumed and fresh profiles together")
            starts = jnp.stack([jnp.asarray(requests[k].initial) for k in indices])
            if starts.shape != states.U.shape:
                raise ValueError("resume field shape does not match the grid")
            states = states._replace(U=jnp.where(template.interior, starts, states.U))
        elif case.sequencing:
            starts = []
            for k, request in enumerate(requests):
                begun = time.perf_counter()
                start, levels[k] = _coarse_start(at_alpha(request.case, angles[k]), grids[k])
                elapsed[k] += time.perf_counter() - begun
                starts.append(start)
            states = states._replace(U=jnp.stack([starts[k] for k in indices]))
        active_padded = np.r_[active, np.zeros(len(indices) - n, dtype=bool)]
        samples = []

        def on_block(b, carry, residuals):
            values = np.asarray(monitor(carry[0].U, carry[1], carry[2]))
            samples.append(values)
            if progress is not None or progress_with_residual is not None:
                for k, request in enumerate(requests):
                    if active[k]:
                        args = (request.name, float(angles[k]), (b + 1) * case.block_iters, float(values[k, 0]))
                        if progress is not None:
                            progress(*args)
                        if progress_with_residual is not None:
                            progress_with_residual(*args, float(residuals[-1, k]))
            if case.numerics.forces_tol is None:
                return False
            histories = np.asarray(samples)
            return np.array([all(cl_spread(histories[:, k, j]) < case.numerics.forces_tol for j in range(3))
                             for k in range(len(indices))])

        carry, history = run_to_steady(
            step, (states, q, geometry), max_iters=case.numerics.max_iters, tol=case.numerics.tol,
            block_iters=case.block_iters, devices=use, active_mask=active_padded, on_block=on_block,
            finite_check=lambda c: jnp.all(jnp.isfinite(c[0].U), axis=tuple(range(1, c[0].U.ndim))))
        previous = carry[0].U
        for k, request in enumerate(requests):
            if not active[k]:
                continue
            sub = at_alpha(request.case, angles[k])
            h = history.case(k)
            U = previous[k]
            n_blocks = h.n_iters // case.block_iters
            result = Result(sub, grids[k], U, integrate_forces(U, grids[k], sub.flow,
                            sub.grid_spec.force_reference()), h, np.asarray(samples)[:n_blocks, k, 0],
                            Timing.of(h, case.block_iters, grids[k].cells, devices=used_devices),
                            levels[k] if point == 0 else ())
            results[k].append(result)
            elapsed[k] += history.seconds
            alive[k] = h.reason in SUCCESS
    return [Polar(r.case, np.asarray(r.alphas[:len(rows)]), rows,
                  ContinuationTiming(float(elapsed[k]), sum(x.history.n_iters for x in rows),
                                     devices=used_devices),
                  mode="continuation", requested_alphas=tuple(r.alphas))
            for k, (r, rows) in enumerate(zip(requests, results))]


def run_campaign(requests: Sequence[PolarRequest], *, batch_size: int | None = None,
                 devices: int | None = None,
                 progress: Callable[[str, float, int, float], None] | None = None,
                 progress_with_residual: Callable[[str, float, int, float, float], None] | None = None) -> Campaign:
    """Continue profiles in compatible batches; preserve each requested alpha order.

    Different topology or physics forms a separate batch. Within a batch the
    geometry differs, but Mach, Reynolds and solver settings must agree.
    ``batch_size`` limits profiles per batch; devices split the profile axis.
    Progress receives ``(profile_name, alpha, iteration, CL)``.
    Progress_with_residual receives the same arguments followed by the last
    residual norm in the block.
    """
    requests = list(requests)
    if not requests:
        raise ValueError("a campaign needs at least one profile")
    names = [r.name for r in requests]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names):
        raise ValueError("profile names must be unique safe directory names")
    if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    if devices is not None and (not isinstance(devices, int) or not 1 <= devices <= len(jax.devices())):
        raise ValueError(f"devices must be between 1 and {len(jax.devices())}")
    begun = time.perf_counter()
    groups = {}
    for request in requests:
        request = dataclasses.replace(request, alphas=tuple(validate_alphas(request.alphas)))
        grid = request.case.grid() if request.grid is None else request.grid
        distance = make_residual(grid, request.case.flow, request.case.numerics).distance
        geometry = _geometry(grid, distance)
        key = (_signature(request.case, grid, geometry), request.case.sequencing)
        groups.setdefault(key, []).append((request, grid, geometry))
    polars, batches = {}, []
    for group in groups.values():
        size = batch_size or len(group)
        for start in range(0, len(group), size):
            chunk = group[start:start + size]
            reqs, grids, geometries = zip(*chunk)
            batches.append(tuple(r.name for r in reqs))
            for request, polar in zip(reqs, _run_group(reqs, grids, geometries, devices, progress, progress_with_residual)):
                polars[request.name] = polar
    return Campaign({name: polars[name] for name in names}, tuple(batches), time.perf_counter() - begun)


def _resume_prefix(request: PolarRequest, directory: Path) -> tuple[list[Result], Atlas]:
    """Read the consecutive successful points and validate the saved configuration and mesh."""
    meta = json.loads((directory / "polar_metadata.json").read_text())
    old_setup = json.loads(json.dumps(meta["case_setup"]))
    new_setup = json.loads(json.dumps(request.case.raw))
    # Iteration limits and convergence tolerances may be changed for the retry.
    old_setup["numerical"].pop("convergence", None)
    new_setup["numerical"].pop("convergence", None)
    if old_setup != new_setup:
        raise ValueError(f"{directory}: saved case configuration differs from the requested case")
    points = meta["points"]
    old_alphas = meta["alphas"]
    if len(points) != len(old_alphas) or old_alphas != list(request.alphas[:len(old_alphas)]):
        raise ValueError(f"{directory}: saved angles do not match the requested sequence")
    grid = request.case.grid()
    results = []
    names = [f"alpha_{a:+.2f}" for a in old_alphas]
    unique = len(set(names)) == len(names)
    for k, (alpha, point) in enumerate(zip(old_alphas, points)):
        if point["exit_reason"] not in SUCCESS:
            break
        folder = directory / (names[k] if unique else f"point_{k:04d}_alpha_{alpha:+.8g}")
        def one(suffix):
            matches = list(folder.glob(f"*{suffix}"))
            if len(matches) != 1:
                raise ValueError(f"{folder}: expected one {suffix} file, found {len(matches)}")
            return matches[0]
        saved = json.loads(one("_metadata.json").read_text())
        with np.load(one("_fields.npz"), allow_pickle=False) as fields:
            if int(fields["n_blocks"]) != len(grid.mesh):
                raise ValueError(f"{folder}: saved mesh has a different number of blocks")
            blocks = []
            for i, mesh in enumerate(grid.mesh):
                for axis, nodes in zip("xyz", mesh.nodes):
                    if not np.array_equal(fields[f"{axis}_node_{i}"], np.asarray(nodes)):
                        raise ValueError(f"{folder}: saved mesh differs from the requested mesh")
                blocks.append(jnp.asarray(fields[f"U_{i}"]))
            U = grid.pack(blocks)
        if not np.all(np.isfinite(np.asarray(U))):
            raise ValueError(f"{folder}: saved field contains non-finite values")
        residual = np.loadtxt(one("_residual.dat"), ndmin=2)[:, 1]
        lift = np.loadtxt(one("_lift.dat"), ndmin=2)[:, 1]
        extra = saved["extra"]
        timing_data = extra["timing"]
        timing = Timing(timing_data["total_s"], timing_data["compile_s"] +
                        request.case.block_iters * timing_data["per_iteration_s"],
                        point["iterations"], request.case.block_iters, grid.cells,
                        cases=timing_data["cases"], devices=timing_data["devices"])
        history = ConvergenceHistory(residual, point["exit_reason"], point["iterations"], timing.total,
                                     timing.first_block)
        case = at_alpha(request.case, alpha)
        results.append(Result(case, grid, U, integrate_forces(U, grid, case.flow,
                              case.grid_spec.force_reference()), history, lift, timing))
    return results, grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume from saved successful points in --out")
    args = parser.parse_args(argv)
    try:
        doc = json.loads(args.manifest.read_text())
        requests = []
        for entry in doc["profiles"]:
            path = args.manifest.resolve().parent / entry["case"]
            angles = entry["alphas"] if "alphas" in entry else alpha_range(*entry["alpha"])
            overrides = tuple(entry.get("overrides", ()))
            if "machs" not in entry:
                requests.append(PolarRequest(entry["name"], load_case(path, overrides), angles))
                continue
            machs = entry["machs"]
            if (not isinstance(machs, list) or not machs or
                    any(isinstance(m, bool) or not isinstance(m, (int, float)) or
                        not np.isfinite(m) or m <= 0 for m in machs)):
                raise ValueError("machs must be a non-empty list of finite positive numbers")
            if len(set(machs)) != len(machs):
                raise ValueError("machs must not contain duplicates")
            for mach in machs:
                label = repr(float(mach)).replace("+", "p").replace("-", "m")
                case = load_case(path, overrides + (f"case.freestream.mach={float(mach)!r}",))
                requests.append(PolarRequest(f"{entry['name']}_mach_{label}", case, angles))

        def progress(name, alpha, iteration, cl, residual):
            print(f"{name} alpha {alpha:+g}: {iteration:8d} residual {residual:.6e} CL {cl:+.6f}", flush=True)

        devices = args.devices if args.devices is not None else doc.get("devices")
        preserved = {}
        if args.resume:
            begun = time.perf_counter()
            polars, batches = {}, []
            for request in requests:
                directory = args.out / request.name
                # Campaigns written before the machs option used the unsuffixed name.
                if not directory.is_dir() and "_mach_" in request.name:
                    legacy = args.out / request.name.rsplit("_mach_", 1)[0]
                    legacy_meta = legacy / "polar_metadata.json"
                    if (legacy_meta.is_file() and
                            json.loads(legacy_meta.read_text())["case_setup"]["case"]["freestream"]["mach"]
                            == request.case.flow.mach):
                        directory = legacy
                        request = dataclasses.replace(request, name=legacy.name)
                if request.name in polars:
                    raise ValueError(f"duplicate resume output directory: {request.name}")
                if directory.exists():
                    prefix, grid = _resume_prefix(request, directory)
                else:
                    prefix, grid = [], request.case.grid()
                preserved[request.name] = len(prefix)
                remaining = tuple(request.alphas[len(prefix):])
                if remaining:
                    retry = dataclasses.replace(request, alphas=remaining, grid=grid,
                                                initial=prefix[-1].U if prefix else None)
                    new = run_campaign([retry], devices=devices,
                                       progress_with_residual=None if args.quiet else progress)
                    tail = new.polars[request.name]
                    batches.extend(new.batches)
                    rows = prefix + tail.results
                    elapsed = sum(r.seconds for r in prefix) + tail.timing.total
                else:
                    rows = prefix
                    elapsed = sum(r.seconds for r in prefix)
                timing = ContinuationTiming(elapsed, sum(r.history.n_iters for r in rows),
                                            devices=devices or 1)
                polars[request.name] = Polar(request.case, np.asarray(request.alphas[:len(rows)]), rows,
                                             timing, mode="continuation", requested_alphas=tuple(request.alphas))
            campaign = Campaign(polars, tuple(batches), time.perf_counter() - begun)
        else:
            campaign = run_campaign(requests, batch_size=args.batch_size if args.batch_size is not None else doc.get("batch_size"),
                                    devices=devices, progress_with_residual=None if args.quiet else progress)
        out = campaign.write(args.out, preserve_prefix=preserved)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(campaign.summary())
    print(f"Results written to {out}")
    return 0 if campaign.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
