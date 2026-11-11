"""Batch compatible profiles, continuing each profile through its own alpha list.

    python -m jaxrans.campaign campaign.json --out results/campaign
    python -m jaxrans.campaign campaign.json --out results/campaign --resume

Topology and solver configuration are static within a batch; geometry is a
dynamic argument of the vmapped step. Angles advance in synchronized rounds.
Failed chains stop independently; finished entries are frozen between blocks.
Each profile may specify ``machs``: a list of freestream Mach numbers, each
run as a separate continuation in a directory named ``<name>_mach_<value>``.
If omitted, the Mach number from the case (or its overrides) is used.
Each completed angle is checkpointed with its fields and metadata. Resume
reuses saved successful points and retries the first unsuccessful point;
profiles without a complete saved point start from scratch.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import re
import shutil
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
from .simulation import Result, Timing, _march, make_stepper
from .convergence import convergence_record, equation_labels, diagnostics_report
from .solvers.driver import ConvergenceHistory, ExitReason, run_to_steady

SUCCESS = (ExitReason.CONVERGED, ExitReason.SETTLED)


@dataclass(frozen=True)
class ContinuationPolicy:
    """Opt-in bounded subdivision; the attempt budget applies to each target."""

    adaptive: bool = False
    min_step: float = 0.125
    max_attempts: int = 32
    on_failure: str = "stop"

    def __post_init__(self):
        if not isinstance(self.adaptive, bool):
            raise ValueError("continuation.adaptive must be a boolean")
        if (isinstance(self.min_step, bool) or not isinstance(self.min_step, (int, float))
                or not np.isfinite(self.min_step) or self.min_step <= 0):
            raise ValueError("continuation.min_step must be finite and positive")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("continuation.max_attempts must be a positive integer")
        if self.on_failure not in ("stop", "independent"):
            raise ValueError("continuation.on_failure must be 'stop' or 'independent'")
        if not self.adaptive and self.on_failure != "stop":
            raise ValueError("independent failure handling requires adaptive continuation")


@dataclass(frozen=True)
class PolarRequest:
    name: str
    case: Case
    alphas: Sequence[float]
    grid: Atlas | None = None
    initial: object | None = None
    continuation: ContinuationPolicy = ContinuationPolicy()


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


def _equation_norms(case: Case, template: Atlas, U, q, g):
    atlas = _with_geometry(template, g)
    return make_residual(atlas, case.flow, case.numerics, freestream=q,
                         distance=g["distance"]).equation_norms(U)


def _run_group(requests, grids, geometries, devices, progress, progress_with_residual, on_point=None,
               progress_with_coefficients=None, progress_with_equation_residuals=None) -> list[Polar]:
    case, template = requests[0].case, grids[0]
    diagnostics = case.numerics.diagnostics and case.numerics.solver == "implicit"
    step_options = {"diagnostics": True} if diagnostics else {}
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
            return make_stepper(case, atlas, freestream=f, distance=geom["distance"], **step_options)[0](s)

        advanced = jax.vmap(one)(states, q, g)
        return ((advanced[0], q, g), *advanced[1:])

    reference = case.grid_spec.force_reference()

    def coefficients(U, q, g):
        c = force_coefficients(U, _with_geometry(template, g), case.flow, reference, freestream=q)
        return jnp.stack([c["cl"], c["cd"], c["cm"]])

    monitor = jax.jit(jax.vmap(coefficients))

    equation_monitor = jax.jit(jax.vmap(lambda U, q, g: _equation_norms(case, template, U, q, g)))

    def detailed_norms(U, q, g):
        atlas = _with_geometry(template, g)
        operator = make_residual(atlas, case.flow, case.numerics, freestream=q, distance=g["distance"])
        return operator.equation_diagnostics(U)

    detailed_monitor = jax.jit(jax.vmap(detailed_norms)) if case.numerics.diagnostics else None
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
            starts = jnp.stack([jnp.asarray(requests[k].initial, dtype=case.numerics.dtype) for k in indices])
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
        equation_peak = np.asarray(equation_monitor(states.U, q, geometry))
        convergence_samples = []

        def on_block(b, carry, residuals):
            nonlocal equation_peak
            detailed_values = (jax.tree_util.tree_map(np.asarray, detailed_monitor(carry[0].U, carry[1], carry[2]))
                               if detailed_monitor is not None else None)
            equation_values = (detailed_values["l2_volume"] if detailed_values is not None
                               else np.asarray(equation_monitor(carry[0].U, carry[1], carry[2])))
            equation_peak = equation_values if equation_peak is None else np.maximum(equation_peak, equation_values)
            values = np.asarray(monitor(carry[0].U, carry[1], carry[2]))
            samples.append(values)
            if (progress is not None or progress_with_residual is not None or progress_with_coefficients is not None
                    or progress_with_equation_residuals is not None):
                for k, request in enumerate(requests):
                    if active[k]:
                        args = (request.name, float(angles[k]), (b + 1) * case.block_iters, float(values[k, 0]))
                        if progress is not None:
                            progress(*args)
                        if progress_with_residual is not None:
                            progress_with_residual(*args, float(residuals[-1, k]))
                        if progress_with_coefficients is not None:
                            progress_with_coefficients(*args, float(residuals[-1, k]),
                                                       float(values[k, 1]), float(values[k, 2]))
                        if progress_with_equation_residuals is not None:
                            progress_with_equation_residuals(
                                *args, float(values[k, 1]), float(values[k, 2]),
                                tuple(zip(equation_labels(len(equation_values[k]), template.components),
                                           np.asarray(equation_values[k], dtype=float))))
            histories = np.asarray(samples)
            records = [convergence_record(equation_values[k], equation_peak[k], histories[:, k], case.numerics,
                                             names=equation_labels(equation_values.shape[-1], template.components))
                       for k in range(len(indices))]
            if case.numerics.diagnostics:
                for k, row in enumerate(records):
                    if k < n and active[k]:
                        operator = make_residual(grids[k], requests[k].case.flow, case.numerics,
                                                 freestream=carry[1][k], distance=geometries[k]["distance"])
                        row["equation_diagnostics"] = diagnostics_report(
                            operator, {key: value[k] for key, value in detailed_values.items()})
            convergence_samples.append([dict(row, iteration=(b + 1) * case.block_iters) for row in records])
            return np.array([row["settled"] for row in records])

        carry, history = run_to_steady(
            step, (states, q, geometry), max_iters=case.numerics.max_iters, tol=0.0,
            block_iters=case.block_iters, devices=use, active_mask=active_padded, on_block=on_block,
            finite_check=lambda c: jnp.all(jnp.isfinite(c[0].U), axis=tuple(range(1, c[0].U.ndim))),
            allow_exact_zero=False, diagnostics=diagnostics,
            control_changed=(lambda a, b: (a[0].cfl != b[0].cfl) |
                             jnp.any(a[0].U != b[0].U, axis=tuple(range(1, a[0].U.ndim))))
                            if case.numerics.solver == "implicit" else None)
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
                            levels[k] if point == 0 else (),
                            coefficients_history=np.asarray(samples)[:n_blocks, k, :],
                            convergence_history=tuple(rows[k] for rows in convergence_samples[:n_blocks]))
            results[k].append(result)
            if on_point is not None:
                on_point(request.name, float(angles[k]), result)
            elapsed[k] += history.seconds
            alive[k] = h.reason in SUCCESS
    return [Polar(r.case, np.asarray(r.alphas[:len(rows)]), rows,
                  ContinuationTiming(float(elapsed[k]), sum(x.history.n_iters for x in rows),
                                     devices=used_devices),
                  mode="continuation", requested_alphas=tuple(r.alphas))
            for k, (r, rows) in enumerate(zip(requests, results))]


def _run_adaptive(request, grid, geometry, devices, progress, progress_with_residual,
                  on_point, on_attempt, saved=(), progress_with_coefficients=None,
                  progress_with_equation_residuals=None):
    """Replay a deterministic attempt log, then continue from its validated frontier."""
    policy = request.continuation
    attempts, rows = [], []
    validated = None
    for index, target in enumerate(request.alphas):
        candidate = float(target)
        target_result = None
        succeeded = False
        for retry in range(policy.max_attempts):
            source = attempts[validated] if validated is not None else None
            provenance = {
                "attempt": len(attempts), "requested_index": index, "requested_alpha": float(target),
                "alpha": candidate, "kind": "requested" if candidate == target else "intermediate",
                "initial_alpha": source.case.flow.alpha_deg if source is not None else None,
                "source_attempt": validated, "retry": retry,
            }
            if len(attempts) < len(saved):
                result = saved[len(attempts)]
                if result.provenance != provenance:
                    raise ValueError("saved adaptive attempt does not match the continuation schedule")
            else:
                one = dataclasses.replace(request, alphas=(candidate,),
                                          initial=source.U if source is not None else (request.initial if index == 0 else None),
                                          continuation=ContinuationPolicy())
                result = _run_group([one], [grid], [geometry], devices, progress, progress_with_residual,
                                    progress_with_coefficients=progress_with_coefficients,
                                    progress_with_equation_residuals=progress_with_equation_residuals)[0].results[0]
                result = dataclasses.replace(result, provenance=provenance)
                if on_attempt is not None:
                    on_attempt(request.name, candidate, result)
            attempts.append(result)
            if candidate == target:
                target_result = result
            if result.history.reason in SUCCESS:
                validated = len(attempts) - 1
                if candidate == target:
                    succeeded = True
                    break
                candidate = float(target)
            else:
                if source is None:
                    break
                midpoint = (float(source.case.flow.alpha_deg) + candidate) / 2
                if abs(midpoint - source.case.flow.alpha_deg) < policy.min_step or midpoint in (candidate, source.case.flow.alpha_deg):
                    break
                candidate = midpoint
        rows.append(target_result)
        if on_point is not None and len(attempts) > len(saved):
            on_point(request.name, float(target), target_result)
        if not succeeded:
            if policy.on_failure == "stop":
                break
            validated = None  # Later requested points start independently after exhaustion.
    if len(saved) > len(attempts):
        raise ValueError("saved adaptive attempt log extends beyond the requested schedule")
    return Polar(request.case, np.asarray(request.alphas[:len(rows)]), rows,
                 ContinuationTiming(sum(r.seconds for r in attempts), sum(r.history.n_iters for r in attempts),
                                    devices=1), mode="continuation", requested_alphas=tuple(request.alphas),
                 attempts=tuple(attempts), continuation=dataclasses.asdict(policy))


def run_campaign(requests: Sequence[PolarRequest], *, batch_size: int | None = None,
                 devices: int | None = None,
                 progress: Callable[[str, float, int, float], None] | None = None,
                 progress_with_residual: Callable[[str, float, int, float, float], None] | None = None,
                 on_point=None, on_group=None, on_attempt=None, progress_with_coefficients=None,
                 progress_with_equation_residuals=None,
                 continuation: ContinuationPolicy | None = None,
                 resume_attempts: dict[str, Sequence[Result]] | None = None) -> Campaign:
    """Continue profiles in compatible batches; preserve each requested alpha order.

    Different topology or physics forms a separate batch. Within a batch the
    geometry differs, but Mach, Reynolds and solver settings must agree.
    ``batch_size`` limits profiles per batch; devices split the profile axis.
    Progress receives ``(profile_name, alpha, iteration, CL)``.
    Progress_with_residual receives the same arguments followed by the last
    residual norm in the block. ``progress_with_coefficients`` receives those
    arguments followed by drag CD (Cx in wind axes) and pitching moment CM.
    ``progress_with_equation_residuals`` receives those coefficient arguments
    followed by normalized equation residuals.
    """
    requests = list(requests)
    if continuation is not None:
        requests = [dataclasses.replace(r, continuation=continuation) for r in requests]
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
        key = (_signature(request.case, grid, geometry), request.case.sequencing,
               request.name if request.continuation.adaptive else None)
        groups.setdefault(key, []).append((request, grid, geometry))
    polars, batches = {}, []
    for group in groups.values():
        size = batch_size or len(group)
        for start in range(0, len(group), size):
            chunk = group[start:start + size]
            reqs, grids, geometries = zip(*chunk)
            if on_group is not None:
                on_group(reqs)
            batches.append(tuple(r.name for r in reqs))
            if reqs[0].continuation.adaptive:
                computed = [_run_adaptive(reqs[0], grids[0], geometries[0], devices, progress,
                                          progress_with_residual, on_point, on_attempt,
                                          (resume_attempts or {}).get(reqs[0].name, ()),
                                          progress_with_coefficients, progress_with_equation_residuals)]
            else:
                computed = _run_group(reqs, grids, geometries, devices, progress, progress_with_residual, on_point,
                                      progress_with_coefficients, progress_with_equation_residuals)
            for request, polar in zip(reqs, computed):
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
        results.append(_load_point(request, grid, folder, alpha, point))
    return results, grid


def _load_point(request, grid, folder, alpha, point):
    def one(suffix):
        matches = list(folder.glob(f"*{suffix}"))
        if len(matches) != 1:
            raise ValueError(f"{folder}: expected one {suffix} file, found {len(matches)}")
        return matches[0]
    saved = json.loads(one("_metadata.json").read_text())
    old_numerics = saved.get("config", {}).get("numerics", {})
    with np.load(one("_fields.npz"), allow_pickle=False) as fields:
        if int(fields["n_blocks"]) != len(grid.mesh):
            raise ValueError(f"{folder}: saved mesh has a different number of blocks")
        blocks = []
        for i, mesh in enumerate(grid.mesh):
            for axis, nodes in zip("xyz", mesh.nodes):
                if not np.array_equal(fields[f"{axis}_node_{i}"], np.asarray(nodes)):
                    raise ValueError(f"{folder}: saved mesh differs from the requested mesh")
            blocks.append(jnp.asarray(fields[f"U_{i}"], dtype=request.case.numerics.dtype))
        U = grid.pack(blocks)
    if point["exit_reason"] in SUCCESS and not np.all(np.isfinite(np.asarray(U))):
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
    coefficients_paths = list(folder.glob("*_forces.dat"))
    coefficients = np.loadtxt(coefficients_paths[0], ndmin=2)[:, 1:] if coefficients_paths else np.empty((0, 3))
    solver_paths = list(folder.glob("*_solver_history.json"))
    solver = json.loads(solver_paths[0].read_text()) if solver_paths else {}
    history.block_diagnostics = tuple(solver.get("newton_blocks", ()))
    if point["exit_reason"] in SUCCESS:
        numerics = request.case.numerics
        names = equation_labels(U.shape[0], grid.components)
        records = solver.get("convergence", ())
        if records and coefficients.size:
            certified = convergence_record(records[-1]["current"], records[-1]["peak"],
                                           coefficients, numerics, names=names)["settled"]
        else:
            # Legacy files have no complete stopping evidence. Only reuse when
            # every old criterion was at least as strict as its replacement.
            old_force = old_numerics.get("forces_tol")
            old_relative = old_numerics.get("tol") if old_force is None else old_numerics.get("equation_tol")
            new_relative = numerics.tol if numerics.forces_tol is None else numerics.equation_tol
            old_overrides = dict(old_numerics.get("equation_abs_tols", ()))
            new_overrides = dict(numerics.equation_abs_tols)
            old_floor = old_numerics.get("equation_abs_tol")
            certified = ("equation_tol" in old_numerics and old_floor is not None
                         and old_relative is not None and old_relative <= new_relative
                         and all((0.0 if old_force is None else old_overrides.get(name, old_floor)) <=
                                 (0.0 if numerics.forces_tol is None else
                                  new_overrides.get(name, numerics.equation_abs_tol)) for name in names)
                         and (numerics.forces_tol is None or
                              old_force is not None and old_force <= numerics.forces_tol))
        if not certified or "equation_tol" not in old_numerics or "equation_abs_tol" not in old_numerics:
            raise ValueError(f"{folder}: saved point was not certified by the current equation tolerances "
                             "and force criteria; start a fresh campaign in a new output directory")
    return Result(case, grid, U, integrate_forces(U, grid, case.flow,
                  case.grid_spec.force_reference()), history, lift, timing,
                  coefficients_history=coefficients,
                  convergence_history=tuple(solver.get("convergence", ())),
                  provenance=extra.get("provenance", {}))


def _atomic_json(path, record):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, default=lambda x: np.asarray(x).tolist()))
    temporary.replace(path)


def _checkpoint_attempt(directory, result):
    """A renamed directory is the commit; partial staging directories are ignored."""
    attempts = directory / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    name = f"attempt_{result.provenance['attempt']:06d}"
    destination = attempts / name
    if destination.exists():
        saved = json.loads((destination / "attempt.json").read_text())
        metadata = list(destination.glob("*_metadata.json"))
        config = {"flow": dataclasses.asdict(result.case.flow),
                  "numerics": dataclasses.asdict(result.case.numerics),
                  "implicit": dataclasses.asdict(result.case.implicit)}
        config = json.loads(json.dumps(config))
        if (saved["provenance"] != result.provenance or saved["exit_reason"] != result.history.reason or
                saved["iterations"] != result.history.n_iters or len(metadata) != 1 or
                json.loads(metadata[0].read_text())["config"] != config):
            raise ValueError(f"{destination}: conflicting adaptive attempt")
        fields = list(destination.glob("*_fields.npz"))
        if len(fields) != 1:
            raise ValueError(f"{destination}: conflicting adaptive fields")
        with np.load(fields[0], allow_pickle=False) as archive:
            for k, (block, mesh) in enumerate(zip(result.grid.unpack(result.U), result.grid.mesh)):
                if not np.array_equal(archive[f"U_{k}"], np.asarray(block), equal_nan=True):
                    raise ValueError(f"{destination}: conflicting adaptive fields")
                for axis, nodes in zip("xyz", mesh.nodes):
                    if not np.array_equal(archive[f"{axis}_node_{k}"], np.asarray(nodes)):
                        raise ValueError(f"{destination}: conflicting adaptive mesh")
        return
    staging = attempts / ("." + name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    result.write(staging, extra={"polar_mode": "continuation", **result.provenance})
    _atomic_json(staging / "attempt.json", {
        "alpha": result.case.flow.alpha_deg, "iterations": result.history.n_iters,
        "exit_reason": result.history.reason, "provenance": result.provenance,
    })
    staging.rename(destination)


def _adaptive_resume(request, directory, resume):
    path = directory / "continuation.json"
    setup = {"version": 1, "case_setup": request.case.raw,
             "continuation": dataclasses.asdict(request.continuation),
             "requested_alphas": list(request.alphas)}
    if path.exists():
        if not resume:
            raise ValueError(f"{directory}: adaptive checkpoints exist; use --resume or a new output directory")
        old = json.loads(path.read_text())
        if (old.get("version") != 1 or old["case_setup"] != setup["case_setup"] or
                old["continuation"] != setup["continuation"] or
                old["requested_alphas"] != setup["requested_alphas"][:len(old["requested_alphas"])]):
            raise ValueError(f"{directory}: adaptive checkpoint configuration, policy or requested angles differ")
    elif resume and (directory / "polar_metadata.json").exists():
        raise ValueError(f"{directory}: legacy output cannot be resumed with adaptive continuation")
    directory.mkdir(parents=True, exist_ok=True)
    grid = request.grid if request.grid is not None else request.case.grid()
    saved = []
    for index, folder in enumerate(sorted((directory / "attempts").glob("attempt_*"))):
        record = json.loads((folder / "attempt.json").read_text())
        if record["provenance"]["attempt"] != index:
            raise ValueError(f"{folder}: adaptive checkpoint log has a gap")
        result = _load_point(request, grid, folder, record["alpha"], record)
        if result.provenance != record["provenance"]:
            raise ValueError(f"{folder}: attempt provenance differs from result metadata")
        saved.append(result)
    _atomic_json(path, setup)
    return dataclasses.replace(request, grid=grid), saved


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
            policy = ContinuationPolicy(**entry.get("continuation", doc.get("continuation", {})))
            if "machs" not in entry:
                requests.append(PolarRequest(entry["name"], load_case(path, overrides), angles, continuation=policy))
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
                requests.append(PolarRequest(f"{entry['name']}_mach_{label}", case, angles, continuation=policy))

        def progress(name, alpha, iteration, cl, cd, cm, equation_residuals):
            residuals = " ".join(f"R{label} {value:.3e}"
                                  for label, value in equation_residuals
                                  if label != "sa")
            print(f"{name} alpha {alpha:+g}: {iteration:8d} {residuals} "
                  f"CL {cl:+.6f} Cx(CD) {cd:+.6f} Cm {cm:+.6f}", flush=True)

        def announce(requests):
            for request in requests:
                print(f"{request.name}: Mach {request.case.flow.mach:g}, "
                      f"alpha {request.alphas[0]:+g} (preparation/compilation)", flush=True)

        names = [request.name for request in requests]
        if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in names):
            raise ValueError("profile names must be unique safe directory names")
        args.out.mkdir(parents=True, exist_ok=True)
        csv_path = args.out / "points.csv"
        columns = ["profile", "case", "mach", "alpha_deg", "reynolds", "gamma", "turbulence",
                   "cl", "cd", "cd_friction", "cm", "iterations", "status", "residual_mass_history",
                   "residual_mass", "residual_momentum_x", "residual_momentum_y", "residual_momentum_z",
                   "residual_energy", "residual_sa"]
        adaptive_names = {r.name for r in requests if r.continuation.adaptive}
        if adaptive_names:
            columns += ["attempt", "requested_index", "requested_alpha", "kind", "initial_alpha", "source_attempt", "retry"]
        if not args.resume or not csv_path.exists():
            with csv_path.open("w", newline="") as stream:
                csv.DictWriter(stream, fieldnames=columns).writeheader()

        request_by_name = {request.name: request for request in requests}
        checkpoint_points = {request.name: [] for request in requests}

        def save_point(name, alpha, result):
            if request_by_name[name].continuation.adaptive:
                return
            flow = result.case.flow
            norms = np.asarray(make_residual(result.grid, flow, result.case.numerics).equation_norms(result.U))
            labels = ["mass", *[f"momentum_{axis}" for axis in "xyz"[:result.grid.components]], "energy"]
            if len(norms) > len(labels):
                labels.append("sa")
            row = {"profile": name, "case": result.case.name, "mach": flow.mach, "alpha_deg": alpha,
                   "reynolds": flow.reynolds, "gamma": flow.gamma, "turbulence": flow.turbulence,
                   "cl": result.forces.cl, "cd": result.forces.cd, "cd_friction": result.forces.cd_friction,
                   "cm": result.forces.cm, "iterations": result.history.n_iters, "status": result.history.reason,
                   "residual_mass_history": result.history.residual[-1]}
            row.update({f"residual_{label}": float(value) for label, value in zip(labels, norms)})
            with csv_path.open("a", newline="") as stream:
                csv.DictWriter(stream, fieldnames=columns).writerow(row)

            # Persist each finished angle immediately. A process interruption can
            # then resume from the last successful point instead of losing the
            # fields held by the in-memory Campaign.
            request = request_by_name[name]
            points = checkpoint_points[name]
            matching = next((k for k, point in enumerate(points)
                             if float(point["alpha"]) == float(alpha)), None)
            if matching is None:
                matching = len(points)
                points.append({})
            point_name = (f"alpha_{alpha:+.2f}" if len(set(f"{a:+.2f}" for a in request.alphas))
                          == len(request.alphas) else
                          f"point_{matching:04d}_alpha_{alpha:+.8g}")
            profile_dir = args.out / name
            result.write(profile_dir / point_name, extra={
                "polar_mode": "continuation",
                "initial_alpha": float(request.alphas[matching - 1]) if matching else None,
            })
            points[matching] = {
                "alpha": float(alpha), "exit_reason": result.history.reason,
                "iterations": result.history.n_iters,
                "initial_alpha": float(request.alphas[matching - 1]) if matching else None,
                "cl": float(result.forces.cl), "cd": float(result.forces.cd),
                "cd_friction": float(result.forces.cd_friction), "cm": float(result.forces.cm),
                "orders_dropped": float(result.history.orders_dropped),
                "cl_spread": float(result.cl_spread),
                "settled": int(result.history.reason in SUCCESS),
            }
            profile_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = profile_dir / "polar_metadata.json"
            metadata_tmp = profile_dir / ".polar_metadata.json.tmp"
            metadata_tmp.write_text(json.dumps({
                "case": request.case.name, "case_setup": request.case.raw,
                "alphas": [point["alpha"] for point in points],
                "timing": result.timing.record(), "settled_cl_spread": 1e-4,
                "mode": "continuation", "requested_alphas": list(request.alphas),
                "points": points,
            }, indent=2, default=float))
            metadata_tmp.replace(metadata_path)

            table = np.asarray([[point["alpha"], point.get("cl", np.nan), point.get("cd", np.nan),
                                 point.get("cd_friction", np.nan), point.get("cm", np.nan),
                                 point.get("orders_dropped", np.nan), point["iterations"],
                                 point.get("cl_spread", np.nan), point.get("settled", 0)]
                                for point in points])
            np.savetxt(profile_dir / "polar.dat", table,
                       header="alpha_deg cl cd cd_friction cm orders_dropped iterations cl_spread settled",
                       fmt=["%.4f", "%.8f", "%.8f", "%.8f", "%.8f", "%.3f", "%d", "%.3e", "%d"])
            if not args.quiet:
                print(f"{name} alpha {alpha:+g}: {result.history.reason} after "
                      f"{result.history.n_iters} iterations (checkpoint saved)", flush=True)

        def save_attempt(name, alpha, result):
            _checkpoint_attempt(args.out / name, result)
            refresh_attempts()

        def refresh_attempts():
            if not adaptive_names:
                return
            with csv_path.open(newline="") as stream:
                rows = [row for row in csv.DictReader(stream) if row["profile"] not in adaptive_names]
            for name in sorted(adaptive_names):
                for folder in sorted((args.out / name / "attempts").glob("attempt_*")):
                    record = json.loads((folder / "attempt.json").read_text())
                    metadata = json.loads(next(folder.glob("*_metadata.json")).read_text())
                    extra, config = metadata["extra"], metadata["config"]["flow"]
                    residuals = np.loadtxt(next(folder.glob("*_residual.dat")), ndmin=2)
                    row = {"profile": name, "case": request_by_name[name].case.name,
                           "alpha_deg": record["alpha"], "iterations": record["iterations"],
                           "status": record["exit_reason"], "residual_mass_history": residuals[-1, 1],
                           **{k: config[k] for k in ("mach", "reynolds", "gamma", "turbulence")},
                           **{k: extra[k] for k in ("cl", "cd", "cm", "cd_friction")},
                           **{k: v for k, v in record["provenance"].items() if k in columns}}
                    row.update({f"residual_{key}": value["l2_volume"]
                                for key, value in extra["equation_residuals"].items()})
                    rows.append(row)
            temporary = csv_path.with_name(".points.csv.tmp")
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(csv_path)

        devices = args.devices if args.devices is not None else doc.get("devices")
        preserved = {}
        if args.resume:
            begun = time.perf_counter()
            polars, batches = {}, []
            for request in requests:
                directory = args.out / request.name
                if request.continuation.adaptive:
                    request, saved = _adaptive_resume(request, directory, True)
                    new = run_campaign([request], devices=devices,
                                       progress_with_equation_residuals=None if args.quiet else progress,
                                       on_attempt=save_attempt,
                                       resume_attempts={request.name: saved},
                                       on_group=None if args.quiet else announce)
                    polars.update(new.polars)
                    batches.extend(new.batches)
                    continue
                if (directory / "continuation.json").exists():
                    raise ValueError(f"{directory}: adaptive output requires its continuation policy")
                source = directory
                # Campaigns written before the machs option used the unsuffixed name.
                if not directory.is_dir() and "_mach_" in request.name:
                    legacy = args.out / request.name.rsplit("_mach_", 1)[0]
                    legacy_meta = legacy / "polar_metadata.json"
                    if (legacy_meta.is_file() and
                            json.loads(legacy_meta.read_text())["case_setup"]["case"]["freestream"]["mach"]
                            == request.case.flow.mach):
                        source = legacy
                if request.name in polars:
                    raise ValueError(f"duplicate resume output directory: {request.name}")
                if source.is_dir() and (source / "polar_metadata.json").is_file():
                    saved_meta = json.loads((source / "polar_metadata.json").read_text())
                    saved_points = [dict(point) for point in saved_meta.get("points", [])]
                    polar_path = source / "polar.dat"
                    if polar_path.is_file() and saved_points:
                        saved_table = np.loadtxt(polar_path, comments="#", ndmin=2)
                        for point, values in zip(saved_points, saved_table):
                            point.update(dict(zip(("alpha", "cl", "cd", "cd_friction", "cm",
                                                   "orders_dropped", "iterations", "cl_spread", "settled"),
                                                  values.tolist())))
                    checkpoint_points[request.name] = saved_points
                    prefix, grid = _resume_prefix(request, source)
                    if source != directory:
                        shutil.copytree(source, directory)
                else:
                    prefix, grid = [], request.case.grid()
                preserved[request.name] = len(prefix)
                remaining = tuple(request.alphas[len(prefix):])
                if remaining:
                    retry = dataclasses.replace(request, alphas=remaining, grid=grid,
                                                initial=prefix[-1].U if prefix else None)
                    new = run_campaign([retry], devices=devices,
                                       progress_with_equation_residuals=None if args.quiet else progress, on_point=save_point,
                                       on_group=None if args.quiet else announce)
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
            for request in requests:
                if request.continuation.adaptive:
                    _adaptive_resume(request, args.out / request.name, False)
                elif (args.out / request.name / "continuation.json").exists():
                    raise ValueError("adaptive output requires its continuation policy")
            campaign = run_campaign(requests, batch_size=args.batch_size if args.batch_size is not None else doc.get("batch_size"),
                                    devices=devices, progress_with_equation_residuals=None if args.quiet else progress,
                                    on_point=save_point, on_group=None if args.quiet else announce,
                                    on_attempt=save_attempt)
        refresh_attempts()
        out = campaign.write(args.out, preserve_prefix=preserved)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(campaign.summary())
    print(f"Results written to {out}")
    return 0 if campaign.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
