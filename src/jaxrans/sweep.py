"""A sweep: one case at many operating points (incidence, Mach), continued.

    python -m jaxrans.sweep CASE_DIR --alpha START STOP STEP [--mach M ...]
                            [--split N] [--cold] [--devices N]
                            [--set file.section.key=value ...] [--out DIR] [--quiet]

The points are arranged in **chains**. Along a chain each point starts from
the previous point's solution (continuation), the solver control (CFL,
residual reference, histories) starting afresh; a chain's first point, and a
point after a failed one, start from the freestream. ``--alpha`` with one
``--mach`` is one chain in alpha; several Mach numbers give one alpha chain
per Mach; a single angle with several Mach numbers is one chain in Mach.
``--split N`` cuts every chain into ``N`` contiguous chains (more of the GPU
busy, fewer continued points); ``--cold`` starts every point from the
freestream (no continuation).

Every chain occupies a **slot** of one batched executable: the incidence and
the Mach number enter only through the freestream vector and ``mu_inf =
M / Re``, both traced, so all points share one compilation. A slot whose
point stops (converged, settled, stalled, non-finite or out of iterations,
the tests of ``simulation.run``) is recorded at once and moves on between two
blocks, to its chain's next point or to a waiting chain; no slot waits for
the slowest point; once no chain is waiting and half the slots are idle, the
batch shrinks to the busy ones (one recompilation). Grid sequencing is not used: cold points start from the
freestream on the case's grid.

Continuation shortens easy points and leaves hard ones (separation) about as
they were. A continued point starts with its forces near their final value;
with the force-settling stop (``convergence.forces_tol``) a slow drift can
pass for settled, so check such points against a tighter ``forces_tol`` or a
``--cold`` run where it matters.

Output: ``<out>/sweep.dat`` (alpha, Mach, CL, CD, CD friction, CM, orders,
iterations, CL spread, settled, continued), ``<out>/sweep_metadata.json``
(case, points, chains, timing, provenance), and one ordinary result
directory per point, ``<out>/M0.7300_a+2.50/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .case import Case, CaseError, load_case
from .core.state import freestream_primitive
from .io.output import environment_record, git_revision
from .mesh.atlas import Atlas
from .numerics.residual import make_residual, swept_atlas
from .post.forces import force_coefficients, integrate_forces
from .simulation import SETTLED_CL_SPREAD, Result, Timing, cl_spread, make_stepper, warm_start
from .solvers.driver import BATCH_AXIS, ConvergenceHistory, ExitReason, _mesh, make_block

SUCCESS = (ExitReason.CONVERGED, ExitReason.SETTLED)


@dataclass(frozen=True)
class Point:
    """One operating point: incidence (degrees) and freestream Mach number."""

    alpha: float
    mach: float

    @property
    def name(self) -> str:
        return f"M{self.mach:.4f}_a{self.alpha:+.2f}"


def at_point(case: Case, point: Point) -> Case:
    """``case`` at ``point``; Reynolds number and temperature unchanged."""
    flow = dataclasses.replace(case.flow, alpha_deg=float(point.alpha), mach=float(point.mach))
    return dataclasses.replace(case, flow=flow, name=f"{case.name}_{point.name}")


def chains_of(alphas: Sequence[float], machs: Sequence[float], split: int = 1) -> list[list[Point]]:
    """The CLI's arrangement: one alpha chain per Mach number, or, for a
    single angle, one chain in Mach; each cut into ``split`` contiguous chains."""
    alphas, machs = [float(a) for a in alphas], [float(m) for m in machs]
    if not alphas or not machs:
        raise ValueError("a sweep needs at least one angle and one Mach number")
    if len(alphas) == 1:
        chains = [[Point(alphas[0], m) for m in machs]]
    else:
        chains = [[Point(a, m) for a in alphas] for m in machs]
    if split < 1:
        raise ValueError(f"split must be at least 1, got {split}")
    return [[chain[i] for i in part] for chain in chains
            for part in np.array_split(np.arange(len(chain)), min(split, len(chain)))]


@dataclass(frozen=True)
class SweepTiming:
    """Wall clock of the whole sweep; ``iterations`` counts every point's own."""

    total: float
    compile: float
    iterations: int
    slots: int
    devices: int

    def record(self) -> dict:
        return {"total_s": self.total, "first_block_s": self.compile, "point_iterations": self.iterations,
                "slots": self.slots, "devices": self.devices}

    def summary(self) -> str:
        return (f"{self.total:.1f} s ({self.compile:.1f} s first block, compilation included), "
                f"{self.iterations} point iterations in {self.slots} slot(s) on {self.devices} device(s)")


@dataclass(frozen=True)
class Sweep:
    """One ``Result`` per point, in the order of ``chains``; ``continued[k]``:
    whether point ``k`` started from the previous point of its chain."""

    case: Case
    chains: tuple[tuple[Point, ...], ...]
    points: tuple[Point, ...]
    results: tuple[Result, ...]
    continued: tuple[bool, ...]
    timing: SweepTiming

    def table(self) -> np.ndarray:
        """Rows ``alpha, Mach, CL, CD, CD friction, CM, orders, iterations, CL spread, settled, continued``."""
        return np.array([[p.alpha, p.mach, r.forces.cl, r.forces.cd, r.forces.cd_friction, r.forces.cm,
                          r.history.orders_dropped, r.history.n_iters, r.cl_spread,
                          float(r.history.reason in SUCCESS), float(c)]
                         for p, r, c in zip(self.points, self.results, self.continued)]).reshape(-1, 11)

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savetxt(directory / "sweep.dat", self.table(),
                   header="alpha Mach CL CD CD_friction CM orders iterations CL_spread settled continued",
                   fmt="%.10g")
        (directory / "sweep_metadata.json").write_text(json.dumps({
            "case": self.case.name, "case_setup": self.case.raw,
            "chains": [[dataclasses.asdict(p) for p in chain] for chain in self.chains],
            "points": [{**dataclasses.asdict(p), "exit_reason": r.history.reason,
                        "iterations": r.history.n_iters, "continued": c}
                       for p, r, c in zip(self.points, self.results, self.continued)],
            "timing": self.timing.record(), "settled_cl_spread": SETTLED_CL_SPREAD,
            "git": git_revision(), "environment": environment_record(),
        }, indent=2, default=float))
        for point, result, continued in zip(self.points, self.results, self.continued):
            result.write(directory / point.name, extra={"sweep_continued": continued})
        return directory


@dataclass
class _Running:
    """Host bookkeeping of the point a slot is computing."""

    chain: int
    position: int
    continued: bool
    began: float
    residuals: list = dataclasses.field(default_factory=list)
    coefficients: list = dataclasses.field(default_factory=list)
    first_block: float = float("nan")


def run_sweep(case: Case, chains: Sequence[Sequence[Point]], grid: Atlas | None = None, *,
              continuation: bool = True, slots: int | None = None, devices: int | None = None,
              progress: Callable[[Point, int, float], None] | None = None) -> Sweep:
    """``case`` at every point of ``chains``, each chain continued point to
    point (``continuation=False``: every point from the freestream), on at
    most ``slots`` batched slots (default: one per chain) split over
    ``devices`` (default all). ``progress(point, iteration, CL)`` per block
    and running point."""
    chains = tuple(tuple(chain) for chain in chains if len(chain))
    if not chains:
        raise ValueError("a sweep needs at least one point")
    grid = case.grid() if grid is None else grid
    numerics, flow = case.numerics, case.flow
    viscous = not flow.inviscid
    components = swept_atlas(grid, flow).components
    use = jax.devices()[: devices or len(jax.devices())]
    n_slots = min(slots or len(chains), len(chains))
    n_slots += -n_slots % len(use)                      # padding slots run copies, never recorded
    widest = n_slots
    block_iters = case.block_iters
    max_blocks = max(1, -(-numerics.max_iters // block_iters))
    distance = make_residual(grid, flow, numerics).distance
    reference = case.grid_spec.force_reference()

    def parameters(point: Point) -> tuple[np.ndarray, float]:
        sub = at_point(case, point).flow
        return np.asarray(freestream_primitive(sub, components)), (sub.mu_inf if viscous else 0.0)

    def stepper(q, mu):
        return make_stepper(case, grid, freestream=q, distance=distance, mu_inf=mu if viscous else None)

    def masked(carry):
        states, q, mu, active = carry
        advanced, residual = jax.vmap(lambda s, f, m: stepper(f, m)[0](s))(states, q, mu)
        keep = lambda new, old: jnp.where(active.reshape(active.shape + (1,) * (old.ndim - 1)), new, old)
        return (jax.tree_util.tree_map(keep, advanced, states), q, mu, active), residual

    block = make_block(masked, block_iters, use if len(use) > 1 else None)
    initialize = jax.jit(jax.vmap(lambda q, mu: stepper(q, mu)[1]))

    def coefficients(U, q, mu):
        c = force_coefficients(U, grid, flow, reference, freestream=q, mu_inf=mu if viscous else None)
        return jnp.stack([c["cl"], c["cd"], c["cm"]])

    monitor = jax.jit(jax.vmap(coefficients))
    put = (lambda tree: tree) if len(use) < 2 else (
        lambda tree: jax.device_put(tree, jax.sharding.NamedSharding(_mesh(use), jax.sharding.PartitionSpec(BATCH_AXIS))))

    pending = deque(range(len(chains)))
    running: list[_Running | None] = [None] * n_slots
    done: dict[tuple[int, int], tuple[Result, bool]] = {}
    q_host = np.zeros((n_slots, components + 2))
    mu_host = np.zeros(n_slots)
    states = None
    started = time.perf_counter()
    first_block = float("nan")

    def schedule(slots_to_fill, previous_ok):
        """Start the next point of every slot in ``slots_to_fill``; a slot
        continues its chain from its own field when ``previous_ok[s]``."""
        nonlocal states
        warm = {}
        for s in slots_to_fill:
            current = running[s]
            if current is not None and current.position + 1 < len(chains[current.chain]):
                chain, position = current.chain, current.position + 1
            elif pending:
                chain, position = pending.popleft(), 0
            else:
                running[s] = None
                continue
            continued = continuation and position > 0 and previous_ok.get(s, False)
            q_host[s], mu_host[s] = parameters(chains[chain][position])
            running[s] = _Running(chain, position, continued, time.perf_counter())
            if continued:
                warm[s] = True
        active_slots = [s for s in range(n_slots) if running[s] is not None]
        for s in range(n_slots):                         # idle slots: a valid copy for the kernels
            if running[s] is None and active_slots:
                q_host[s], mu_host[s] = q_host[active_slots[0]], mu_host[active_slots[0]]
        fresh = initialize(jnp.asarray(q_host, numerics.dtype), jnp.asarray(mu_host, numerics.dtype))
        if states is None:
            states = fresh
            return
        restart = np.zeros(n_slots, dtype=bool)
        restart[list(slots_to_fill)] = True
        continued_U = jax.vmap(lambda f, u: warm_start(f, u, grid))(fresh.U, states.U)
        use_warm = jnp.asarray([s in warm for s in range(n_slots)])
        fresh = fresh._replace(U=jnp.where(use_warm.reshape((-1,) + (1,) * (fresh.U.ndim - 1)),
                                           continued_U, fresh.U))
        mask = jnp.asarray(restart)
        states = jax.tree_util.tree_map(
            lambda f, o: jnp.where(mask.reshape(mask.shape + (1,) * (o.ndim - 1)), f, o), fresh, states)

    schedule(range(n_slots), {})
    while any(r is not None for r in running):
        active = np.array([r is not None for r in running])
        q, mu = jnp.asarray(q_host, numerics.dtype), jnp.asarray(mu_host, numerics.dtype)
        carry, residuals = block(put((states, q, mu, jnp.asarray(active))))
        states = carry[0]
        residuals = np.asarray(residuals).reshape(block_iters, n_slots)
        if np.isnan(first_block):
            first_block = time.perf_counter() - started
        forces = np.asarray(monitor(states.U, q, mu))
        finite = np.asarray(jnp.all(jnp.isfinite(states.U), axis=tuple(range(1, states.U.ndim))))
        finished, outcome = [], {}
        for s in np.flatnonzero(active):
            r = running[s]
            point = chains[r.chain][r.position]
            column = residuals[:, s]
            r.residuals.append(column)
            r.coefficients.append(forces[s])
            if np.isnan(r.first_block):
                r.first_block = time.perf_counter() - r.began
            if progress is not None:
                progress(point, len(r.residuals) * block_iters, float(forces[s, 0]))
            history = np.concatenate(r.residuals)
            peak = np.max(np.where(np.isfinite(history), history, 0.0))
            coefficients_so_far = np.asarray(r.coefficients)
            settled = numerics.forces_tol is not None and all(
                cl_spread(h) < numerics.forces_tol for h in coefficients_so_far.T)
            reason = (ExitReason.NON_FINITE if not (np.all(np.isfinite(column)) and finite[s])
                      else ExitReason.CONVERGED if peak > 0 and column[-1] <= numerics.tol * peak
                      else ExitReason.STALLED if len(column) > 1 and np.all(column == column[0])
                      else ExitReason.SETTLED if settled
                      else ExitReason.MAX_ITERS if len(r.residuals) >= max_blocks else None)
            if reason is None:
                continue
            seconds = time.perf_counter() - r.began
            h = ConvergenceHistory(history, reason, history.size, seconds, r.first_block)
            sub = at_point(case, point)
            U = states.U[s]
            result = Result(sub, grid, U, integrate_forces(U, grid, sub.flow, reference), h,
                            coefficients_so_far[:, 0],
                            Timing.of(h, block_iters, grid.cells, cases=n_slots, devices=len(use)))
            done[(r.chain, r.position)] = (result, r.continued)
            finished.append(s)
            outcome[s] = reason in SUCCESS
        if finished:
            schedule(finished, outcome)
            busy = [s for s in range(n_slots) if running[s] is not None]
            if not pending and busy and len(busy) <= n_slots // 2:
                # Idle slots still cost a full step each (the batch is vmapped):
                # drop them, at one recompilation per halving.
                keep = busy + [busy[0]] * (-len(busy) % len(use))
                states = jax.tree_util.tree_map(lambda leaf: leaf[np.asarray(keep)], states)
                q_host, mu_host = q_host[keep], mu_host[keep]
                running = [running[s] for s in busy] + [None] * (len(keep) - len(busy))
                n_slots = len(keep)

    order = [(c, p) for c in range(len(chains)) for p in range(len(chains[c]))]
    results = tuple(done[key][0] for key in order)
    timing = SweepTiming(time.perf_counter() - started, first_block,
                         sum(r.history.n_iters for r in results), widest, len(use))
    return Sweep(case, chains, tuple(chains[c][p] for c, p in order), results,
                 tuple(done[key][1] for key in order), timing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jaxrans.sweep", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", type=Path, help="case directory (case.json + numerical.json)")
    parser.add_argument("--alpha", type=float, nargs=3, metavar=("START", "STOP", "STEP"),
                        help="angles START..STOP (included) by STEP, degrees, in that order "
                             "(a negative STEP sweeps down); default: the case's")
    parser.add_argument("--mach", type=float, nargs="+", help="Mach numbers, in order; default: the case's")
    parser.add_argument("--split", type=int, default=1, help="cut every chain into this many (default 1)")
    parser.add_argument("--cold", action="store_true", help="every point from the freestream")
    parser.add_argument("--devices", type=int, help="devices to split the slots over (default: all)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="FILE.SECTION.KEY=VALUE", help="override one key; repeatable")
    parser.add_argument("--out", type=Path, help="output directory (default: <save_path>/<case_name>_sweep)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    try:
        case = load_case(args.case, tuple(args.overrides))
    except (CaseError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.alpha is None:
        alphas = [case.flow.alpha_deg]
    else:
        start, stop, step = args.alpha
        if step == 0 or (stop - start) * step < 0:
            parser.error("--alpha needs a nonzero STEP pointing from START to STOP")
        alphas = list(np.round(start + step * np.arange(int(np.floor((stop - start) / step + 1e-9)) + 1), 10))
    machs = args.mach or [case.flow.mach]
    try:
        chains = chains_of(alphas, machs, args.split)
    except ValueError as error:
        parser.error(str(error))
    print(f"{case.name}: {sum(map(len, chains))} points in {len(chains)} chain(s), "
          f"{'cold' if args.cold else 'continued'}, on {[str(d) for d in jax.devices()]}")

    def progress(point, iteration, cl):
        print(f"{point.name}  {iteration:7d}  CL {cl:+.5f}")

    sweep = run_sweep(case, chains, continuation=not args.cold, devices=args.devices,
                      progress=None if args.quiet else progress)
    out = sweep.write(args.out or case.output / f"{case.name}_sweep")
    print("   alpha    Mach        CL          CD   orders  iters  CL spread  exit")
    for point, result in zip(sweep.points, sweep.results):
        print(f"{point.alpha:+8.2f}  {point.mach:.4f}  {result.forces.cl:+.6f}  {result.forces.cd:+.7f}  "
              f"{result.history.orders_dropped:6.2f}  {result.history.n_iters:5d}  {result.cl_spread:9.1e}  "
              f"{result.history.reason}")
    print(f"{sweep.timing.summary()}\nwritten to {out}")
    return int(not all(r.history.reason in SUCCESS for r in sweep.results))


if __name__ == "__main__":
    raise SystemExit(main())
