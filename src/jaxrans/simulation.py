"""Run a case: build the operator, march to steady state, write the result.

    from jaxrans import load_case, run

    case = load_case("examples/naca0012_euler")
    result = run(case)              # progress=None: silent
    result.write()                  # -> <case output>/<case name>/

``run`` composes the residual (discretization), the case's pseudo-time path
(solvers) and the shared device driver; it knows nothing about reference data
or validation, which live in ``validation/``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .convergence import convergence_record, diagnostics_report, equations_settled as device_equations_settled
from .convergence import cl_spread as cl_spread
from .convergence import equations_settled as _equations_settled
from .core.state import conservative_to_primitive
from .mesh.atlas import Atlas
from .mesh.coarsen import prolong
from .mesh.quality import quality
from .mesh.sectional import SectionalAtlas
from .numerics.residual import make_residual
from .numerics.slabs import SlabAtlas
from .case import Case
from .post.forces import Forces, force_coefficients, forces_from_coefficients, surface, surface_zones
from .io.output import ResultWriter
from .solvers import explicit, implicit
from .solvers.driver import ConvergenceHistory, ExitReason, make_block, run_to_steady

Progress = Callable[[int, float, float, Forces], None]
"""``(iteration, residual, orders dropped, forces) -> None``, once per block."""


SETTLED_CL_SPREAD = 1e-4
"""Largest CL range over the last quarter of a run for it to count as settled:
most RANS runs stall at round-off (9.8-10.8 orders) before ``tol``."""


def equations_settled(current, peak, numerics, *, names=None) -> bool:
    """Backward-compatible host decision using the common convergence policy."""
    return bool(_equations_settled(current, peak, numerics, names=names))


@dataclass(frozen=True)
class Timing:
    """Wall clock of a run; the first block carries the compilation, the later
    ones give the steady cost. ``cases``: advanced together (a polar's angles)."""

    total: float
    first_block: float
    iterations: int
    block_iters: int
    cells: int
    cases: int = 1
    devices: int = 1

    @classmethod
    def of(cls, history: ConvergenceHistory, block_iters: int, cells: int, **batch) -> "Timing":
        return cls(history.seconds, history.first_block_seconds, history.n_iters, block_iters,
                   cells, **batch)

    @property
    def per_iteration(self) -> float:
        """Steady seconds per iteration (of the whole batch), from every block
        after the first; NaN for a single-block run."""
        later = self.iterations - self.block_iters
        return (self.total - self.first_block) / later if later > 0 else float("nan")

    @property
    def compile(self) -> float:
        """The first block minus its iterations at the steady rate."""
        return self.first_block - self.block_iters * self.per_iteration

    @property
    def case_iterations_per_second(self) -> float:
        return self.cases / self.per_iteration

    def record(self) -> dict[str, float]:
        return {"total_s": self.total, "compile_s": self.compile,
                "per_iteration_s": self.per_iteration,
                "case_iterations_per_s": self.case_iterations_per_second,
                "ns_per_cell_iteration": 1e9 / (self.case_iterations_per_second * self.cells),
                "iterations": self.iterations, "cells": self.cells,
                "cases": self.cases, "devices": self.devices}

    def summary(self) -> str:
        if not self.per_iteration == self.per_iteration:          # one block: no steady rate
            return f"{self.total:.1f} s (one block, compilation included)"
        return (f"{self.total:.1f} s, compile ~{self.compile:.1f} s, "
                f"{1e3 * self.per_iteration:.2f} ms/iteration"
                + (f" for {self.cases} cases on {self.devices} device(s)" if self.cases > 1
                   else f" on {self.devices} devices" if self.devices > 1 else ""))


@dataclass(frozen=True)
class Result:
    """A run: ``U`` on the atlas canvas, ``(nvar, NI, NJ)`` (``grid.unpack`` gives
    the blocks), and the lift at the end of every block, ``cl_history``;
    ``levels``, the coarser grids solved first (grid sequencing), coarsest first."""

    case: Case
    grid: Atlas
    U: Array
    forces: Forces
    history: ConvergenceHistory
    cl_history: np.ndarray
    timing: Timing
    levels: tuple[dict, ...] = ()
    coefficients_history: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    convergence_history: tuple[dict, ...] = ()
    provenance: dict = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return self.timing.total

    @property
    def cl_spread(self) -> float:
        return cl_spread(self.cl_history)

    @property
    def cells(self) -> int:
        return self.grid.cells

    def surface(self) -> dict[str, np.ndarray]:
        """Wall distributions per face: ``x``, ``y``, ``body``, ``cp`` [, ``cf``]."""
        return surface(self.U, self.grid, self.case.flow)

    def write(self, directory: str | Path | None = None, extra: dict | None = None) -> Path:
        """Write fields, wall distributions, residual and lift histories, and
        the resolved case with provenance, to ``directory`` (default:
        ``<case output>/<case name>``). Returns the directory."""
        directory = Path(directory) if directory is not None else self.case.output / self.case.name
        writer = ResultWriter(directory, case_name=self.case.name)
        operator = make_residual(self.grid, self.case.flow, self.case.numerics)
        equation_residuals = diagnostics_report(operator, operator.equation_diagnostics(self.U))
        writer.write_metadata(
            {"flow": self.case.flow, "numerics": self.case.numerics, "implicit": self.case.implicit},
            extra={
                "case_setup": self.case.raw,
                "grid": self.case.grid_spec.describe(self.case.directory),
                "cells": self.cells,
                "field_dtype": str(self.U.dtype),
                "precision_policy": {"state": str(self.U.dtype), "residual": str(self.U.dtype),
                                     "jacobian": "float32" if self.case.numerics.mixed_precision else str(self.U.dtype),
                                     "krylov_basis": "float32" if self.case.numerics.mixed_precision else str(self.U.dtype),
                                     "preconditioner": "float32" if self.case.numerics.mixed_precision else str(self.U.dtype)},
                "grid_quality": quality(self.grid.mesh).record(),
                "equation_residuals": equation_residuals,
                "bodies": [dict(zip(("cl", "cd", "cm"), b)) for b in self.forces.bodies],
                "cl": self.forces.cl, "cd": self.forces.cd,
                "cd_friction": self.forces.cd_friction, "cm": self.forces.cm,
                "exit_reason": self.history.reason,
                "orders_dropped": self.history.orders_dropped,
                "iterations": self.history.n_iters,
                "seconds": self.seconds,
                "cl_spread": self.cl_spread,
                "timing": self.timing.record(),
                **({"sectional": {**asdict(self.grid.spec), "depth": self.grid.depth,
                                   "axes": ["normal_chord", "up", "sweep"],
                                   "reference_sweep_deg": self.grid.spec.reference_sweep_deg,
                                   "relaxation_start": self.grid.relaxation_start,
                                   "relaxation_end": self.grid.relaxation_end,
                                   "primitive_closure": "cartesian_copy_fixed_reference_viscosity",
                                   **({"streamwise_chord": self.grid.geometry.chord,
                                       "relaxation_law": "quintic_wall_distance",
                                       "sa_wall_distance": "positive_conical_sectors"}
                                      if self.grid.geometry is not None else {})}}
                   if isinstance(self.grid, SectionalAtlas) else {}),
                "convergence": self.convergence_history[-1] if self.convergence_history else {},
                "provenance": self.provenance,
                **({"grid_sequencing": list(self.levels)} if self.levels else {}),
                **(extra or {}),
            },
        )
        writer.write_residual_history(self.history.residual)
        writer.write_lift_history(self.cl_history, self.case.block_iters)
        writer.write_solver_history(self.coefficients_history, self.convergence_history,
                                    self.history.block_diagnostics, self.case.block_iters)
        wall = self.surface()
        writer.write_surface(wall)
        if "z" in wall and wall["z"].size:
            writer.write_surface_tecplot(wall, surface_zones(self.grid))
        blocks = [np.asarray(u) for u in self.grid.unpack(self.U)]
        writer.write_fields(blocks, [np.asarray(b.nodes) for b in self.grid.mesh])
        if self.grid.dim == 3:
            for k, (U, b) in enumerate(zip(blocks, self.grid.mesh)):
                writer.write_vts(np.asarray(b.nodes), cell_fields(U, self.case.flow.gamma), k)
        return directory


def cell_fields(U: np.ndarray, gamma: float) -> dict[str, np.ndarray]:
    """Density, velocity, pressure and Mach number of a 3D block, for ParaView."""
    rho, velocity, p = conservative_to_primitive(np.asarray(U), gamma, 3)
    mach = np.linalg.norm(velocity, axis=0) / np.sqrt(gamma * p / rho)
    return {"density": rho, "velocity": velocity, "pressure": p, "mach": mach}


def make_stepper(case: Case, grid: Atlas, freestream: Array | None = None,
                 distance: Array | None = None, initial: Array | None = None,
                 *, diagnostics: bool = False, mu_inf: Array | None = None) -> tuple[Callable, object]:
    """``(step, initial state)`` of the case's pseudo-time path; ``freestream``
    (``(4,)`` primitives, possibly traced) is how a polar maps it over angles,
    and ``mu_inf`` (``M / Re``, possibly traced) how a campaign batches Mach
    and Reynolds numbers."""
    flow, numerics = case.flow, case.numerics
    operator = make_residual(grid, flow, numerics, freestream=freestream, distance=distance, mu_inf=mu_inf)
    if numerics.solver == "implicit":
        jacobian = None if case.implicit.jacobian == "exact" else make_residual(
            grid, flow, numerics, first_order=True, freestream=freestream, distance=operator.distance,
            mu_inf=mu_inf)
        step = implicit.make_newton_step(operator, case.implicit, jacobian, diagnostics=diagnostics,
                                        mixed_precision=numerics.mixed_precision)
        state = implicit.initial_state(operator, case.implicit, dtype=numerics.dtype)
    else:
        step = explicit.make_rk_step(operator, scheme=case.rk_scheme)
        state = explicit.initial_state(operator, dtype=numerics.dtype)
    if initial is not None:
        initial = jnp.asarray(initial, dtype=numerics.dtype)
        if initial.shape != state.U.shape:
            raise ValueError(f"initial field must have shape {state.U.shape}, got {initial.shape}")
        # Only physical cells are inherited; ghosts and gaps belong to the new freestream.
        state = state._replace(U=jnp.where(grid.interior, initial, state.U))
    return step, state


def run(case: Case, grid: Atlas | None = None, progress: Progress | None = None,
        on_level: Callable[[int, Atlas], None] | None = None, devices: Sequence | None = None,
        *, initial: Array | None = None, device_convergence: bool = False) -> Result:
    """March ``case`` to steady state on ``grid`` (default: ``case.grid()``),
    after its ``sequencing`` coarser grids, each started from the one before;
    ``on_level(level, grid)`` announces each grid (0: the case's own).
    ``devices`` (two or more) split the state in slabs along the canvas's
    first axis, one per device (``numerics/slabs.py``). Only the case's own
    grid is split; its coarser levels, small, run on one device.
    ``initial`` restarts on this exact grid, copying its interior field,
    resetting solver control, and bypassing grid sequencing. With
    ``device_convergence`` the block-wise stopping tests and histories stay on
    one device until the solve stops. A progress callback requires host polling."""
    grid = case.grid() if grid is None else grid
    devices = tuple(devices) if devices is not None and len(devices) > 1 else None
    if device_convergence and (progress is not None or devices is not None):
        raise ValueError("device convergence requires a single device and no progress callback")
    if initial is not None:
        # A restart is already on the final grid: do not repeat grid sequencing.
        return _march(case, grid, progress, None, devices, initial=initial, device_convergence=device_convergence)
    levels, start = [], None
    for level in range(case.sequencing, 0, -1):
        coarse = case.grid(coarsen=level)
        if on_level is not None:
            on_level(level, coarse)
        r = _march(case, coarse, progress, start, device_convergence=device_convergence)
        levels.append({"cells": coarse.cells, "iterations": r.history.n_iters, "seconds": r.seconds,
                       "exit_reason": r.history.reason, "cl": r.forces.cl, "cd": r.forces.cd})
        start = (r.U, coarse)
    if levels and on_level is not None:
        on_level(0, grid)
    return replace(_march(case, grid, progress, start, devices, device_convergence=device_convergence),
                   levels=tuple(levels))


def _march_on_device(step, state, monitor, on_block, equation_peak, names, case, diagnostics):
    """Advance blocks and decide when to stop without transferring history mid-run."""
    numerics = case.numerics
    block_iters = case.block_iters
    max_blocks = max(1, -(-numerics.max_iters // block_iters))
    block = make_block(step, block_iters, diagnostics=diagnostics)
    _, block_output = jax.eval_shape(block, state)
    _, _, measurements = jax.eval_shape(monitor, state, state)
    reasons = (ExitReason.MAX_ITERS, ExitReason.NON_FINITE, ExitReason.CONVERGED,
               ExitReason.STALLED, ExitReason.SETTLED)

    @jax.jit
    def solve(initial, initial_peak):
        def empty(spec):
            return jnp.zeros((max_blocks,) + spec.shape, dtype=spec.dtype)

        block_history, measured_history = jax.tree_util.tree_map(empty, (block_output, measurements))
        coeff_columns = jnp.arange(max_blocks)[:, None]

        def advance(carry):
            i, _, old, eq_peak, mass_peak, blocks, measured = carry
            current, output = block(old)
            finite, changed, values = monitor(old, current)
            blocks = jax.tree_util.tree_map(lambda hist, value: hist.at[i].set(value), blocks, output)
            measured = jax.tree_util.tree_map(lambda hist, value: hist.at[i].set(value), measured, values)
            residuals = output[0] if diagnostics else output
            eq = values[0]["l2_volume"] if numerics.diagnostics else values[0]
            eq_peak = jnp.maximum(eq_peak, eq)
            mass_peak = jnp.maximum(mass_peak, jnp.max(jnp.where(jnp.isfinite(residuals), residuals, 0)))
            failed = ~finite | ~jnp.all(jnp.isfinite(residuals))
            converged = (mass_peak > 0) & (residuals[-1] <= 0)
            frozen = (block_iters > 1) & jnp.all(residuals == residuals[0])
            if diagnostics:
                frozen &= output[1]["changed_count"] == 0
            if changed is not None:
                frozen &= ~changed
            settled = device_equations_settled(eq, eq_peak, numerics, names=names)
            if numerics.forces_tol is not None:
                count = i + 1
                start = 3 * count // 4
                tail = (coeff_columns >= start) & (coeff_columns < count)
                forces = jnp.stack([measured[1][key] for key in ("cl", "cd", "cm")], axis=1)
                spread = (jnp.max(jnp.where(tail, forces, -jnp.inf), axis=0)
                          - jnp.min(jnp.where(tail, forces, jnp.inf), axis=0))
                settled &= ((count - start >= 4)
                            & jnp.all(jnp.isfinite(spread) & (spread < numerics.forces_tol)))
            reason = jnp.select([failed, converged, frozen, settled], [1, 2, 3, 4], default=0).astype(jnp.int32)
            return i + 1, reason, current, eq_peak, mass_peak, blocks, measured

        residual_spec = block_output[0] if diagnostics else block_output
        first = (jnp.int32(0), jnp.int32(0), initial, initial_peak, jnp.zeros((), residual_spec.dtype),
                 block_history, measured_history)
        i, reason, final, _, _, blocks, measured = jax.lax.while_loop(
            lambda c: (c[0] < max_blocks) & (c[1] == 0), advance, first)
        return i, reason, final, blocks, measured

    start = time.perf_counter()
    compiled = solve.lower(state, equation_peak).compile()
    compile_time = time.perf_counter() - start
    count, reason, final, blocks, measured = compiled(state, equation_peak)
    count, reason, blocks, measured = jax.device_get((count, reason, blocks, measured))
    count = int(count)
    if diagnostics:
        residuals, reports = blocks
    else:
        residuals, reports = blocks, {}
    for i in range(count):
        on_block(i, final, residuals[i], jax.tree_util.tree_map(lambda a: a[i], measured))
    elapsed = time.perf_counter() - start
    # Without a synchronization in the loop, attribute its average block time to the first block.
    first_block = compile_time + (elapsed - compile_time) / count
    history = ConvergenceHistory(residuals[:count].reshape(-1), reasons[int(reason)], count * block_iters,
                                 elapsed, first_block, block_diagnostics=tuple(
                                     {**{key: value[i] for key, value in reports.items()},
                                      "iteration": (i + 1) * block_iters}
                                     for i in range(count)) if diagnostics else ())
    return final, history


def _march(case: Case, grid: Atlas, progress: Progress | None, start: tuple[Array, Atlas] | None,
           devices: Sequence | None = None, *, initial: Array | None = None,
           device_convergence: bool = False) -> Result:
    """One grid to steady state, from freestream or from ``start``, a coarser
    grid's solution ``(U, atlas)``."""
    flow, numerics = case.flow, case.numerics
    length = case.grid_spec.force_reference()
    if devices:
        grid = SlabAtlas.of(grid, devices)
    diagnostics = numerics.diagnostics and numerics.solver == "implicit"
    step, state = make_stepper(case, grid, initial=initial, diagnostics=diagnostics)
    if start is not None:
        state = state._replace(U=prolong(*start, grid, state.U))

    cl_history: list[float] = []
    coefficients: list[tuple[float, float, float]] = []
    peak = [0.0]
    equation_operator = make_residual(grid, flow, numerics)
    names = ["mass", *[f"momentum_{a}" for a in "xyz"[:grid.components]], "energy"]
    if state.U.shape[0] > len(names):
        names.append("sa")
    equation_monitor = jax.jit(equation_operator.equation_diagnostics if numerics.diagnostics
                               else equation_operator.equation_norms)
    initial_equations = equation_monitor(state.U)
    equation_peak = np.asarray(initial_equations["l2_volume"] if numerics.diagnostics else initial_equations)
    convergence_history = []
    forces = None
    implicit_step = numerics.solver == "implicit"
    measure_equations = equation_operator.equation_diagnostics if numerics.diagnostics else equation_operator.equation_norms

    @jax.jit
    def monitor(before, current):
        finite = jnp.all(jnp.isfinite(current.U))
        changed = ((before.cfl != current.cfl) | jnp.any(before.U != current.U)) if implicit_step else None
        return finite, changed, (measure_equations(current.U), force_coefficients(current.U, grid, flow, length))

    def on_block(index: int, _current, residuals: np.ndarray, measurements) -> bool:
        nonlocal equation_peak, forces
        measured, loads = measurements
        equation_norms = np.asarray(measured["l2_volume"] if numerics.diagnostics else measured)
        equation_peak = equation_norms if equation_peak is None else np.maximum(equation_peak, equation_norms)
        finite = residuals[np.isfinite(residuals)]
        if finite.size:
            peak[0] = max(peak[0], float(finite.max()))
        forces = forces_from_coefficients(loads)
        cl_history.append(forces.cl)
        coefficients.append((forces.cl, forces.cd, forces.cm))
        if progress is not None:
            last = float(residuals[-1])
            orders = float(np.log10(peak[0] / last)) if peak[0] > 0.0 and last > 0.0 else 0.0
            progress((index + 1) * case.block_iters, last, orders, forces)
        # NaN (too few blocks) compares false: a short run never stops here.
        record = convergence_record(equation_norms, equation_peak, np.asarray(coefficients), numerics, names=names)
        if numerics.diagnostics:
            record["equations"] = diagnostics_report(equation_operator, measured)
        convergence_history.append(dict(record, iteration=(index + 1) * case.block_iters))
        return record["settled"]

    if device_convergence:
        state, history = _march_on_device(step, state, monitor, on_block, jnp.asarray(equation_peak),
                                          names, case, diagnostics)
    else:
        state, history = run_to_steady(
            step, state, max_iters=numerics.max_iters, tol=0.0, block_iters=case.block_iters,
            on_block=on_block, monitor_block=monitor, slabs=devices, allow_exact_zero=False,
            diagnostics=diagnostics)
    assert forces is not None
    return Result(
        case=case, grid=grid, U=state.U,
        forces=forces,
        history=history, cl_history=np.asarray(cl_history),
        timing=Timing.of(history, case.block_iters, grid.cells, devices=len(devices) if devices else 1),
        coefficients_history=np.asarray(coefficients),
        convergence_history=tuple(convergence_history),
    )
