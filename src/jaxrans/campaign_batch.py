"""A bounded batch of geometries and flow conditions in one compiled scan.

Within each geometry, vmap advances Mach/alpha lanes with their own M/Re.
Different geometry groups are traced into the same device program: the Python
loop assembles the graph, not a sequence of host-side CFD solves. Tuple states
allow different block topologies without padding meshes. One GPU per batch.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from .core.state import freestream_primitive
from .numerics.residual import make_residual
from .post.forces import integrate_forces
from .simulation import Result, Timing, cl_spread, make_stepper
from .solvers.driver import ConvergenceHistory, ExitReason


def run_case_batch(cases, grids, progress=None) -> list[Result]:
    """Cases with common numerics/physics, arbitrary grids; called on chosen device.

    Monitor once per block, as for a single run. Completed/failed lanes are
    frozen at subsequent blocks. ``max_iters`` is respected exactly.
    Returned fields live on the host so successive batches do not fill VRAM.
    """
    if not cases or len(cases) != len(grids):
        raise ValueError("a batch needs matching nonempty cases and grids")
    prototype = cases[0]
    for case in cases:
        if (case.numerics, case.implicit, case.rk_scheme, case.block_iters) != (
                prototype.numerics, prototype.implicit, prototype.rk_scheme, prototype.block_iters):
            raise ValueError("a batch needs common numerics")
        if (case.flow.gamma, case.flow.reynolds, case.flow.sutherland_ratio, case.flow.turbulence,
                case.flow.nu_tilde_ratio) != (prototype.flow.gamma, prototype.flow.reynolds,
                prototype.flow.sutherland_ratio, prototype.flow.turbulence, prototype.flow.nu_tilde_ratio):
            raise ValueError("a batch needs common gas, Reynolds and turbulence")
        if case.sequencing:
            raise ValueError("campaign batches do not support grid_sequencing; set it to 0")
    groups = {}
    for k, grid in enumerate(grids):
        groups.setdefault(id(grid), []).append(k)
    steps, states, lanes, conditions = [], [], [], []
    for indices in groups.values():
        case, grid = cases[indices[0]], grids[indices[0]]
        q = jnp.asarray([freestream_primitive(cases[k].flow, grid.components) for k in indices])
        mu = jnp.asarray([0.0 if cases[k].flow.inviscid else cases[k].flow.mu_inf for k in indices])
        distance = make_residual(grid, case.flow, case.numerics).distance

        # Bind each geometry's closure now, before moving to the next group.
        def build(case=case, grid=grid, distance=distance):
            def initial(qi, mui):
                return make_stepper(case, grid, qi, distance, mui)[1]

            def advance(state, qi, mui):
                return make_stepper(case, grid, qi, distance, mui)[0](state)
            return initial, advance

        initial, advance = build()
        states.append(jax.jit(jax.vmap(initial))(q, mu))
        steps.append(jax.vmap(advance))
        conditions.append((q, mu))
        lanes.append(np.asarray(indices))
    states = tuple(states)
    conditions = tuple(conditions)
    count = len(cases)

    def one_step(states, active, conditions):
        next_states, residuals = [], []
        for state, step, (q, mu), ids in zip(states, steps, conditions, lanes):
            updated, norm = step(state, q, mu)
            mask = active[ids]
            updated = jax.tree_util.tree_map(
                lambda new, old: jnp.where(mask.reshape((len(ids),) + (1,) * (new.ndim-1)), new, old),
                updated, state)
            # A finite density residual alone does not guarantee a finite energy/SA state.
            finite = jnp.all(jnp.isfinite(updated.U).reshape((len(ids), -1)), axis=1)
            residuals.append(jnp.where(finite, norm, jnp.nan))
            next_states.append(updated)
        order = np.argsort(np.concatenate(lanes))
        return tuple(next_states), jnp.concatenate(residuals)[order]

    @jax.jit
    def block(states, active, conditions):
        return jax.lax.scan(lambda s, _: one_step(s, active, conditions), states,
                            xs=None, length=prototype.block_iters)

    active = np.ones(count, dtype=bool)
    reasons = np.full(count, ExitReason.MAX_ITERS, dtype=object)
    peaks = np.zeros(count)
    histories = [[] for _ in cases]
    coefficients = [[] for _ in cases]
    iterations = np.zeros(count, dtype=int)
    started = time.perf_counter()
    first_block = None
    done = 0
    while done < prototype.numerics.max_iters and active.any():
        n = min(prototype.block_iters, prototype.numerics.max_iters - done)
        if n == prototype.block_iters:
            states, residual = block(states, jnp.asarray(active), conditions)
        else:
            states, residual = jax.jit(lambda s, a, c: jax.lax.scan(
                lambda t, _: one_step(t, a, c), s, xs=None, length=n))(states, jnp.asarray(active), conditions)
        residual = np.asarray(residual)  # synchronize before timing or inspecting results
        if first_block is None:
            first_block = time.perf_counter() - started
        fields = {}
        for state, ids in zip(states, lanes):
            for local, k in enumerate(ids):
                fields[k] = np.asarray(state.U[local])
        for k in np.flatnonzero(active):
            column = residual[:, k]
            histories[k].extend(column.tolist())
            iterations[k] += n
            peaks[k] = max(peaks[k], float(np.max(np.where(np.isfinite(column), column, 0))))
            force = integrate_forces(fields[k], grids[k], cases[k].flow, cases[k].grid_spec.force_reference())
            coefficients[k].append((force.cl, force.cd, force.cm))
            if not np.isfinite(column).all() or not np.isfinite(coefficients[k][-1]).all():
                reasons[k] = ExitReason.NON_FINITE
            elif peaks[k] > 0 and column[-1] <= prototype.numerics.tol * peaks[k]:
                reasons[k] = ExitReason.CONVERGED
            elif n > 1 and np.all(column == column[0]):
                reasons[k] = ExitReason.STALLED
            elif prototype.numerics.forces_tol is not None and all(
                    cl_spread(h) < prototype.numerics.forces_tol for h in np.transpose(coefficients[k])):
                reasons[k] = ExitReason.SETTLED
            active[k] = reasons[k] == ExitReason.MAX_ITERS
        done += n
        if progress is not None:
            progress(done, tuple(reasons))
    elapsed = time.perf_counter() - started
    results = []
    for k, (case, grid) in enumerate(zip(cases, grids)):
        history = ConvergenceHistory(np.asarray(histories[k]), str(reasons[k]), int(iterations[k]),
                                     elapsed, first_block)
        force = integrate_forces(fields[k], grid, case.flow, case.grid_spec.force_reference())
        timing = Timing.of(history, prototype.block_iters, grid.cells, cases=count)
        results.append(Result(case, grid, fields[k], force, history,
                              np.asarray(coefficients[k])[:, 0], timing))
    return results
