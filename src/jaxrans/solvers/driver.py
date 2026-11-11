"""The iteration driver: blocks of steps fused into one on-device ``lax.scan``,
with Python between blocks for monitoring. It knows nothing of the equations,
so the explicit and implicit paths share it and its convergence test. A step
may advance a batch of independent cases (one residual each), split over
devices along the leading axis with no communication; or one case may be
split over devices in slabs of its canvas (``slabs``): the residual's
stencils exchange their halos explicitly (``numerics/slabs.py``), XLA
partitions the rest of the step (GSPMD).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Protocol, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from ..numerics.slabs import SPACE_AXIS

BATCH_AXIS = "case"


class StepFunction(Protocol):
    """``state -> (state, residual norm)``, pure and jittable; the norm is ``(n_cases,)`` for a batch."""

    def __call__(self, state: Any) -> tuple[Any, Array]: ...


class ExitReason:
    """Why the driver stopped."""

    CONVERGED = "converged"
    SETTLED = "settled"
    MAX_ITERS = "max_iters"
    NON_FINITE = "non_finite"
    STALLED = "stalled"
    SKIPPED = "skipped"


@dataclass
class ConvergenceHistory:
    """``residual``: ``(n_iters,)`` or ``(n_iters, n_cases)``; ``reason``: ``CONVERGED``
    only if every case converged; times include compilation (the first block's)."""

    residual: np.ndarray
    reason: str
    n_iters: int
    seconds: float = float("nan")
    first_block_seconds: float = float("nan")
    case_reasons: tuple[str, ...] = ()
    case_iterations: tuple[int, ...] = ()
    block_diagnostics: tuple[dict, ...] = ()

    def case(self, k: int) -> "ConvergenceHistory":
        """Case ``k`` of a batch, as a single run's history."""
        n = self.case_iterations[k] if self.case_iterations else self.n_iters
        return ConvergenceHistory(self.residual[:n, k], self.case_reasons[k], n,
                                  self.seconds, self.first_block_seconds,
                                  block_diagnostics=tuple({key: value if key == "iteration" else np.asarray(value)[k]
                                                          for key, value in row.items()}
                                      for row in self.block_diagnostics if row["iteration"] <= n))

    @property
    def converged(self) -> bool:
        return self.reason == ExitReason.CONVERGED

    @property
    def peak(self) -> float:
        """Largest finite residual over the run: the convergence reference."""
        finite = self.residual[np.isfinite(self.residual)]
        return float(finite.max()) if finite.size else float("nan")

    @property
    def orders_dropped(self) -> float:
        """Orders of magnitude the residual fell from its peak (``run_to_steady``)."""
        if self.residual.size == 0:
            return 0.0
        peak, last = self.peak, self.residual[-1]
        if not (np.isfinite(peak) and np.isfinite(last)) or peak <= 0 or last <= 0:
            return float("nan")
        return float(np.log10(peak / last))

    def summary(self) -> str:
        return (
            f"{self.reason} after {self.n_iters} iterations, "
            f"residual peak {self.peak:.6e} -> {self.residual[-1]:.6e} "
            f"({self.orders_dropped:.2f} orders from peak)"
        )


def make_block(step: StepFunction, block_iters: int, devices: Sequence | None = None,
               shardings: Any = None, diagnostics: bool = False) -> Callable[[Any], tuple[Any, Array]]:
    """``block_iters`` steps as one jitted call, ``state -> (state, residuals)``;
    with two or more ``devices`` a batch is split over them (``shard_map``);
    with ``shardings`` (``slab_shardings``) one state is kept split so."""

    def run_block(state: Any) -> tuple[Any, Array]:
        def one(carry, _):
            result = step(carry)
            return result[0], (result[1], result[2]) if diagnostics else result[1]
        state, output = lax.scan(one, state, xs=None, length=block_iters)
        if not diagnostics:
            return state, output
        residual, report = output
        summary = {}
        for key, value in report._asdict().items():
            if value.dtype == jnp.bool_:
                summary[key + "_count"] = jnp.sum(value, axis=0)
            else:
                summary[key + "_min"] = jnp.min(value, axis=0)
                summary[key + "_max"] = jnp.max(value, axis=0)
                summary[key + "_last"] = value[-1]
        return state, (residual, summary)

    if shardings is not None:
        return jax.jit(run_block, in_shardings=(shardings,), out_shardings=(shardings, None))
    if devices is None or len(devices) < 2:
        return jax.jit(run_block)
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    output_spec = (P(None, BATCH_AXIS), P(BATCH_AXIS)) if diagnostics else P(None, BATCH_AXIS)
    return jax.jit(shard_map(run_block, mesh=_mesh(devices), in_specs=(P(BATCH_AXIS),),
                             out_specs=(P(BATCH_AXIS), output_spec), check_rep=False))


def slab_shardings(state: Any, devices: Sequence) -> Any:
    """Shardings splitting every field of ``state`` (``(nvar, *canvas)``)
    along its first spatial axis over ``devices``; scalars replicated."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = _mesh(devices, SPACE_AXIS)
    return jax.tree_util.tree_map(
        lambda leaf: NamedSharding(mesh, P(None, SPACE_AXIS) if np.ndim(leaf) > 1 else P()), state)


def _mesh(devices: Sequence, axis: str = BATCH_AXIS):
    from jax.sharding import Mesh

    return Mesh(np.array(devices), (axis,))


@lru_cache(maxsize=16)
def _masked_block(step, block_iters, devices, diagnostics=False):
    """Reuse one executable across continuation rounds (freestream is dynamic)."""
    def masked_step(carry):
        current, active, last = carry
        output = step(current)
        candidate, residual = output[:2]
        kept = jax.tree_util.tree_map(
            lambda new, old: jnp.where(active.reshape(active.shape + (1,) * (old.ndim - 1)),
                                      new, old), candidate, current)
        residual = jnp.where(active, residual, last)
        result = ((kept, active, residual), residual)
        if diagnostics:
            report = output[2]._replace(**{
                key: jnp.where(active, value, jnp.full_like(value, jnp.nan)
                               if jnp.issubdtype(value.dtype, jnp.floating) else jnp.zeros_like(value))
                for key, value in output[2]._asdict().items()})
            return *result, report
        return result

    return make_block(masked_step, block_iters, devices, diagnostics=diagnostics)


def run_to_steady(
    step: StepFunction,
    state: Any,
    *,
    max_iters: int,
    tol: float,
    block_iters: int = 100,
    on_block: Callable[[int, Any, np.ndarray], bool | np.ndarray | None] | None = None,
    finite_check: Callable[[Any], Array] | None = None,
    devices: Sequence | None = None,
    slabs: Sequence | None = None,
    active_mask: np.ndarray | None = None,
    allow_exact_zero: bool = True,
    diagnostics: bool = False,
    control_changed: Callable[[Any, Any], Array] | None = None,
) -> tuple[Any, ConvergenceHistory]:
    """Iterate until the residual is ``tol`` times its running **peak**, per case.

    The peak, not the first value: a uniform-freestream start has a near-zero
    first residual (free-stream preservation), a degenerate reference.
    ``on_block(block, state, residuals)`` runs between blocks (the only place
    for I/O); returning true stops every running case as ``SETTLED`` (the
    caller's own test, e.g. on forces). A block in which a case's residual
    does not change at all stops that case as ``STALLED``: no step was
    taken. ``finite_check(state)`` is an optional device-side health test.
    ``devices`` split a batch of cases, ``slabs`` one case (the step's own,
    ``numerics/slabs.py``).
    With ``active_mask`` every state leaf has a leading case axis. Finished
    cases are frozen between blocks, inactive entries are skipped, and
    ``case(k)`` trims their history to their actual stopping iteration.
    ``on_block`` and ``finite_check`` may return one boolean per case.
    ``diagnostics`` accepts a step's third NamedTuple result, reducing its
    scalar facts on device before transferring one summary per block.
    ``control_changed(old, new)`` prevents a constant residual being declared
    stalled while solver control (for example a CFL cutback) is progressing.
    """
    masked = active_mask is not None
    if masked:
        if slabs:
            raise ValueError("active_mask cannot be combined with spatial slabs")
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.ndim != 1 or not active_mask.size or not active_mask.any():
            raise ValueError("active_mask must be a nonempty vector with an active case")
        if any(np.ndim(leaf) < 1 or np.shape(leaf)[0] != active_mask.size
               for leaf in jax.tree_util.tree_leaves(state)):
            raise ValueError("every masked state leaf must have a leading axis matching active_mask")
        state = (state, jnp.asarray(active_mask), jnp.zeros(active_mask.shape))
    shardings = slab_shardings(state, slabs) if slabs else None
    if shardings is not None:
        state = jax.device_put(state, shardings)
    block = (_masked_block(step, block_iters, tuple(devices) if devices else None, diagnostics) if masked
             else make_block(step, block_iters, devices, shardings, diagnostics))
    if devices is not None and len(devices) > 1:
        from jax.sharding import NamedSharding, PartitionSpec as P

        state = jax.device_put(state, NamedSharding(_mesh(devices), P(BATCH_AXIS)))
    n_blocks = max(1, -(-max_iters // block_iters))  # ceiling division

    blocks: list[np.ndarray] = []
    peak = reasons = finished_iters = None
    started = time.perf_counter()
    first_block = float("nan")
    reports = []
    for b in range(n_blocks):
        before = state[0] if masked else state
        state, block_residual = block(state)
        if diagnostics:
            block_residual, report = block_residual
            reports.append({key: np.asarray(value) for key, value in report.items()})
        block_residual = np.asarray(block_residual)
        if b == 0:
            first_block = time.perf_counter() - started
        blocks.append(block_residual)
        current = state[0] if masked else state
        stop = on_block(b, current, block_residual) if on_block is not None else False

        columns = block_residual.reshape(block_residual.shape[0], -1)   # (block, n_cases)
        if peak is None:
            peak = np.zeros(columns.shape[1])
            reasons = np.full(columns.shape[1], ExitReason.MAX_ITERS, dtype=object)
            finished_iters = np.zeros(columns.shape[1], dtype=int)
            if masked:
                reasons[~active_mask] = ExitReason.SKIPPED
        finite = np.where(np.isfinite(columns), columns, 0.0)
        peak = np.maximum(peak, finite.max(axis=0))
        last = columns[-1]
        running = reasons == ExitReason.MAX_ITERS
        failed = ~np.all(np.isfinite(columns), axis=0)
        if finite_check is not None:
            failed |= ~np.asarray(finite_check(current), dtype=bool)
        reasons[running & failed] = ExitReason.NON_FINITE
        zero = (last == 0) if allow_exact_zero else np.zeros_like(last, dtype=bool)
        reasons[running & ~failed & (((peak > 0) & (last <= tol * peak)) | zero)] = ExitReason.CONVERGED
        # A whole block with bitwise the same residual: no step was accepted
        # (the state is frozen), which would also look settled.
        frozen = (len(columns) > 1) & np.all(columns == columns[0], axis=0)
        if diagnostics and "changed_count" in reports[-1]:
            frozen &= ~(reports[-1]["changed_count"] > 0).reshape(-1)
        if control_changed is not None:
            frozen &= ~np.asarray(control_changed(before, current), dtype=bool).reshape(-1)
        reasons[(reasons == ExitReason.MAX_ITERS) & frozen] = ExitReason.STALLED
        reasons[(reasons == ExitReason.MAX_ITERS) & np.asarray(stop, dtype=bool)] = ExitReason.SETTLED
        finished_iters[running & (reasons != ExitReason.MAX_ITERS)] = (b + 1) * block_iters
        if masked:
            state = (state[0], jnp.asarray(reasons == ExitReason.MAX_ITERS), state[2])
        if not np.any(reasons == ExitReason.MAX_ITERS):
            break

    residual = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)
    finished_iters[reasons == ExitReason.MAX_ITERS] = residual.shape[0]
    failures = [r for r in reasons if r not in (ExitReason.CONVERGED, ExitReason.SETTLED, ExitReason.SKIPPED)]
    settled = ExitReason.SETTLED in reasons
    return state[0] if masked else state, ConvergenceHistory(
        residual=residual,
        reason=failures[0] if failures else ExitReason.SETTLED if settled else ExitReason.CONVERGED,
        n_iters=int(residual.shape[0]),
        seconds=time.perf_counter() - started,
        first_block_seconds=first_block,
        case_reasons=tuple(reasons) if residual.ndim > 1 else (),
        case_iterations=tuple(int(n) for n in finished_iters) if masked else (),
        block_diagnostics=tuple(dict(row, iteration=(i + 1) * block_iters) for i, row in enumerate(reports)),
    )
