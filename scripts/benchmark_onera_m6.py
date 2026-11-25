"""Time the ONERA M6 Newton step on CUDA, with synchronized JIT blocks."""
import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from jaxrans import load_case
from jaxrans.numerics.residual import make_residual
from jaxrans.post.forces import integrate_forces
from jaxrans.simulation import Result, Timing, make_stepper
from jaxrans.solvers.driver import ConvergenceHistory

CHECKPOINTS = (20, 200, 400, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", choices=("fp64", "mixed"), default="mixed")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--save-checkpoints", type=Path)
    parser.add_argument("--replay-checkpoints", type=Path)
    parser.add_argument("--write-result", type=Path)
    args = parser.parse_args()
    device = jax.devices()[0]
    if device.platform != "gpu":
        raise RuntimeError(f"CUDA required, got {device}")
    case = load_case("examples/onera_m6_sa", (f"case.general.precision={args.precision}",))
    grid = case.grid()
    step, state = make_stepper(case, grid, diagnostics=True)
    length = case.block_iters
    max_iters = case.numerics.max_iters if args.max_iters is None else args.max_iters
    if max_iters < length or max_iters % length:
        parser.error(f"--max-iters must be a positive multiple of {length}")

    def one(state, _):
        updated, norm, diagnostics = step(state)
        return updated, (norm, diagnostics)

    def block(state):
        return jax.lax.scan(one, state, None, length=length)

    start = time.perf_counter()
    compiled = jax.jit(block).lower(state).compile()
    compile_s = time.perf_counter() - start
    print(f"{device.device_kind}, {grid.cells} cells, {args.precision}, compile {compile_s:.3f}s", flush=True)

    def timed_block(state):
        start = time.perf_counter()
        updated, (norms, diag) = jax.block_until_ready(compiled(state))
        elapsed = time.perf_counter() - start
        report = dict(seconds=elapsed, accepted=int(np.sum(diag.accepted)), changed=int(np.sum(diag.changed)),
                      fallback=int(np.sum(diag.linear_fp64_fallback)), residual=float(norms[-1]),
                      cfl=float(updated.cfl))
        return updated, np.asarray(norms), report

    records, residuals, checkpoints = [], [], {}
    for iteration in range(length, max_iters + 1, length):
        state, norms, report = timed_block(state)
        residuals.append(norms)
        records.append(report)
        if args.save_checkpoints and iteration in CHECKPOINTS:
            for field in state._fields:
                checkpoints[f"{iteration}_{field}"] = np.asarray(getattr(state, field))
        if iteration <= 100 or iteration % 100 == 0:
            print(iteration, report, flush=True)
        if norms[-1] <= case.numerics.tol:
            break

    if args.save_checkpoints:
        np.savez(args.save_checkpoints, **checkpoints)
    replay = {}
    source = args.replay_checkpoints or args.save_checkpoints
    if source:
        with np.load(source) as inputs:
            for iteration in CHECKPOINTS:
                initial = type(state)(*(jax.device_put(inputs[f"{iteration}_{field}"]) for field in state._fields))
                jax.block_until_ready(initial)
                _, _, _ = timed_block(initial)
                replay[iteration] = [timed_block(initial)[2] for _ in range(7)]

    op = make_residual(grid, case.flow, case.numerics)
    final_residual = float(op.norm(op.residual(state.U)))
    forces = integrate_forces(state.U, grid, case.flow, case.grid_spec.force_reference())
    payload = dict(gpu=device.device_kind, cells=grid.cells, jax=jax.__version__, length=length,
                   precision=args.precision, tolerance=case.numerics.tol, max_iters=max_iters,
                   reached_tolerance=bool(norms[-1] <= case.numerics.tol),
                   compile_s=compile_s, field_dtype=str(state.U.dtype), residual_dtype=str(norms.dtype),
                   records=records, replay=replay, final_residual=final_residual, cl=forces.cl, cd=forces.cd)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.write_result:
        elapsed = sum(r["seconds"] for r in records)
        reason = "benchmark_residual_target" if payload["reached_tolerance"] else "max_iters"
        history = ConvergenceHistory(np.concatenate(residuals), reason, len(records) * length,
                                     compile_s + elapsed, compile_s + records[0]["seconds"])
        timing = Timing.of(history, length, grid.cells)
        Result(case, grid, state.U, forces, history, np.empty(0), timing).write(args.write_result)
    print({key: value for key, value in payload.items() if key not in ("records", "replay")}, flush=True)


if __name__ == "__main__":
    main()
