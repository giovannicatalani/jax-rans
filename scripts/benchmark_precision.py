#!/usr/bin/env python3
"""Measure fixed-work implicit RANS steps on a multiblock C-grid.

Run each precision in a fresh process, for example:
  python scripts/benchmark_precision.py --precision fp64 --out results/fp64.json
  python scripts/benchmark_precision.py --precision fp32 --out results/fp32.json
The first block includes compilation; later blocks are synchronized and timed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxrans import load_case
from jaxrans.numerics.residual import make_residual
from jaxrans.solvers import implicit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=Path('examples/naca0012_blunt_sa'))
    parser.add_argument('--precision', choices=('fp32', 'fp64', 'mixed', 'mixt'), required=True)
    parser.add_argument('--block-iters', type=int, default=20)
    parser.add_argument('--blocks', type=int, default=5)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.block_iters < 1 or args.blocks < 2:
        parser.error('Need positive block-iters and at least two blocks')
    device = jax.devices()[0]
    if device.platform != 'gpu':
        raise RuntimeError(f'GPU required, got {device}')
    case = load_case(args.case, (f'case.general.precision={args.precision}',))
    grid = case.grid()
    if len(grid.mesh) < 2 or case.raw['case']['domain']['grid'] != 'c_grid':
        raise ValueError('Benchmark requires a multiblock C-grid')
    if case.numerics.solver != 'implicit' or case.flow.turbulence == 'none':
        raise ValueError('Benchmark requires implicit turbulent RANS')
    op = make_residual(grid, case.flow, case.numerics)
    jac = None if case.implicit.jacobian == 'exact' else make_residual(
        grid, case.flow, case.numerics, first_order=True, distance=op.distance)
    step = implicit.make_newton_step(op, case.implicit, jac, diagnostics=True, mixed_precision=case.numerics.mixed_precision)
    state = implicit.initial_state(op, case.implicit, dtype=case.numerics.dtype)

    def scan(state):
        def one(state, _):
            updated, norm, diag = step(state)
            changed = jnp.any(updated.U != state.U)
            return updated, (norm, diag.accepted, diag.correction_finite, diag.physical, changed)
        return jax.lax.scan(one, state, None, length=args.block_iters)

    run = jax.jit(scan)
    records = []
    for index in range(args.blocks):
        start = time.perf_counter()
        state, history = run(state)
        jax.block_until_ready((state, history))
        elapsed = time.perf_counter() - start
        norms, accepted, finite, physical, changed = map(np.asarray, history)
        records.append(dict(seconds=elapsed, iterations=args.block_iters,
                            accepted=int(accepted.sum()), changed=int(changed.sum()),
                            finite_corrections=int(finite.sum()), physical=int(physical.sum()),
                            residual_first=float(norms[0]), residual_last=float(norms[-1])))
        print(f'{args.precision} block {index + 1}: {elapsed:.3f} s, '
              f'accepted {int(accepted.sum())}/{args.block_iters}, residual {norms[-1]:.6g}', flush=True)
    source = Path('src/jaxrans')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*.py'))}
    result = dict(precision=args.precision, field_dtype=str(state.U.dtype), device=str(device.device_kind),
                  jax_version=jax.__version__, case=case.raw, cells=grid.cells, blocks_in_mesh=len(grid.mesh),
                  source_sha256=hashes, blocks=records, finite_fields=bool(np.isfinite(np.asarray(state.U)).all()),
                  steady_seconds_per_iteration=float(np.median([r['seconds'] / args.block_iters for r in records[1:]])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str) + '\n')
    if not result['finite_fields'] or any(r['accepted'] != args.block_iters or r['changed'] != args.block_iters
                                          or r['finite_corrections'] != args.block_iters for r in records):
        raise RuntimeError('Benchmark contains failed or unchanged steps; do not compare as successful work')


if __name__ == '__main__':
    main()
