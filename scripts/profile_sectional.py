#!/usr/bin/env python3
"""Capture a JAX trace of fixed-work implicit 2.5D or 2.75D steps.

Run only when no other GPU work is competing, e.g.:
  JAX_PLATFORMS=cuda python scripts/profile_sectional.py \
    examples/rae2822_transonic_conical_verification --set case.domain.sectional.taper_rate=0.2 \
    --out results/profiles/transonic_tau02

Warm-up and compilation are measured separately; the trace contains subsequent
synchronized blocks and is a diagnostic, not a time-to-convergence benchmark.
Use the JAX/XLA trace viewer to inspect GPU kernels and host-device transfers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from jaxrans import load_case
from jaxrans.numerics.residual import make_residual
from jaxrans.solvers import implicit
from jaxrans.solvers.driver import make_block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--out', type=Path, required=True, help='new, empty trace directory')
    parser.add_argument('--block-iters', type=int, default=20)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--warmup-blocks', type=int, default=2)
    parser.add_argument('--coarsen', type=int, default=0, help='mesh coarsening; useful for CPU smoke tests')
    parser.add_argument('--allow-cpu', action='store_true', help='allow CPU for functional testing only')
    args = parser.parse_args()
    if min(args.block_iters, args.blocks, args.warmup_blocks) < 1 or args.coarsen < 0:
        parser.error('block counts must be positive and coarsen nonnegative')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('output directory must be empty')
    device = jax.devices()[0]
    if device.platform != 'gpu' and not args.allow_cpu:
        parser.error(f'GPU required, got {device}; --allow-cpu is only for functional tests')
    case = load_case(args.case, tuple(['case.general.precision=mixed', *args.overrides]))
    if case.numerics.solver != 'implicit':
        parser.error('an implicit case is required')
    grid = case.grid(coarsen=args.coarsen)
    operator = make_residual(grid, case.flow, case.numerics)
    jac = None if case.implicit.jacobian == 'exact' else make_residual(
        grid, case.flow, case.numerics, first_order=True, distance=operator.distance)
    step = implicit.make_newton_step(operator, case.implicit, jac, diagnostics=True,
                                     mixed_precision=case.numerics.mixed_precision)
    block = make_block(step, args.block_iters, diagnostics=True)
    state = implicit.initial_state(operator, case.implicit, dtype=case.numerics.dtype)
    args.out.mkdir(parents=True, exist_ok=True)
    warmup = []
    for i in range(args.warmup_blocks):
        start = time.perf_counter()
        state, report = block(state)
        jax.block_until_ready((state, report))
        warmup.append(time.perf_counter() - start)
        print(f'warmup {i + 1}: {warmup[-1]:.3f}s', flush=True)
    durations = []
    summaries = []
    with jax.profiler.trace(str(args.out)):
        for i in range(args.blocks):
            start = time.perf_counter()
            with jax.profiler.StepTraceAnnotation('implicit_block', step_num=i):
                state, report = block(state)
                jax.block_until_ready((state, report))
            durations.append(time.perf_counter() - start)
            norms, diagnostics = report
            summaries.append({'residual_last': float(np.asarray(norms)[-1]),
                              'accepted': int(np.asarray(diagnostics['accepted_count'])),
                              'fallbacks': int(np.asarray(diagnostics['linear_fp64_fallback_count']))})
            print(f'traced block {i + 1}: {durations[-1]:.3f}s', flush=True)
    (args.out / 'metadata.json').write_text(json.dumps({
        'case': str(args.case), 'resolved_case': case.raw, 'precision': case.numerics.precision,
        'device': str(device), 'device_kind': device.device_kind, 'jax_version': jax.__version__,
        'cells': grid.cells, 'coarsen': args.coarsen, 'block_iters': args.block_iters,
        'warmup_seconds': warmup, 'traced_seconds': durations, 'block_diagnostics': summaries,
        'note': 'Trace overhead included; fixed work from freestream, not time to convergence.',
    }, indent=2, default=str) + '\n')
    print(f'Trace written to {args.out}', flush=True)


if __name__ == '__main__':
    main()
