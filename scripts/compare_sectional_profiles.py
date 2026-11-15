#!/usr/bin/env python3
"""Capture comparable transonic RAE2822 2.5D and 2.75D JAX profiles.

Run after other GPU jobs finish. Each case runs in a fresh process; the
sectional case uses zero taper by default to match the 2.5D geometry.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='new or empty output directory')
    parser.add_argument('--taper', type=float, default=0., help='2.75D taper; 0 matches the 2.5D section')
    parser.add_argument('--block-iters', type=int, default=20)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--warmup-blocks', type=int, default=2)
    parser.add_argument('--coarsen', type=int, default=0)
    parser.add_argument('--allow-cpu', action='store_true', help='functional smoke test only')
    args = parser.parse_args()
    if min(args.block_iters, args.blocks, args.warmup_blocks) < 1 or args.coarsen < 0:
        parser.error('block counts must be positive and coarsen nonnegative')
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error('output directory must be new or empty')
    args.out = args.out.resolve()
    root = Path(__file__).resolve().parent.parent
    profiler = root / 'scripts/profile_sectional.py'
    args.out.mkdir(parents=True, exist_ok=True)
    profiles = {}
    for label, case in [('2p5d', 'rae2822_transonic_swept_verification'),
                        ('2p75d', 'rae2822_transonic_conical_verification')]:
        directory = args.out / label
        command = [sys.executable, str(profiler), str(root / 'examples' / case),
                   '--out', str(directory), '--block-iters', str(args.block_iters),
                   '--blocks', str(args.blocks), '--warmup-blocks', str(args.warmup_blocks),
                   '--coarsen', str(args.coarsen)]
        if args.allow_cpu:
            command.append('--allow-cpu')
        if label == '2p75d':
            command.extend(['--set', f'case.domain.sectional.taper_rate={args.taper}'])
        print(f'Profiling {label}', flush=True)
        subprocess.run(command, cwd=root, check=True)
        profiles[label] = json.loads((directory / 'metadata.json').read_text())
    a, b = profiles['2p5d'], profiles['2p75d']
    if a['device_kind'] != b['device_kind'] or a['cells'] != b['cells']:
        raise RuntimeError('Profiles do not use the same device kind and number of cells')
    seconds = {name: statistics.median(data['traced_seconds']) / args.block_iters
               for name, data in profiles.items()}
    result = {'taper': args.taper, 'device_kind': a['device_kind'], 'cells': a['cells'],
              'median_traced_seconds_per_iteration': seconds,
              'ratio_2p75d_over_2p5d': seconds['2p75d'] / seconds['2p5d'],
              'profiles': {key: {'path': key, 'warmup_seconds': value['warmup_seconds'],
                                  'block_diagnostics': value['block_diagnostics']}
                           for key, value in profiles.items()},
              'limitations': ['Trace overhead included; fixed work from freestream, not time to convergence.',
                              'Diagnostics enabled in both cases; timings are not a production benchmark.',
                              'Nonzero taper changes geometry and flow relative to 2.5D.']}
    (args.out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Ratio 2.75D / 2.5D: {result['ratio_2p75d_over_2p5d']:.3f}")
    print(f'Results: {args.out}', flush=True)


if __name__ == '__main__':
    main()
