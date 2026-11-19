"""Locate native crashes during the final state check or interpreter shutdown.

Replay saved fields: python -X faulthandler scripts/diagnose_shutdown.py fields DIR
Trace a real run:    python -X faulthandler scripts/diagnose_shutdown.py run CASE [CLI options]
Select CPU/CUDA with JAX_PLATFORMS. No solver or exit semantics are changed.
"""

from __future__ import annotations

import argparse
import atexit
import faulthandler
from pathlib import Path
import sys


def mark(message):
    print(f"[shutdown diagnostic] {message}", file=sys.stderr, flush=True)


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fields", "run"))
    parser.add_argument("path", type=Path)
    args, extra = parser.parse_known_args()
    # Register before JAX imports: this callback runs after callbacks registered later.
    atexit.register(mark, "early-registered atexit callback reached")
    mark("importing JAX and jaxrans")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jaxrans.core import state

    mark(f"devices: {jax.devices()}; JAX {jax.__version__}")
    original = state.first_bad_cell

    def traced_check(*positional, **keywords):
        mark("enter first_bad_cell")
        result = original(*positional, **keywords)
        mark(f"leave first_bad_cell: {result}")
        return result

    state.first_bad_cell = traced_check
    if args.mode == "run":
        from jaxrans.__main__ import main as cli_main

        status = cli_main([str(args.path), *extra])
        mark(f"CLI returned {status}")
    else:
        if extra:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        paths = list(args.path.glob("*fields.npz")) if args.path.is_dir() else [args.path]
        if len(paths) != 1:
            parser.error("expected exactly one fields.npz file")
        import json

        metadata = json.loads(paths[0].with_name(
            paths[0].name.replace("_fields.npz", "_metadata.json")
        ).read_text())
        gamma = metadata["config"]["flow"]["gamma"]
        dim = len(metadata["extra"].get("sectional", {}).get("axes", []))
        if dim not in (2, 3):
            parser.error("fields replay currently requires sectional axes in metadata")
        status = 0
        with np.load(paths[0], allow_pickle=False) as fields:
            for block in range(int(fields["n_blocks"])):
                mark(f"transferring block {block} to device")
                u = jnp.asarray(fields[f"U_{block}"])
                u.block_until_ready()
                physical, _ = traced_check(u, gamma, dim)
                status = max(status, int(not physical))
    mark("work complete; returning to interpreter")
    atexit.register(mark, "late-registered atexit callback reached")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
