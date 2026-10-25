"""One case split over devices in slabs (``run(..., devices=...)``,
``numerics/slabs.py``): the split step computes what the unsplit one does,
to round-off.

Four CPU devices, forced in a subprocess as the two-device polar test does,
both versions in it (another process compiles for another CPU configuration).
From a perturbed state, 40 implicit steps agree to 1e-13 on the M6 example,
1e-10 on the small C-grid.
Multi-block 3D: the M6 cut into blocks, as cut (every block's lines along
``j``: the slice chains) and re-indexed (lines along two axes: the flat
chains), written to CGNS (tests/unit/test_assemble3d.py).
From freestream they do not, by design of inexact Newton: the first residual
is tiny, GMRES stops at 1e-2 of it, and round-off decides whether a restart
cycle runs, so the first updates differ by ~1e-6 (then 2e-8 after 40 steps,
both converging on the same solution); there ``run(..., devices=...)`` is
only checked to run end to end.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from jaxrans import load_case
from jaxrans.numerics.slabs import SlabAtlas

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "2d": ("examples/naca0012_laminar", ("case.domain.cells_airfoil=64", "case.domain.cells_wake=16",
                                         "case.domain.cells_normal=24")),
    "3d": ("examples/onera_m6_sa", ()),
    "3d_blocks": ("examples/onera_m6_sa", ()),
    "3d_reindexed": ("examples/onera_m6_sa", ()),
}
BLOCKS = {"3d_blocks": (((6, 11), (), ()), False), "3d_reindexed": (((6,), (20,), ()), True)}


def blocks_case(directory: Path, name: str) -> Path:
    """The M6 example on its grid cut into blocks (re-indexed or not), as CGNS."""
    import json

    from ..unit.test_assemble3d import M6, ORIENTATIONS, cut, reindex
    from ..unit.test_grid_files import write_cgns
    from jaxrans.mesh.plot3d import read_plot3d_3d

    cuts, reorient = BLOCKS[name]
    (nodes,) = read_plot3d_3d(M6)
    rng = np.random.default_rng(5)
    blocks = [reindex(b, *ORIENTATIONS[rng.integers(48)]) if reorient else b for b in cut(nodes, cuts)]
    case = directory / "case"
    case.mkdir()
    write_cgns(case / "blocks.cgns", [((*b,), []) for b in blocks])
    doc = json.loads((ROOT / "examples/onera_m6_sa/case.json").read_text())
    doc["domain"]["file"] = "blocks.cgns"
    (case / "case.json").write_text(json.dumps(doc))
    (case / "numerical.json").write_text((ROOT / "examples/onera_m6_sa/numerical.json").read_text())
    return case


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(CASES))
def test_four_slabs_compute_the_one_device_step(name, tmp_path) -> None:
    directory, sets = CASES[name]
    directory = blocks_case(tmp_path, name) if name in BLOCKS else ROOT / directory
    script = textwrap.dedent(f"""
        import jax, jax.numpy as jnp, numpy as np
        from jaxrans import load_case, run
        from jaxrans.numerics.residual import make_residual
        from jaxrans.numerics.slabs import SlabAtlas
        from jaxrans.simulation import make_stepper
        from jaxrans.solvers.driver import make_block, slab_shardings
        assert len(jax.devices()) == 4, jax.devices()
        from jaxrans.solvers.preconditioners import chain_layout
        case = load_case({str(directory)!r}, {sets!r})
        grid = case.grid()
        slabs = SlabAtlas.of(grid, jax.devices())
        print("layout", type(chain_layout(slabs)).__name__)
        step, state = make_stepper(case, grid)
        step4, _ = make_stepper(case, slabs)
        state = state._replace(U=state.U * (1.0 + 1e-3 * jnp.sin(jnp.arange(state.U.size).reshape(state.U.shape))))
        split = slab_shardings(state, jax.devices())
        R1 = jax.jit(make_residual(grid, case.flow, case.numerics).residual)(state.U)
        R4 = jax.jit(make_residual(slabs, case.flow, case.numerics).residual,
                     in_shardings=(split.U,))(jax.device_put(state.U, split.U))
        one, four = make_block(step, 20), make_block(step4, 20, shardings=split)
        a, b = state, jax.device_put(state, split)
        for _ in range(2):
            a, b = one(a)[0], four(b)[0]
        np.savez("out.npz", R1=R1, R4=R4, U1=a.U, U4=b.U)

        sets = {sets!r} + ("numerical.convergence.max_iters=40", "numerical.convergence.forces_tol=null")
        r4 = run(load_case({str(directory)!r}, sets), devices=jax.devices())
        assert r4.timing.devices == 4 and np.isfinite([r4.forces.cl, r4.forces.cd]).all()
    """)
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=4"}
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-3000:]
    expected = "ChainLayout" if name in ("2d", "3d_reindexed") else "SliceChains"
    assert f"layout {expected}" in done.stdout, done.stdout
    out = np.load(tmp_path / "out.npz")
    assert np.max(np.abs(out["R4"] - out["R1"])) <= 1e-12 * np.max(np.abs(out["R1"]))
    # 40 implicit steps amplify the round-off: measured 1e-13 (M6), 1e-10 (the C-grid).
    assert np.max(np.abs(out["U4"] - out["U1"])) <= 1e-9 * np.max(np.abs(out["U1"]))


def test_an_uneven_split_is_refused() -> None:
    import jax

    atlas = load_case(ROOT / "examples/onera_m6_sa").grid()          # canvas first axis 20
    with pytest.raises(ValueError, match="not a multiple of"):
        SlabAtlas.of(atlas, [jax.devices()[0]] * 3)
