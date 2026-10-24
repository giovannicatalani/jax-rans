"""Split-block regression in 3D: the gate for multi-block 3D grids.

The ONERA M6 example grid cut into blocks, each block re-indexed (its index
directions permuted and reversed), then re-assembled from the geometry
alone, is the same discretization written differently: every face sees the
same two cells. Its residual, time step, wall distance and wall loads must
be the single block's, cell for cell. The cuts make four blocks meet along
edges, run across the pole and the O seam, and the re-indexing makes
interfaces that permute index directions, whose edge ghosts need the fill's
repeated passes (mesh/atlas3d.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxaero.core.config import FlowConditions, ImplicitConfig, NumericsConfig
from jaxaero.mesh.assemble3d import assemble_3d, load_grid_3d
from jaxaero.mesh.atlas import build_atlas
from jaxaero.mesh.cgns import RawBlock3D
from jaxaero.mesh.mesh import assign_conditions, wall_distance
from jaxaero.mesh.plot3d import read_plot3d_3d
from jaxaero.numerics.residual import make_residual
from jaxaero.post.forces import ForceReference, integrate_forces, surface
from jaxaero.solvers import implicit

from ..unit.test_assemble3d import M6, M6_RULES, ORIENTATIONS, check_interfaces, cut, reindex

FLOWS = {
    "euler": FlowConditions(0.84, 3.06),
    "laminar": FlowConditions(0.5, 3.0, 5000.0),
    "sa": FlowConditions(0.84, 3.06, 14.6e6, sutherland_ratio=198.6 / 540.0, turbulence="sa_noft2"),
}
REFERENCE = ForceReference(0.8, 1.15, (0.0, 0.0, 0.0))


def kinds(flow: str) -> dict[str, str]:
    wall = "slip_wall" if FLOWS[flow].inviscid else "no_slip_wall"
    return {"wall": wall, "farfield": "far_field", "symmetry": "symmetry"}


def cell_ids(cuts, seed: int):
    """``(mesh, ids)``: the M6 grid cut at ``cuts``, each block re-indexed at
    random and re-assembled; ``ids[b]`` the original flat index of each cell
    of block ``b``."""
    (nodes,) = read_plot3d_3d(M6)
    shape = tuple(n - 1 for n in nodes.shape[1:])
    rng = np.random.default_rng(seed)
    raw, ids = [], []
    original = np.arange(np.prod(shape)).reshape(shape)
    edges = [(0, *c, n) for c, n in zip(cuts, shape)]
    windows = [(slice(i0, i1), slice(j0, j1), slice(k0, k1))
               for i0, i1 in zip(edges[0], edges[0][1:])
               for j0, j1 in zip(edges[1], edges[1][1:])
               for k0, k1 in zip(edges[2], edges[2][1:])]
    for block, window in zip(cut(nodes, cuts), windows):
        perm, flips = ORIENTATIONS[rng.integers(len(ORIENTATIONS))]
        raw.append(RawBlock3D(reindex(block, perm, flips)))
        ids.append(reindex(original[window][None], perm, flips)[0])
    mesh, _ = assemble_3d(raw, M6_RULES)
    for b, (block, given) in enumerate(zip(mesh, raw)):
        if not np.array_equal(np.asarray(block.nodes), given.nodes):     # made right-handed: k reversed
            assert np.array_equal(np.asarray(block.nodes), given.nodes[..., ::-1])
            ids[b] = ids[b][..., ::-1]
    return mesh, ids


CUTS = ((5, 11), (22,), ())          # span (through the pole's block), wall-normal; four blocks per edge


@pytest.fixture(scope="module")
def grids():
    (single,), _ = load_grid_3d(M6, rules=M6_RULES)
    mesh, ids = cell_ids(CUTS, seed=2)
    return single, mesh, ids


@pytest.fixture(scope="module")
def atlases(grids):
    built = {}

    def make(flow: str):
        key = FLOWS[flow].inviscid
        if key not in built:
            single, mesh, _ = grids
            built[key] = (build_atlas(assign_conditions(single, kinds(flow))),
                          build_atlas(assign_conditions(mesh, kinds(flow))))
        return built[key]

    return make


def to_blocks(field, atlas_single, atlas_multi, ids):
    """A single-block canvas field on the multi-block canvas."""
    (block,) = atlas_single.unpack(field)
    flat = block.reshape(block.shape[0], -1)
    return atlas_multi.pack([flat[:, i.ravel()].reshape((flat.shape[0],) + i.shape) for i in ids])


def to_single(field, atlas_multi, ids, n_cells):
    """Multi-block canvas field ``(nvar, ...)`` -> per original cell ``(nvar, n_cells)``."""
    out = np.full((field.shape[0], n_cells), np.nan)
    for block, i in zip(atlas_multi.unpack(field), ids):
        out[:, i.ravel()] = np.asarray(block).reshape(block.shape[0], -1)
    return out


def perturbed(operator, atlas, seed=1):
    U = operator.initial_state()
    rng = np.random.default_rng(seed)
    noise = jnp.asarray(rng.standard_normal(U.shape))
    return jnp.where(atlas.interior, U * (1.0 + 0.02 * noise), U)


def test_the_blocks_tile_the_grid_and_the_fill_needs_more_passes(grids, atlases) -> None:
    single, mesh, ids = grids
    assert sum(int(np.prod(b.shape)) for b in mesh) == int(np.prod(single.shape))
    assert np.array_equal(np.sort(np.concatenate([i.ravel() for i in ids])), np.arange(np.prod(single.shape)))
    assert check_interfaces(mesh) > 0
    one, many = atlases("euler")
    assert len(one.stages) == 3 and len(many.stages) in (6, 9)


def test_wall_distance_is_the_single_block_one(grids) -> None:
    """To round-off of the coordinates (chord 1): a re-indexed block lists a
    wall face's corners in another order, and the patch distance's Newton
    iterates (mesh/wall3d.py) differ there; measured 2e-14."""
    single, mesh, ids = grids
    (want,) = wall_distance(assign_conditions(single, kinds("sa")))
    got = np.full(want.size, np.nan)
    for d, i in zip(wall_distance(assign_conditions(mesh, kinds("sa"))), ids):
        got[i.ravel()] = d.ravel()
    np.testing.assert_allclose(got, want.ravel(), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("flow,first_order,kappa", [
    ("euler", True, 0.0), ("sa", False, 0.0),
    pytest.param("euler", False, 0.0, marks=pytest.mark.slow),
    pytest.param("laminar", False, 0.0, marks=pytest.mark.slow),
    pytest.param("sa", True, 0.0, marks=pytest.mark.slow),
    pytest.param("sa", False, 1.0 / 3.0, marks=pytest.mark.slow),
])
def test_the_residual_is_the_single_block_one(grids, atlases, flow, first_order, kappa) -> None:
    single, mesh, ids = grids
    one, many = atlases(flow)
    numerics = NumericsConfig(limiter="none" if kappa else "van_albada", kappa=kappa)
    a = make_residual(one, FLOWS[flow], numerics, first_order=first_order)
    b = make_residual(many, FLOWS[flow], numerics, first_order=first_order)
    U = perturbed(a, one)
    want = to_single(np.asarray(jax.jit(a.residual)(U)), one, [np.arange(np.prod(single.shape)).reshape(single.shape)],
                     np.prod(single.shape))
    Ub = jnp.where(many.interior, to_blocks(U, one, many, ids), b.initial_state())
    got = to_single(np.asarray(jax.jit(b.residual)(Ub)), many, ids, np.prod(single.shape))
    scale = np.abs(want).max(axis=1, keepdims=True)
    assert np.max(np.abs(got - want) / scale) < 1e-12, np.max(np.abs(got - want) / scale)


def test_norm_and_time_step_are_the_single_block_ones(grids, atlases) -> None:
    """The time step to 1e-10: a wall cell's volume, ``(1/3) sum S . x_f``,
    loses digits to cancellation (cells 1e-5 high at coordinates ~1), and a
    re-indexed block sums its faces in another order; measured 3.5e-12."""
    single, mesh, ids = grids
    one, many = atlases("sa")
    numerics = NumericsConfig(limiter="van_albada")
    a, b = make_residual(one, FLOWS["sa"], numerics), make_residual(many, FLOWS["sa"], numerics)
    U = perturbed(a, one)
    Ub = jnp.where(many.interior, to_blocks(U, one, many, ids), b.initial_state())
    assert float(b.norm(b.residual(Ub))) == pytest.approx(float(a.norm(a.residual(U))), rel=1e-12)
    n = np.prod(single.shape)
    want = to_single(np.asarray(a.time_step(U))[None], one, [np.arange(n).reshape(single.shape)], n)
    np.testing.assert_allclose(to_single(np.asarray(b.time_step(Ub))[None], many, ids, n), want, rtol=1e-10)


def test_forces_and_surface_are_the_single_block_ones(grids, atlases) -> None:
    single, mesh, ids = grids
    one, many = atlases("sa")
    a = make_residual(one, FLOWS["sa"], NumericsConfig())
    U = perturbed(a, one)
    Ub = to_blocks(U, one, many, ids)
    want, got = (integrate_forces(*args, FLOWS["sa"], REFERENCE) for args in ((U, one), (Ub, many)))
    for name in ("cl", "cd", "cm", "cd_friction", "cy"):
        assert getattr(got, name) == pytest.approx(getattr(want, name), rel=1e-12, abs=1e-14), name
    sa, sb = surface(U, one, FLOWS["sa"]), surface(Ub, many, FLOWS["sa"])
    order = lambda s: np.lexsort(tuple(np.round(s[c], 9) for c in "zyx"))  # noqa: E731  (ties to round-off)
    # Skin friction to 1e-10: a velocity difference over the first cell's
    # height (1e-5 at coordinates ~1), round-off amplified as in the volumes.
    for key, rtol in (("x", 1e-12), ("y", 1e-12), ("z", 1e-12), ("cp", 1e-12),
                      ("cf_x", 1e-10), ("cf_y", 1e-10), ("cf_z", 1e-10)):
        np.testing.assert_allclose(sb[key][order(sb)], sa[key][order(sa)], rtol=rtol, atol=1e-14, err_msg=key)


@pytest.mark.slow
def test_an_implicit_step_is_the_single_block_one_when_the_lines_are(atlases) -> None:
    """Cut along the span only, every block's lines run wall to far field as
    the single block's: the preconditioner and so the whole step agree."""
    (single,), _ = load_grid_3d(M6, rules=M6_RULES)
    mesh, ids = cell_ids(((5, 11), (), ()), seed=4)
    flow = FLOWS["sa"]
    numerics = NumericsConfig(limiter="van_albada", solver="implicit")
    config = ImplicitConfig(cfl=10.0)
    steps = []
    for grid in (single, mesh):
        atlas = build_atlas(assign_conditions(grid, kinds("sa")))
        operator = make_residual(atlas, flow, numerics)
        jacobian = make_residual(atlas, flow, numerics, first_order=True, distance=operator.distance)
        steps.append((atlas, operator, jax.jit(implicit.make_newton_step(operator, config, jacobian))))
    (one, a, step_a), (many, b, step_b) = steps
    U = perturbed(a, one)
    state_a = implicit.initial_state(a, config)._replace(U=U)
    state_b = implicit.initial_state(b, config)
    state_b = state_b._replace(U=jnp.where(many.interior, to_blocks(U, one, many, ids), state_b.U))
    (new_a, norm_a), (new_b, norm_b) = step_a(state_a), step_b(state_b)
    assert float(norm_b) == pytest.approx(float(norm_a), rel=1e-12)
    n = np.prod(single.shape)
    want = to_single(np.asarray(new_a.U), one, [np.arange(n).reshape(single.shape)], n)
    got = to_single(np.asarray(new_b.U), many, ids, n)
    assert np.max(np.abs(got - want) / np.abs(want).max(axis=1, keepdims=True)) < 1e-9
