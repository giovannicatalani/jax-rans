"""Multi-block 3D assembly from the geometry (mesh/assemble3d.py) and 3D CGNS
zones (mesh/cgns.py): interfaces in any orientation, partial faces, declared
boundaries, poles and seams as before."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from jaxrans.mesh.assemble import PatchRule
from jaxrans.mesh.assemble3d import _plane, _quads, assemble_3d, is_3d, load_grid_3d
from jaxrans.mesh.block3d import SIDES_3D, BoundaryRect, InterfaceRect
from jaxrans.mesh.cgns import Declared, RawBlock3D, read_cgns
from jaxrans.mesh.plot3d import read_plot3d_3d

from .test_grid3d import warped_box
from .test_grid_files import write_cgns

ROOT = Path(__file__).resolve().parents[2]
M6 = ROOT / "examples/onera_m6_sa/om6_strct_16x44x24.ufmt"
CRM = ROOT / "data/grids/crm_adodg/crm_450k_hdf5.cgns"
M6_RULES = (PatchRule("symmetry", y_max=1e-9),
            PatchRule("wall", x_min=-1.0, x_max=3.0, y_max=3.0, z_min=-1.0, z_max=1.0),
            PatchRule("farfield"))
ORIENTATIONS = [(perm, flips) for perm in itertools.permutations(range(3))
                for flips in itertools.product((False, True), repeat=3)]


def reindex(nodes: np.ndarray, perm, flips) -> np.ndarray:
    """The same block with its index directions permuted and reversed."""
    out = np.transpose(nodes, (0, *(1 + p for p in perm)))
    for axis, flip in enumerate(flips):
        if flip:
            out = np.flip(out, axis=1 + axis)
    return np.ascontiguousarray(out)


def cut(nodes: np.ndarray, cuts) -> list[np.ndarray]:
    """Blocks of ``nodes`` cut at the cell lines ``cuts`` (one tuple per index)."""
    edges = [(0, *c, nodes.shape[1 + a] - 1) for a, c in enumerate(cuts)]
    return [nodes[:, i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]
            for i0, i1 in zip(edges[0], edges[0][1:])
            for j0, j1 in zip(edges[1], edges[1][1:])
            for k0, k1 in zip(edges[2], edges[2][1:])]


def face_centre(grid, side: str, a) -> np.ndarray:
    return _quads(_plane(np.asarray(grid.nodes), side))[a[0], a[1]].mean(axis=0)


def check_interfaces(mesh) -> int:
    """Every interface cell faces the partner face of the same centroid, and
    the partner's interface points back. Returns the interface faces seen."""
    seen = 0
    for b, grid in enumerate(mesh):
        for side in SIDES_3D:
            for r in grid.sides.side(side):
                if not isinstance(r, InterfaceRect) or r.period != (0, 0):
                    continue
                other = mesh[r.block]
                for a in itertools.product(range(r.start[0], r.stop[0]), range(r.start[1], r.stop[1])):
                    p = r.partner(a)
                    np.testing.assert_allclose(face_centre(grid, side, a), face_centre(other, r.side, p),
                                               atol=1e-12)
                    back = [q for q in other.sides.side(r.side) if isinstance(q, InterfaceRect)
                            and all(q.start[m] <= p[m] < q.stop[m] for m in range(2))]
                    assert len(back) == 1 and back[0].block == b and back[0].side == side
                    assert back[0].partner(p) == a
                    seen += 1
    return seen


def test_blocks_in_every_orientation_reassemble_with_mutual_interfaces() -> None:
    nodes = warped_box((4, 6, 5), amplitude=0.1, seed=3)
    rng = np.random.default_rng(1)
    picks = rng.choice(len(ORIENTATIONS), 8, replace=False)
    raw = [RawBlock3D(reindex(b, *ORIENTATIONS[k])) for b, k in zip(cut(nodes, ((2,), (3,), (2,))), picks)]
    mesh, info = assemble_3d(raw, (PatchRule("outer"),))
    assert info.blocks == 8 and info.cells == 4 * 6 * 5 and info.interface_mismatch == 0.0
    assert info.patch_faces == {"outer": 2 * (4 * 6 + 4 * 5 + 6 * 5)}
    # 3 internal cut planes: 6x5 + 4x5 + 4x6 faces, seen from both sides
    assert check_interfaces(mesh) == 2 * (6 * 5 + 4 * 5 + 4 * 6)
    flips = {r.flip for g in mesh for s in SIDES_3D for r in g.sides.side(s) if isinstance(r, InterfaceRect)}
    axes = {r.axes for g in mesh for s in SIDES_3D for r in g.sides.side(s) if isinstance(r, InterfaceRect)}
    assert len(flips) > 1 and len(axes) == 2                  # orientations really vary


def test_a_face_against_two_blocks_is_cut_into_two_interfaces() -> None:
    nodes = warped_box((4, 4, 3), amplitude=0.1, seed=4)
    bottom = nodes[:, :, :3]
    left, right = nodes[:, :3, 2:], nodes[:, 2:, 2:]
    mesh, info = assemble_3d([RawBlock3D(bottom), RawBlock3D(left), RawBlock3D(reindex(right, (2, 0, 1), (1, 0, 0)))],
                             (PatchRule("outer"),))
    assert [r.block for r in mesh[0].sides.j_max] == [1, 2]
    assert check_interfaces(mesh) == 2 * (4 * 3 + 2 * 3)        # bottom-top, left-right


def test_declared_boundaries_never_match_and_left_handed_blocks_keep_them() -> None:
    nodes = warped_box((3, 4, 4), amplitude=0.05, seed=5)
    below, above = nodes[..., :3], nodes[..., 2:]
    # 'above' given left-handed (k reversed): the shared face is its k_max as
    # read, its k_min once reversed; a rectangle declared on i_min at k = 0
    # moves to k = 1 with it
    flipped = np.ascontiguousarray(above[..., ::-1])
    raw = [RawBlock3D(below, (Declared("k_max", (0, 1), (3, 2), name="plate"),)),
           RawBlock3D(flipped, (Declared("k_max", (0, 1), (3, 2), name="plate"),
                                Declared("i_min", (0, 0), (4, 1), name="inlet")))]
    mesh, info = assemble_3d(raw, (PatchRule("outer"),))
    assert info.patch_faces["plate"] == 6 and info.patch_faces["inlet"] == 4
    (inlet,) = [r for r in mesh[1].sides.i_min if isinstance(r, BoundaryRect) and r.patch == "inlet"]
    assert (inlet.start, inlet.stop) == ((0, 1), (4, 2))
    shared = [r for r in mesh[0].sides.k_max if isinstance(r, InterfaceRect)]
    assert {r.block for r in shared} == {1}
    assert sum((r.stop[0] - r.start[0]) * (r.stop[1] - r.start[1]) for r in shared) == 3 * 4 - 3
    assert check_interfaces(mesh) == 2 * 9


def test_a_declared_interface_must_be_found() -> None:
    nodes = warped_box((2, 2, 4), amplitude=0.0)
    gap = nodes[:, :, :, 2:] + np.array([0.0, 0.0, 0.5])[:, None, None, None]
    raw = [RawBlock3D(nodes[:, :, :, :3], (Declared("k_max", (0, 0), (2, 2), donor=1),)), RawBlock3D(gap)]
    with pytest.raises(ValueError, match="not matched"):
        assemble_3d(raw, (PatchRule("outer"),))


def test_an_unnamed_face_is_an_error_naming_its_block() -> None:
    with pytest.raises(ValueError, match="block 0: faces of j_min"):
        assemble_3d([RawBlock3D(warped_box((2, 2, 2)))], (PatchRule("top", z_min=1e9),))


def test_3d_cgns_zones_read_with_their_boundaries(tmp_path) -> None:
    nodes = warped_box((3, 4, 4), amplitude=0.05, seed=6)
    a, b = nodes[..., :3], nodes[..., 2:]
    ni, nj, nk = a.shape[1:]
    bcs = [("w", "BCWallViscous", None, [1, 1, 1], [ni, nj, 1]),
           ("f", "BCFarfield", "outer", [1, 1, nk], [ni, nj, nk])]
    path = tmp_path / "two.cgns"
    write_cgns(path, [((*a,), bcs[:1]), ((*b,), bcs[1:])])
    assert is_3d(path)
    raw = read_cgns(path, dim=3)
    assert [d.side for d in raw[0].declared] == ["k_min"] and raw[0].declared[0].stop == (3, 4)
    mesh, info = load_grid_3d(path, rules=(PatchRule("sides"),))
    assert info.patch_faces == {"wall": 12, "outer": 12, "sides": 2 * 4 * (4 + 3)}
    assert [r.block for r in mesh[0].sides.k_max] == [1]


def test_the_m6_grid_assembles_alike_from_plot3d_and_cgns(tmp_path) -> None:
    (nodes,) = read_plot3d_3d(M6)
    path = tmp_path / "m6.cgns"
    write_cgns(path, [((*nodes,), [])])
    (p3d,), _ = load_grid_3d(M6, rules=M6_RULES)
    (cgns,), _ = load_grid_3d(path, rules=M6_RULES)
    assert p3d.sides == cgns.sides
    (pole,) = p3d.sides.i_min
    assert pole.side == "i_min" and pole.period == (0, 24) and pole.origin == (0, 12)
    assert p3d.sides.k_min[0].side == "k_max" and p3d.sides.k_max[0].side == "k_min"


@pytest.mark.skipif(not CRM.exists(), reason="CRM grid not downloaded (data/grids/README.md)")
def test_the_crm_wing_grid_assembles() -> None:
    mesh, info = load_grid_3d(CRM)
    assert info.blocks == 26 and info.cells == 450560 and info.interfaces == 98
    assert info.patch_faces == {"sym": 4800, "wall": 11264, "far": 11264}
    assert info.interface_mismatch < 1e-12
