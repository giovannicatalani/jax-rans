"""A 3D structured grid file to a mesh: blocks, their interfaces and patches.

The 3D counterpart of mesh/assemble.py, for any number of blocks, assembled
from the geometry:

1. every block made right-handed (``k`` reversed if its volumes are negative);
2. boundary faces matched into interfaces where their four corners coincide,
   within ``pair_tol`` of the smaller of the face's shortest edge and its
   cell's width across it, in any of the eight relative orientations of two
   quadrilaterals -- between blocks, or within one (an O seam) -- never
   across a face the file declares a boundary; a file's 1-to-1 interfaces
   must agree. The matched cells of a side are cut into ``InterfaceRect``s,
   one per region with a single affine map to the partner (mesh/block3d.py);
3. a side collapsed to a line (every face of zero area) is a **pole**: an
   interface with itself across the axis, half a turn round the periodic
   running direction; that direction must be an O seam of the block;
4. every other face named: by the file (CGNS families and BC types), else by
   the case's patch rules (``PatchRule``, boxes in ``x, y, z``), first match
   wins; a side whose faces get several names is cut into rectangles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .assemble import FORMATS, MeshInfo, PatchRule
from .block import Grid, metrics, validate_metrics
from .block3d import SIDES_3D, BoundaryRect, InterfaceRect, Sides3D, normal_axis, running_axes
from .cgns import RawBlock3D, is_3d_cgns, is_cgns, read_cgns
from .mesh import Mesh
from .plot3d import read_plot3d_3d

CORNERS = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
"""A face's corners in its running indices, round it in order."""

ORIENTATIONS = np.array([[(s * k + r) % 4 for k in range(4)] for s in (1, -1) for r in range(4)])
"""The partner corner facing each of my corners, for the four rotations and
four reflections of a quadrilateral."""


def is_3d(path: str | Path, format: str = "auto") -> bool:
    """True for a grid file of 3D blocks that are not a thin extrusion of a
    2D grid (more than two nodes in every direction)."""
    if format == "cgns" or (format == "auto" and is_cgns(Path(path))):
        return is_3d_cgns(path)
    try:
        blocks = read_plot3d_3d(path)
    except ValueError:
        return False
    return all(min(b.shape[1:]) > 2 for b in blocks)


def load_grid_3d(path: str | Path, format: str = "auto", coordinate_scale: float = 1.0,
                 rules: tuple[PatchRule, ...] = (), dtype=np.float64,
                 pair_tol: float = 0.05) -> tuple[Mesh, MeshInfo]:
    """Read and assemble a 3D Plot3D or CGNS file."""
    path = Path(path)
    if format not in FORMATS:
        raise ValueError(f"unknown grid format {format!r}; choose from {FORMATS}")
    if format == "cgns" or (format == "auto" and is_cgns(path)):
        raw = read_cgns(path, dim=3)
    else:
        raw = [RawBlock3D(nodes) for nodes in read_plot3d_3d(path)]
    raw = [RawBlock3D(coordinate_scale * b.nodes, b.declared, b.name) for b in raw]
    return assemble_3d(raw, rules, dtype, pair_tol, source=str(path))


# --- Assembly ------------------------------------------------------------------------


def _right_handed(block: RawBlock3D) -> RawBlock3D:
    """``k`` reversed (and the declared faces with it) if the volumes are negative."""
    _, volume, _ = metrics(block.nodes)
    if np.all(volume > 0.0):
        return block
    if not np.all(volume < 0.0):
        raise ValueError(f"block {block.name or ''} has {int(np.sum(volume <= 0.0))} cells of non-positive "
                         f"volume among positive ones")
    nk = volume.shape[2]
    swap = {"k_min": "k_max", "k_max": "k_min"}
    declared = []
    for d in block.declared:
        start, stop = d.start, d.stop
        if not d.side.startswith("k"):                          # k is the second running index
            start, stop = (start[0], nk - stop[1]), (stop[0], nk - start[1])
        declared.append(type(d)(swap.get(d.side, d.side), start, stop, d.name, d.donor))
    return RawBlock3D(block.nodes[..., ::-1].copy(), tuple(declared), block.name)


def _plane(nodes: np.ndarray, side: str, layer: int = 0) -> np.ndarray:
    """Nodes ``layer`` planes inward from a side, ``(3, na + 1, nb + 1)`` in its running order."""
    n = normal_axis(side)
    last = nodes.shape[1 + n] - 1
    return np.take(nodes, layer if side.endswith("min") else last - layer, axis=1 + n)


def _quads(plane: np.ndarray) -> np.ndarray:
    """The faces of a plane of nodes, ``(na, nb, 4, 3)``, corners in ``CORNERS`` order."""
    corners = [plane[:, a:a + plane.shape[1] - 1, b:b + plane.shape[2] - 1] for a, b in CORNERS]
    return np.moveaxis(np.stack(corners), (0, 1), (2, 3))


class _Faces:
    """Every face of every block side, flattened: ``block``, ``side`` (index
    into ``SIDES_3D``), running position ``cell (n, 2)``, ``corners (n, 4, 3)``,
    the coincidence tolerance ``tol`` and whether the face has zero area."""

    def __init__(self, blocks: list[np.ndarray], pair_tol: float):
        parts = {k: [] for k in ("block", "side", "cell", "corners", "tol", "collapsed")}
        for b, nodes in enumerate(blocks):
            for s, side in enumerate(SIDES_3D):
                quads = _quads(_plane(nodes, side))
                inner = _quads(_plane(nodes, side, 1)).mean(axis=2)
                na, nb = quads.shape[:2]
                edges = np.linalg.norm(quads - np.roll(quads, -1, axis=2), axis=-1)
                area = 0.5 * np.linalg.norm(np.cross(quads[:, :, 2] - quads[:, :, 0],
                                                     quads[:, :, 3] - quads[:, :, 1]), axis=-1)
                width = np.linalg.norm(quads.mean(axis=2) - inner, axis=-1)
                a, c = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
                parts["block"].append(np.full(na * nb, b))
                parts["side"].append(np.full(na * nb, s))
                parts["cell"].append(np.stack([a.ravel(), c.ravel()], axis=1))
                parts["corners"].append(quads.reshape(-1, 4, 3))
                shortest = np.where(edges > 0.0, edges, np.inf).min(axis=-1)
                parts["tol"].append(pair_tol * np.minimum(shortest, width).ravel())
                parts["collapsed"].append((area <= 1e-10 * edges.max(axis=-1) ** 2).ravel())
        for key, values in parts.items():
            setattr(self, key, np.concatenate(values))

    def rows(self, block: int, side: str) -> np.ndarray:
        """Rows of one side, ``(na, nb)`` in its running order."""
        return np.flatnonzero((self.block == block) & (self.side == SIDES_3D.index(side)))


def _match(faces: _Faces, excluded: np.ndarray):
    """For every face, the coincident face (-1: none), the orientation (row
    of ``ORIENTATIONS``) taking my corners to its corners, and the largest
    corner distance of any pair. ``excluded`` faces match nothing."""
    centre = faces.corners.mean(axis=1)
    candidates = cKDTree(centre).query_ball_point(centre, r=faces.tol)
    me = np.repeat(np.arange(len(centre)), [len(c) for c in candidates])
    other = np.concatenate([np.asarray(c, dtype=np.int64) for c in candidates]) if len(me) else me
    keep = (other != me) & ~excluded[me] & ~excluded[other]
    me, other = me[keep], other[keep]
    # distance[p, o] = largest corner distance under orientation o
    gap = np.linalg.norm(faces.corners[me][:, None, :, :] - faces.corners[other][:, ORIENTATIONS], axis=-1)
    distance = gap.max(axis=-1)
    orientation = distance.argmin(axis=1)
    distance = distance[np.arange(len(me)), orientation]
    hit = distance <= faces.tol[me]
    me, other, orientation, distance = me[hit], other[hit], orientation[hit], distance[hit]
    if np.unique(me).size != me.size:
        raise ValueError("a face coincides with more than one other face")
    partner = np.full(len(centre), -1)
    turn = np.zeros(len(centre), dtype=np.int64)
    partner[me], turn[me] = other, orientation
    return partner, turn, float(distance.max(initial=0.0))


def _interface_label(faces: _Faces, row: int, partner: int, turn: int) -> tuple:
    """The affine map of face ``row`` to its partner, as a hashable label
    constant over an ``InterfaceRect``: ``(block, side, axes, flip, c)``
    with partner running position ``p[m] = c[m] + s_m a[axes[m]]``."""
    sigma = ORIENTATIONS[turn]
    p0, pa, pb = CORNERS[sigma[0]], CORNERS[sigma[1]], CORNERS[sigma[3]]
    axes, flip = [0, 0], [False, False]
    for mine, step in ((0, pa - p0), (1, pb - p0)):
        m = int(np.flatnonzero(step)[0])
        axes[m], flip[m] = mine, bool(step[m] < 0)
    a = faces.cell[row]
    p = faces.cell[partner]
    c = tuple(int(p[m] + (a[axes[m]] if flip[m] else -a[axes[m]])) for m in range(2))
    return (int(faces.block[partner]), SIDES_3D[faces.side[partner]], tuple(axes), tuple(flip), c)


def _rectangles(labels: np.ndarray) -> tuple:
    """Rectangles of equal label tiling a side: runs along the second running
    index, merged across consecutive rows of the first where identical. A
    label is a patch name or an ``_interface_label``."""
    rows = []
    for a in range(labels.shape[0]):
        runs, start = [], 0
        for b in range(1, labels.shape[1] + 1):
            if b == labels.shape[1] or labels[a, b] != labels[a, start]:
                runs.append((start, b, labels[a, start]))
                start = b
        rows.append(runs)
    out, open_ = [], {}
    for a, runs in enumerate(rows + [[]]):
        keys = set(runs)
        for key, first in list(open_.items()):
            if key not in keys:
                out.append(_rect(key[2], (first, key[0]), (a, key[1])))
                del open_[key]
        for key in runs:
            open_.setdefault(key, a)
    return tuple(sorted(out, key=lambda r: (r.start, r.stop)))


def _rect(label, start: tuple[int, int], stop: tuple[int, int]):
    if isinstance(label, str):
        return BoundaryRect(label, start, stop)
    block, side, axes, flip, c = label
    origin = tuple(c[m] + (-start[axes[m]] if flip[m] else start[axes[m]]) for m in range(2))
    return InterfaceRect(start, stop, block, side, origin, axes, flip)


def _pole(block: int, side: str, shape, sides: dict) -> tuple[InterfaceRect]:
    """A collapsed side: the interface with itself half a turn round its
    periodic running direction (the axis of an O seam of the block)."""
    extent = tuple(shape[x] for x in running_axes(side))
    periodic = [m for m, axis in enumerate(running_axes(side))
                if any(isinstance(r, InterfaceRect) and r.block == block and normal_axis(s) == axis
                       and normal_axis(r.side) == axis for s in sides for r in sides[s])]
    if len(periodic) != 1 or extent[periodic[0]] % 2:
        raise ValueError(f"block {block}: side {side} is collapsed but not round one periodic direction "
                         f"of even extent")
    m = periodic[0]
    origin, period = [0, 0], [0, 0]
    origin[m], period[m] = extent[m] // 2, extent[m]
    return (InterfaceRect((0, 0), extent, block, side, tuple(origin), period=tuple(period)),)


def assemble_3d(raw: list[RawBlock3D], rules: tuple[PatchRule, ...] = (), dtype=np.float64,
                pair_tol: float = 0.05, source: str = "") -> tuple[Mesh, MeshInfo]:
    """3D blocks as read -> a mesh with interfaces and named patches."""
    raw = [_right_handed(RawBlock3D(np.asarray(b.nodes, dtype=np.float64), b.declared, b.name)) for b in raw]
    blocks = [b.nodes for b in raw]
    faces = _Faces(blocks, pair_tol)

    declared_name = np.full(len(faces.block), None, dtype=object)
    donors = []
    for b, block in enumerate(raw):
        for d in block.declared:
            n = tuple(blocks[b].shape[1 + x] - 1 for x in running_axes(d.side))
            rows = faces.rows(b, d.side).reshape(n)[d.start[0]:d.stop[0], d.start[1]:d.stop[1]].ravel()
            if d.name is not None:
                declared_name[rows] = d.name
            elif d.donor is not None:
                donors.append((rows, d.donor))
    partner, turn, mismatch = _match(faces, (declared_name != None) | faces.collapsed)  # noqa: E711
    for rows, donor in donors:
        if np.any(partner[rows] < 0) or np.any(faces.block[partner[rows]] != donor):
            raise ValueError(f"a 1-to-1 interface the file declares is not matched by coincident faces "
                             f"to its donor zone {donor}")

    names = declared_name.copy()
    mid = faces.corners.mean(axis=1)
    for rule in rules:
        take = (names == None) & (partner < 0) & rule.contains(*mid.T)  # noqa: E711
        names[take] = rule.name

    mesh = []
    for b, nodes in enumerate(blocks):
        shape = tuple(n - 1 for n in nodes.shape[1:])
        sides, poles = {}, []
        for side in SIDES_3D:
            rows = faces.rows(b, side)
            if np.all(faces.collapsed[rows]):
                poles.append(side)
                continue
            labels = np.empty(len(rows), dtype=object)
            for k, r in enumerate(rows):
                if partner[r] >= 0:
                    labels[k] = _interface_label(faces, r, partner[r], turn[r])
                elif names[r] is not None:
                    labels[k] = names[r]
                else:
                    raise ValueError(f"block {b}: faces of {side} match no patch rule and no other face; "
                                     f"add a rule without bounds (`domain.patches`)")
            sides[side] = _rectangles(labels.reshape(tuple(shape[x] for x in running_axes(side))))
        for side in poles:
            sides[side] = _pole(b, side, shape, sides)
        grid = Grid.from_node_array(nodes, Sides3D(shape, *(sides[s] for s in SIDES_3D)), dtype)
        report = validate_metrics(grid)
        if not (report.min_area > 0.0 and report.max_normal_closure < 1e-12):
            raise ValueError(f"block {b} fails the metric checks: {report.summary()}")
        mesh.append(grid)
    mesh = tuple(mesh)
    return mesh, _info(mesh, source, mismatch)


def _info(mesh: Mesh, source: str, mismatch: float) -> MeshInfo:
    counts: dict[str, int] = {}
    interfaces, spacing = 0, np.inf
    for grid in mesh:
        nodes = np.asarray(grid.nodes)
        for side in SIDES_3D:
            for r in grid.sides.side(side):
                if isinstance(r, InterfaceRect):
                    interfaces += 1
                    continue
                counts[r.patch] = counts.get(r.patch, 0) + (r.stop[0] - r.start[0]) * (r.stop[1] - r.start[1])
                if r.patch == "wall":
                    window = (slice(None), slice(r.start[0], r.stop[0] + 1), slice(r.start[1], r.stop[1] + 1))
                    h = _plane(nodes, side, 1)[window] - _plane(nodes, side)[window]
                    spacing = min(spacing, float(np.linalg.norm(h, axis=0).min()))
    radius = max(float(np.linalg.norm(np.asarray(g.nodes), axis=0).max()) for g in mesh)
    return MeshInfo(source, len(mesh), sum(int(np.prod(g.shape)) for g in mesh), interfaces, counts,
                    int("wall" in counts), radius, spacing, mismatch)
