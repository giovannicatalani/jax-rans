"""The atlas of a 3D mesh: every block with its ghost ring as a tile of one canvas.

Builds the same static plans as the 2D atlas (mesh/atlas.py), in NumPy, so
everything downstream -- the ghost fill, face averages, the line
preconditioner, wall loads -- runs on it unchanged:

* the tiles packed side by side in two canvas axes, as deep as the deepest
  along the third (the axis chosen to make the canvas smallest);
* the ghost fill as three gather stages, each block's sides in its
  ``fill_order``: first the pair normal to its wall (``j`` on the wing grids),
  then the other two pairs, each extended over the ghosts of the stages
  before it, so edge and corner ghosts come from the neighbour (or the
  condition) as in 2D. A ghost between the wall and another physical side
  is thus filled the same way whatever the block's index order (two plane
  reflections do not commute). Across an interface that permutes index
  directions, an extended ghost may read a partner ghost of the *same*
  stage, not yet filled: the three stages are then repeated until every
  read is of a filled cell (``_passes``; one pass on a single block);
* ghost centres: reflections through physical faces, copies across
  interfaces (the pole's ghost is the opposite cell, centre included);
* line chains along each block's wall-normal index (``line_axis``) for every
  pair of the other two;
* wall faces with their family, canvas face index and orientation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .atlas import Atlas, Lines, Stage, Walls, _pack, apply_reflections
from .block import WALL_KINDS, Grid, metrics
from .block3d import SIDES_3D, InterfaceRect, box_sides, normal_axis, running_axes
from .wall3d import wall_quads

MAX_PASSES = 4


def cell_on(side: str, a, layer: int, shape, inward: bool) -> tuple[int, int, int]:
    """Block cell ``layer`` cells inward from (or, ``inward=False``, outward
    beyond) ``side`` at running position ``a``."""
    n = normal_axis(side)
    out = [0, 0, 0]
    for m, axis in enumerate(running_axes(side)):
        out[axis] = a[m]
    low = side.endswith("min")
    if inward:
        out[n] = layer if low else shape[n] - 1 - layer
    else:
        out[n] = -1 - layer if low else shape[n] + layer
    return tuple(out)


def _extended_nodes(nodes: np.ndarray, g: int) -> np.ndarray:
    """Nodes extended by ``g`` linearly extrapolated layers on every side."""
    for axis in range(1, 4):
        a = np.moveaxis(nodes, axis, 0)
        lo = [a[0] - (k + 1) * (a[1] - a[0]) for k in range(g)][::-1]
        hi = [a[-1] + (k + 1) * (a[-1] - a[-2]) for k in range(g)]
        nodes = np.moveaxis(np.concatenate([np.stack(lo), a, np.stack(hi)]), 0, axis)
    return nodes


def _face_geometry(block: Grid, side: str):
    """Outward unit normals and midpoints ``(3, na, nb)`` of a side's faces."""
    n = normal_axis(side)
    index = 0 if side.endswith("min") else block.shape[n]
    S = np.take(np.asarray(block.faces[n]), index, axis=1 + n)
    nodes = np.take(np.asarray(block.nodes), index, axis=1 + n)
    length = np.linalg.norm(S, axis=0)
    sign = -1.0 if side.endswith("min") else 1.0
    normal = np.where(length > 0.0, sign * S / np.where(length > 0.0, length, 1.0), 0.0)
    mid = 0.25 * (nodes[:, :-1, :-1] + nodes[:, 1:, :-1] + nodes[:, :-1, 1:] + nodes[:, 1:, 1:])
    return normal, mid


def _pack_3d(tiles: list[tuple[int, int, int]]):
    """Origins of the tiles and the canvas shape: packed in the plane of two
    axes by ``atlas._pack``, as deep as the deepest along the third, for the
    choice of depth axis giving the smallest canvas."""
    best = None
    for depth in range(3):
        a, b = (x for x in range(3) if x != depth)
        flat, (na, nb) = _pack([(t[a], t[b]) for t in tiles])
        shape = [0, 0, 0]
        shape[a], shape[b], shape[depth] = na, nb, max(t[depth] for t in tiles)
        if best is None or np.prod(shape) < np.prod(best[1]):
            origins = []
            for x, y in flat:
                o = [0, 0, 0]
                o[a], o[b] = x, y
                origins.append(tuple(o))
            best = (origins, tuple(shape))
    return best


def line_axis(block: Grid) -> int:
    """The index direction of a block's line chains: the normal of its wall
    faces (of the most of them, if on several axes), else of its far-field
    faces (the outer blocks of a wall-to-far-field family); ``j`` without
    either."""
    for kinds in (WALL_KINDS, ("far_field",)):
        count = [0, 0, 0]
        for side, rect in block.sides.boundaries():
            if rect.kind in kinds:
                count[normal_axis(side)] += (rect.stop[0] - rect.start[0]) * (rect.stop[1] - rect.start[1])
        if max(count):
            return int(np.argmax(count))
    return 1


def fill_order(block: Grid) -> tuple[int, int, int]:
    """The order in which a block's side pairs are filled: its ``line_axis``,
    then the other two in ascending order (``j, i, k`` for a wall on ``j``)."""
    first = line_axis(block)
    return (first, *(a for a in range(3) if a != first))


def build_atlas_3d(mesh, g: int = 2) -> Atlas:
    """The atlas of a 3D mesh whose patches have conditions assigned."""
    unset = sorted({r.patch for block in mesh for _, r in block.sides.boundaries() if r.kind is None})
    if unset:
        raise ValueError(f"patch(es) {unset} have no boundary condition; use mesh.assign_conditions")
    origins, canvas = _pack_3d([tuple(n + 2 * g for n in block.shape) for block in mesh])
    NI, NJ, NK = canvas
    n_canvas = NI * NJ * NK

    def pos(b, i, j, k):
        oi, oj, ok = origins[b]
        return ((oi + g + i) * NJ + (oj + g + j)) * NK + (ok + g + k)

    windows = tuple(tuple(slice(o + g, o + g + n) for o, n in zip(origins[b], block.shape))
                    for b, block in enumerate(mesh))
    interior = np.zeros(canvas, dtype=bool)
    for window in windows:
        interior[window] = True

    # Geometry: each tile's nodes extended by extrapolated ghost layers, its
    # block's own metrics inside; gaps a lattice of unit cells, never used.
    lattice = np.stack(np.meshgrid(*(np.arange(n, dtype=np.float64) for n in canvas), indexing="ij"))
    centres, volume = 1e3 + 10.0 * lattice, np.ones(canvas)
    faces = [np.zeros((3,) + tuple(n + (a == x) for x, n in enumerate(canvas))) for a in range(3)]
    for block, origin, window in zip(mesh, origins, windows):
        tile_centres, tile_volume, tile_faces = metrics(_extended_nodes(np.asarray(block.nodes, dtype=np.float64), g))
        tile = tuple(slice(o, o + n) for o, n in zip(origin, tile_volume.shape))
        centres[(slice(None),) + tile] = tile_centres
        volume[tile] = np.where(np.abs(tile_volume) > 1e-300, tile_volume, 1.0)
        volume[window] = np.asarray(block.volume)
        for a, f in enumerate(tile_faces):
            faces[a][(slice(None),) + tuple(slice(o, o + n) for o, n in zip(origin, f.shape[1:]))] = f

    plans = {layers: [_stage(mesh, layers, stage, pos, n_canvas) for stage in range(3)] for layers in (g, 1)}
    passes = {layers: _passes(plan, interior.ravel()) for layers, plan in plans.items()}
    flat_centres = centres.reshape(3, -1)
    for stage in plans[1] * passes[1]:
        flat_centres = apply_reflections(stage, flat_centres)
    centres = flat_centres.reshape((3,) + canvas)
    grid = Grid(jnp.zeros((3,) + tuple(n + 1 for n in canvas)), jnp.asarray(centres), jnp.asarray(volume),
                tuple(jnp.asarray(f) for f in faces), box_sides(canvas, **dict.fromkeys(SIDES_3D, "canvas")))
    padded_centres = jnp.asarray(np.pad(centres, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="edge"))
    return Atlas(tuple(mesh), g, tuple(origins), windows, grid, interior, padded_centres,
                 tuple(plans[g] * passes[g]), _neighbour_map(mesh, pos, n_canvas),
                 _lines(mesh, origins, g, canvas), _walls(mesh, origins, pos, g, canvas), (0.25, 0.0))


def _stage(mesh, g: int, stage: int, pos, n_canvas: int) -> Stage:
    """The gather plan of fill stage ``stage``: every block's side pair of
    that place in its ``fill_order``, each side extended, along running axes
    filled by an earlier stage, ``g`` cells into those ghosts."""
    index = np.arange(n_canvas)
    entries: dict[str, list] = {}
    for b, block in enumerate(mesh):
        shape = block.shape
        order = fill_order(block)
        handled = set(order[:stage])
        name = "ijk"[order[stage]]
        for side in (f"{name}_min", f"{name}_max"):
            normal, mid = _face_geometry(block, side)
            axes = running_axes(side)
            n = tuple(shape[a] for a in axes)
            for rect in block.sides.side(side):
                ranges = []
                for m, axis in enumerate(axes):
                    ext = g if axis in handled else 0
                    lo = rect.start[m] - (ext if rect.start[m] == 0 else 0)
                    hi = rect.stop[m] + (ext if rect.stop[m] == n[m] else 0)
                    ranges.append(range(lo, hi))
                for a in ((x, y) for x in ranges[0] for y in ranges[1]):
                    for layer in range(g):
                        ghost = pos(b, *cell_on(side, a, layer, shape, inward=False))
                        if isinstance(rect, InterfaceRect):
                            partner = mesh[rect.block].shape
                            p = list(rect.partner(a))
                            for m, axis in enumerate(running_axes(rect.side)):
                                corner = g if axes[rect.axes[m]] in handled else 0
                                p[m] = min(max(p[m], -corner), partner[axis] - 1 + corner)
                            index[ghost] = pos(rect.block, *cell_on(rect.side, p, layer, partner, inward=True))
                        else:
                            read = layer if rect.kind in (*WALL_KINDS, "symmetry") else 0
                            source = pos(b, *cell_on(side, a, read, shape, inward=True))
                            f = tuple(min(max(a[m], 0), n[m] - 1) for m in range(2))
                            entries.setdefault(rect.kind, []).append(
                                (ghost, source, *normal[:, f[0], f[1]], *mid[:, f[0], f[1]]))
    conditions, offset = [], n_canvas
    for kind, rows in entries.items():
        rows = np.array(rows)
        index[rows[:, 0].astype(np.int64)] = offset + np.arange(len(rows))
        offset += len(rows)
        conditions.append((kind, rows[:, 1].astype(np.int64), rows[:, 2:5].T.copy(), rows[:, 5:8].T.copy()))
    return Stage(index, tuple(conditions))


def _passes(stages: list[Stage], interior: np.ndarray) -> int:
    """How many times the stages must run so that the last pass reads filled
    cells only (interior cells, or ghosts written from filled cells): the
    fill is then a function of the interior alone."""
    filled = interior.copy()
    for passes in range(1, MAX_PASSES + 1):
        clean = True
        for stage in stages:
            written = stage.index != np.arange(len(filled))
            read = np.concatenate([filled] + [filled[source] for _, source, _, _ in stage.conditions])[stage.index]
            clean &= bool(np.all(read[written]))
            filled = np.where(written, read, filled)
        if clean:
            return passes
    raise ValueError(f"the ghost fill does not settle in {MAX_PASSES} passes: an interface map is inconsistent")


def _neighbour_map(mesh, pos, n_canvas: int) -> np.ndarray:
    """Across every block face, the cell on the other side, or the cell itself."""
    index = np.arange(n_canvas)
    for b, block in enumerate(mesh):
        for side in SIDES_3D:
            for rect in block.sides.side(side):
                for a in ((x, y) for x in range(rect.start[0], rect.stop[0])
                          for y in range(rect.start[1], rect.stop[1])):
                    ghost = pos(b, *cell_on(side, a, 0, block.shape, inward=False))
                    if isinstance(rect, InterfaceRect):
                        partner = mesh[rect.block].shape
                        index[ghost] = pos(rect.block, *cell_on(rect.side, rect.partner(a), 0, partner, inward=True))
                    else:
                        index[ghost] = pos(b, *cell_on(side, a, 0, block.shape, inward=True))
    return index


def _lines(mesh, origins, g: int, canvas) -> Lines:
    """One chain per block and pair of indices, along the block's
    ``line_axis``; links across the faces between its cells (flat indices
    into the canvas face arrays of the three families, concatenated)."""
    dims = [tuple(n + (a == x) for x, n in enumerate(canvas)) for a in range(3)]
    offsets = np.cumsum([0] + [int(np.prod(d)) for d in dims])
    chains, faces = [], []
    for block, origin in zip(mesh, origins):
        axis = line_axis(block)
        cells = np.stack(np.meshgrid(*(o + g + np.arange(n) for o, n in zip(origin, block.shape)), indexing="ij"))
        cells = np.moveaxis(cells, 1 + axis, -1).reshape(3, -1, block.shape[axis])  # chains, cells along
        above = cells.copy()
        above[axis] += 1
        chains.append(np.ravel_multi_index(tuple(cells), canvas))
        faces.append(offsets[axis] + np.ravel_multi_index(tuple(above), dims[axis]))
    length = max(c.shape[1] for c in chains)
    pad = lambda a, fill: np.pad(a, ((0, 0), (0, length - a.shape[1])), constant_values=fill)  # noqa: E731
    cells = np.concatenate([pad(c, -1) for c in chains]).astype(np.int64)
    up_sign = ((cells >= 0) & (np.roll(cells, -1, axis=1) >= 0)).astype(np.int64)
    up_sign[:, -1] = 0
    face = np.where(up_sign != 0, np.concatenate([pad(f, 0) for f in faces]), 0).astype(np.int64)
    return Lines(cells, face, up_sign, face.copy(), -up_sign)


def _walls(mesh, origins, pos, g: int, canvas) -> Walls:
    quads = wall_quads(mesh)
    cell, family, face, sign = [], [], [], []
    for b, s, a in zip(quads.block, quads.side, quads.cell):
        side = SIDES_3D[s]
        c = cell_on(side, tuple(a), 0, mesh[b].shape, inward=True)
        cell.append(pos(b, *c))
        n = normal_axis(side)
        f = [o + g + x for o, x in zip(origins[b], c)]
        f[n] += side.endswith("max")
        dims = list(canvas)
        dims[n] += 1
        family.append(n)
        face.append((f[0] * dims[1] + f[1]) * dims[2] + f[2])
        sign.append(1.0 if side.endswith("min") else -1.0)
    return Walls(quads, np.array(cell, dtype=np.int64), np.array(family, dtype=np.int64),
                 np.array(face, dtype=np.int64), np.array(sign))
