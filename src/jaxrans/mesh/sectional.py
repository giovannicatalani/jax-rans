"""A section with one independent cell across an artificial 3D extrusion.

The reference is the existing 2D normal-section atlas. Its topology and
physical boundary plans are carried into every spanwise plane, then all
lateral ghost values are gathered from the central plane. Ghost *positions*
are at their translated images, never at the source cell's position.

Axes are those of the 2.5D solver: x normal chord, y up, z along sweep.
Zero taper uses straight extrusion. Finite taper replaces the metrics and
face conditions with a conical mapping relaxed in the far field. The residual
is the ordinary three-direction finite-volume stencil, including lateral fluxes.
"""

from dataclasses import dataclass, fields, replace

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..core.sectional import SectionalSpec
from ..geometry.conical import ConicalGeometry, distance_to_section, relaxation_weight
from .atlas import Atlas, Lines, Stage, Walls, _extended_nodes, build_atlas, ghost_cell, lift_atlas
from .block import OUTWARD, SIDES, Boundary, Grid, metrics
from .block3d import SIDES_3D, BoundaryRect, InterfaceRect, box_sides
from .mesh import Mesh, as_mesh
from .wall3d import WallQuads


@dataclass(frozen=True)
class SectionalAtlas(Atlas):
    """A genuine 3D atlas with a single independent spanwise plane per block."""

    reference: Atlas
    spec: SectionalSpec
    depth: float
    geometry: ConicalGeometry | None = None
    relaxation_start: float = 1.0
    relaxation_end: float = 5.0

    def section_values(self, values: Array) -> Array:
        """Central plane on the reference canvas, including its 2D ghosts."""
        return values[..., self.g]

    def expand(self, values: Array) -> Array:
        """A section field copied into the spanwise canvas (ghosts included)."""
        return jnp.broadcast_to(values[..., None], values.shape + (self.shape[-1],))


def _extrude(mesh: Mesh, depth: float) -> Mesh:
    out = []
    for b, block in enumerate(mesh):
        def carry(segment):
            if isinstance(segment, Boundary):
                return BoundaryRect(segment.patch, (segment.start, 0), (segment.stop, 1), segment.kind)
            return InterfaceRect(
                (segment.start, 0), (segment.stop, 1), segment.block, segment.side,
                (segment.partner_start + (segment.stop - segment.start - 1 if segment.reversed else 0), 0),
                flip=(segment.reversed, False),
            )

        sides = {s: tuple(carry(segment) for segment in block.sides.side(s)) for s in SIDES}
        whole = ((0, 0), block.shape)
        sides.update(k_min=(InterfaceRect(*whole, b, "k_max", (0, 0)),),
                     k_max=(InterfaceRect(*whole, b, "k_min", (0, 0)),))
        xy = np.asarray(block.nodes)
        shape = block.nodes.shape[1:] + (2,)
        nodes = np.concatenate([
            np.broadcast_to(xy[..., None], (2,) + shape),
            np.broadcast_to(np.array([-0.5, 0.5]) * depth, shape)[None],
        ])
        out.append(Grid.from_node_array(nodes, box_sides(block.shape + (1,), **sides)))
    return tuple(out)


def _stage(stage: Stage, count: int, depth: float, g: int, cells: int) -> Stage:
    """Lift a reference gather, including its condition-value indices."""
    k = np.arange(count)
    index = (stage.index[:, None] * count + k).ravel()
    conditions = []
    for kind, source, normal, midpoint in stage.conditions:
        sources = (source[:, None] * count + k).ravel()
        normals = np.repeat(normal, count, axis=1)
        midpoints = np.vstack([np.repeat(midpoint, count, axis=1),
                               np.tile((k - g) * depth, len(source))])
        conditions.append((kind, sources, normals, midpoints))
    assert len(index) == cells * count
    return Stage(index, tuple(conditions))


def build_sectional_atlas(mesh: Mesh | Grid, spec: SectionalSpec, g: int = 2,
                         depth: float = 1.0, relaxation_start: float = 1.0,
                         relaxation_end: float = 5.0,
                         geometry: ConicalGeometry | None = None) -> SectionalAtlas:
    """Extrude an assigned 2D mesh, already in the normal-section frame.

    ``depth`` is a numerical length in grid units, not a physical wing span.
    The artificial planes are at z = +/-depth/2; the unknown is at z = 0.
    """
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError("sectional depth must be finite and positive")
    if (not np.isfinite(relaxation_start) or not np.isfinite(relaxation_end)
            or not 0 <= relaxation_start < relaxation_end):
        raise ValueError("sectional relaxation needs finite 0 <= relaxation_start < relaxation_end")
    mesh = as_mesh(mesh)
    if any(block.dim != 2 for block in mesh):
        raise ValueError("a sectional calculation needs a 2D source mesh")
    reference = lift_atlas(build_atlas(mesh, g))
    count = 1 + 2 * g
    shape = reference.shape + (count,)
    size = int(np.prod(reference.shape))

    def repeat(values):
        return np.repeat(np.asarray(values)[..., None], count, axis=-1)

    xy = np.asarray(reference.centres)
    centres = np.concatenate([
        np.repeat(xy[..., None], count + 2, axis=-1),
        np.broadcast_to((np.arange(count + 2) - g - 1) * depth,
                        (1,) + xy.shape[1:] + (count + 2,)),
    ])
    faces = [repeat(s) * depth for s in reference.grid.faces]
    area = np.repeat(np.asarray(reference.grid.volume)[..., None], count + 1, axis=-1)
    faces.append(np.stack([np.zeros_like(area), np.zeros_like(area), area]))
    volume = repeat(reference.grid.volume) * depth
    mesh3 = _extrude(mesh, depth)
    grid = Grid(jnp.zeros((3,) + tuple(n + 1 for n in shape)), jnp.asarray(centres[:, 1:-1, 1:-1, 1:-1]),
                jnp.asarray(volume), tuple(jnp.asarray(s) for s in faces),
                box_sides(shape, **dict.fromkeys(SIDES_3D, "canvas")))
    interior = np.zeros(shape, bool)
    interior[..., g] = reference.interior

    stages = tuple(_stage(stage, count, depth, g, size) for stage in reference.stages)
    lateral = np.broadcast_to(np.arange(size)[:, None] * count + g, (size, count)).ravel().copy()
    stages += (Stage(lateral, ()),)
    neighbours = (reference.neighbours[:, None] * count + np.arange(count)).reshape(shape)
    neighbours[..., :g] = reference.neighbours.reshape(reference.shape)[..., None] * count + g
    neighbours[..., g + 1:] = reference.neighbours.reshape(reference.shape)[..., None] * count + g

    w, f = reference.walls, reference.walls.faces
    a = np.column_stack([f.a, np.full(len(f.a), -depth / 2)])
    b = np.column_stack([f.b, np.full(len(f.b), -depth / 2)])
    shift = np.array([0.0, 0.0, depth])
    quads = WallQuads(f.block, np.array([SIDES_3D.index(SIDES[s]) for s in f.side]),
                     np.column_stack([f.cell, np.zeros(len(f.cell), int)]), f.body,
                     np.stack([a, b, b + shift, a + shift], axis=1))
    walls = Walls(quads, w.cell * count + g, w.family, w.face * count + g, w.into_fluid)
    lines = reference.lines
    lines = Lines(np.where(lines.cells >= 0, lines.cells * count + g, -1),
                  lines.up_face * count + g, lines.up_sign, lines.down_face * count + g, lines.down_sign)
    base = Atlas(mesh3, g, tuple(o + (0,) for o in reference.origins),
                 tuple(w + (slice(g, g + 1),) for w in reference.windows), grid, interior,
                 jnp.asarray(centres), stages, neighbours.ravel(), lines, walls, reference.quarter_chord)
    atlas = SectionalAtlas(**{f.name: getattr(base, f.name) for f in fields(Atlas)},
                           reference=reference, spec=spec, depth=depth,
                           relaxation_start=relaxation_start, relaxation_end=relaxation_end)
    if spec.taper_rate != 0.0:
        geometry = geometry or ConicalGeometry.of(mesh, spec, "normal")
        if geometry.spec != spec:
            raise ValueError("conical geometry and sectional specification must match")
        return _conical_geometry(atlas, geometry)
    return atlas


def _conical_geometry(atlas: SectionalAtlas, geometry: ConicalGeometry) -> SectionalAtlas:
    """Replace extrusion metrics and physical face conditions, retaining the gather plan.

    All three face families come from the same hexahedral nodes. This keeps
    the closed-cell metric identity, also in the relaxation region. Lateral
    values remain Cartesian copies: radial *fluxes* are not removed.
    """
    reference, g, count = atlas.reference, atlas.g, atlas.shape[-1]
    centres = np.asarray(atlas.grid.centres).copy()
    volume = np.asarray(atlas.grid.volume).copy()
    faces = [np.asarray(f).copy() for f in atlas.grid.faces]
    stages = [Stage(s.index, tuple((kind, source, normal.copy(), midpoint.copy())
                                   for kind, source, normal, midpoint in s.conditions)) for s in atlas.stages]
    walls = reference.walls.faces
    mesh = []
    z = (np.arange(count + 1) - g - 0.5) * atlas.depth
    n2 = int(np.prod(reference.shape))
    for b, (block, origin) in enumerate(zip(reference.mesh, reference.origins)):
        xy = np.stack(_extended_nodes(block.x_node, block.y_node, g))
        weight = relaxation_weight(distance_to_section(xy, walls) / geometry.chord,
                                   atlas.relaxation_start, atlas.relaxation_end)
        shape = xy.shape[1:] + (count + 1,)
        points = np.concatenate([np.broadcast_to(xy[..., None], (2,) + shape),
                                  np.broadcast_to(z, shape)[None]])
        nodes = geometry.map_points(points, weight[..., None])
        xc, vol, sf = metrics(nodes)
        active = vol[g:g + block.ni, g:g + block.nj]
        if not np.isfinite(vol).all() or np.any(active <= 0):
            raise ValueError("conical extrusion folds or crosses the apex; reduce sectional depth or relaxation extent")
        tile = tuple(slice(o, o + n) for o, n in zip(origin + (0,), vol.shape))
        centres[(slice(None),) + tile] = xc
        volume[tile] = vol
        for a, f in enumerate(sf):
            window = tuple(slice(o, o + n) for o, n in zip(origin + (0,), f.shape[1:]))
            faces[a][(slice(None),) + window] = f
        physical = nodes[:, g:g + block.ni + 1, g:g + block.nj + 1, g:g + 2]
        mesh.append(Grid.from_node_array(physical, atlas.mesh[b].sides))

        def position(i, j):
            return (origin[0] + g + i) * reference.shape[1] + origin[1] + g + j

        for stage_number, (sides, pad) in enumerate([(("j_min", "j_max"), False),
                                                    (("i_min", "i_max"), True)]):
            plan = reference.stages[stage_number]
            offsets = np.cumsum([0] + [len(c[1]) for c in plan.conditions])
            for side in sides:
                segments = block.sides.side(side)
                extent = block.sides.length(side)
                axis = 0 if side.startswith("i") else 1
                fixed = g + (0 if side.endswith("min") else block.shape[axis])
                for number, segment in enumerate(segments):
                    if not isinstance(segment, Boundary):
                        continue
                    ext = g if pad else 0
                    lo = segment.start - (ext if number == 0 else 0)
                    hi = segment.stop + (ext if number == len(segments) - 1 else 0)
                    for along in range(lo, hi):
                        a = min(max(along, 0), extent - 1) + g
                        i, j = (fixed, a) if axis == 0 else (a, fixed)
                        normal = OUTWARD[side] * sf[axis][:, i, j]
                        normal = normal / np.linalg.norm(normal, axis=0)
                        face_nodes = nodes[:, i, j:j + 2] if axis == 0 else nodes[:, i:i + 2, j]
                        midpoint = 0.25 * (face_nodes[:, 0, :-1] + face_nodes[:, 1, :-1]
                                           + face_nodes[:, 0, 1:] + face_nodes[:, 1, 1:])
                        for layer in range(g):
                            target = position(*ghost_cell(side, along, layer, block.ni, block.nj))
                            slot = int(plan.index[target] - n2)
                            kind = int(np.searchsorted(offsets, slot, side="right") - 1)
                            row = slot - offsets[kind]
                            stages[stage_number].conditions[kind][2][:, row * count:(row + 1) * count] = normal
                            stages[stage_number].conditions[kind][3][:, row * count:(row + 1) * count] = midpoint

    quads = atlas.walls.faces
    corners = np.moveaxis(quads.corners, -1, 0)
    corners = np.moveaxis(geometry.map_points(corners, np.ones(corners.shape[1:])), 0, -1)
    grid = replace(atlas.grid, centres=jnp.asarray(centres), volume=jnp.asarray(volume),
                   faces=tuple(jnp.asarray(s) for s in faces))
    return replace(atlas, mesh=tuple(mesh), grid=grid, geometry=geometry, stages=tuple(stages),
                   centres=jnp.asarray(np.pad(centres, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="edge")),
                   walls=replace(atlas.walls, faces=replace(quads, corners=corners)))
