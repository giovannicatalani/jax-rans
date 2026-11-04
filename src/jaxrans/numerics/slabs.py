"""Domain decomposition over devices: the canvas cut in equal slabs along its
first axis, one per device.

Everything here runs only on a ``SlabAtlas`` (``jaxrans CASE --devices N``);
an ordinary atlas never reaches it. Three pieces:

* the stencils (``block_residual``) need every cell's neighbours up to ``g``
  cells away: each device takes ``g`` rows from either neighbour in one
  exchange (``ppermute``) and computes its own rows from its slab, as the
  whole canvas would (at the canvas's two ends, the edge padding the unsplit
  residual uses). One exchange per evaluation, in place of one per shifted
  array that automatic partitioning makes;
* the ghost fill and the neighbour fill of the face averages gather and write
  only the cells that move (the same values as ``boundary.fill_ghosts`` and
  ``Atlas.fill_neighbours``, which gather the whole canvas: split, a whole
  gather is a copy of the state to every device);
* the line chains (``solvers/preconditioners.py``) take their slice form.

Geometry (volumes, face vectors, centres, wall distance) is cut once, each
device's piece with the rows its faces and gradients reach.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from ..mesh.atlas import Atlas
from . import boundary as bc

SPACE_AXIS = "space"
"""The device axis of a slab decomposition (the driver splits the state on it too)."""


@dataclass(frozen=True)
class SlabAtlas(Atlas):
    """An atlas whose canvas is split over ``devices``, in slabs of its first axis."""

    devices: tuple = ()

    @classmethod
    def of(cls, atlas: Atlas, devices: Sequence) -> SlabAtlas:
        """``atlas`` split over ``devices``.

        Raises:
            ValueError: the canvas's first axis does not divide over the devices.
        """
        if atlas.shape[0] % len(devices):
            raise ValueError(f"cannot split the canvas over {len(devices)} devices: its first axis has "
                             f"{atlas.shape[0]} cells (ghosts included), not a multiple of {len(devices)}")
        return cls(**{f.name: getattr(atlas, f.name) for f in fields(Atlas)}, devices=tuple(devices))

    def fill_neighbours(self, a: Array) -> Array:
        lead = a.shape[:a.ndim - self.dim]
        flat = a.reshape(lead + (-1,))
        moved = np.flatnonzero(self.neighbours != np.arange(self.neighbours.size))
        return flat.at[..., moved].set(flat[..., self.neighbours[moved]]).reshape(a.shape)


def fill_ghosts(q: Array, atlas: SlabAtlas, conditions: bc.Conditions) -> Array:
    """``boundary.fill_ghosts``, gathering and writing only each stage's ghosts
    (its ``index`` is the identity elsewhere)."""
    flat = q.reshape(q.shape[0], -1)
    n = flat.shape[1]
    for stage in atlas.stages:
        ghosts = np.flatnonzero(stage.index != np.arange(n))
        source = stage.index[ghosts]
        copied = source < n                      # the rest take a boundary condition's value
        values = [bc.CONDITIONS[kind](flat[:, cells], bc.Face(jnp.asarray(normal, dtype=q.dtype),
                                                           jnp.asarray(midpoint, dtype=q.dtype)), conditions)
                  for kind, cells, normal, midpoint in stage.conditions]       # from the stage's start, as unsplit
        new = flat.at[:, ghosts[copied]].set(flat[:, source[copied]])
        if values:
            new = new.at[:, ghosts[~copied]].set(jnp.concatenate(values, axis=1)[:, source[~copied] - n])
        flat = new
    return flat.reshape(q.shape)


def _pieces(a, rows: int, extra: int, n: int, axis: int) -> np.ndarray:
    """Device ``d``'s rows ``[d rows, (d + 1) rows + extra)`` of ``a`` along
    ``axis``, stacked on a new leading axis."""
    a = np.asarray(a)
    return np.stack([np.take(a, np.arange(d * rows, (d + 1) * rows + extra), axis=axis) for d in range(n)])


def slab_stencil(atlas: SlabAtlas, distance: Array | None, stencil: Callable) -> Callable[[Array], Array]:
    """``filled -> R``: ``stencil(padded, grid, centres, distance)`` (the
    residual's ``block_residual`` with its options bound) of the ghost-filled
    canvas ``(nvar, *canvas shape)``, each device computing its slab."""
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh, PartitionSpec as P

    n, g, grid, dim = len(atlas.devices), atlas.g, atlas.grid, atlas.dim
    rows = atlas.shape[0] // n
    geometry = {
        "volume": _pieces(grid.volume, rows, 0, n, 0),
        "faces": tuple(_pieces(f, rows, 1 if a == 0 else 0, n, 1) for a, f in enumerate(grid.faces)),
        "centres": _pieces(atlas.centres, rows, 2, n, 1),
        "distance": None if distance is None else _pieces(distance, rows, 0, n, 0),
    }
    leading = jax.tree_util.tree_map(lambda _: P(SPACE_AXIS), geometry)
    edge = ((0, 0), (0, 0)) + ((g, g),) * (dim - 1)
    up, down = [(k, k + 1) for k in range(n - 1)], [(k + 1, k) for k in range(n - 1)]

    def local(filled: Array, geo: dict) -> Array:
        geo = jax.tree_util.tree_map(lambda a: a[0], geo)
        me = lax.axis_index(SPACE_AXIS)
        below = lax.ppermute(filled[:, -g:], SPACE_AXIS, up)      # from the device before
        above = lax.ppermute(filled[:, :g], SPACE_AXIS, down)     # from the device after
        below = jnp.where(me == 0, jnp.repeat(filled[:, :1], g, axis=1), below)
        above = jnp.where(me == n - 1, jnp.repeat(filled[:, -1:], g, axis=1), above)
        padded = jnp.pad(jnp.concatenate([below, filled, above], axis=1), edge, mode="edge")
        piece = replace(grid, volume=geo["volume"], faces=geo["faces"])
        return stencil(padded, piece, geo["centres"], geo["distance"])

    mesh = Mesh(np.array(atlas.devices), (SPACE_AXIS,))
    split = shard_map(local, mesh=mesh, in_specs=(P(None, SPACE_AXIS), leading),
                      out_specs=P(None, SPACE_AXIS), check_rep=False)
    return lambda filled: split(filled, geometry)
