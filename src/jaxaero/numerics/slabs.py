"""Domain decomposition of the residual's stencils over devices.

The canvas is cut in equal slabs along its first axis, one per device. The
ghost fill (boundary conditions, interfaces) is data movement and runs on the
split canvas as it is; the stencil part, ``block_residual``, needs every cell's
neighbours up to ``g`` cells away, so each device takes ``g`` rows from either
neighbour in one exchange (``ppermute``) and computes its own rows from its
slab, as the whole canvas would: at the two ends of the canvas the rows are
the edge padding the unsplit residual uses. One exchange per evaluation, in
place of one per shifted array that automatic partitioning makes.

Geometry (volumes, face vectors, centres, wall distance) is cut once, each
device's piece with the rows its faces and gradients reach.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from ..mesh.atlas import Atlas

SPACE_AXIS = "space"
"""The device axis of a slab decomposition (the driver splits the state on it too)."""


def _pieces(a, rows: int, extra: int, n: int, axis: int) -> np.ndarray:
    """Device ``d``'s rows ``[d rows, (d + 1) rows + extra)`` of ``a`` along
    ``axis``, stacked on a new leading axis."""
    a = np.asarray(a)
    return np.stack([np.take(a, np.arange(d * rows, (d + 1) * rows + extra), axis=axis) for d in range(n)])


def slab_stencil(atlas: Atlas, distance: Array | None, devices: Sequence,
                 stencil: Callable) -> Callable[[Array], Array]:
    """``filled -> R``: ``stencil(padded, grid, centres, distance)`` (the
    residual's ``block_residual`` with its options bound) of the ghost-filled
    canvas ``(nvar, *canvas shape)``, each device computing its slab.

    Raises:
        ValueError: the canvas's first axis does not divide over the devices.
    """
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh, PartitionSpec as P

    n, g, grid, dim = len(devices), atlas.g, atlas.grid, atlas.dim
    if atlas.shape[0] % n:
        raise ValueError(f"cannot split the canvas over {n} devices: its first axis has {atlas.shape[0]} "
                         f"cells (ghosts included), not a multiple of {n}")
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

    mesh = Mesh(np.array(devices), (SPACE_AXIS,))
    split = shard_map(local, mesh=mesh, in_specs=(P(None, SPACE_AXIS), leading),
                      out_specs=P(None, SPACE_AXIS), check_rep=False)
    return lambda filled: split(filled, geometry)
