"""Three blocks round one edge: the vertex no split of a single block makes.

A hexagonal prism cut into three quadrilateral prisms that meet along its
axis, warped by a smooth map. The edge ghosts along the axis have no cell
of their own; the fill copies one of the three neighbours into each, value
and centre together (mesh/atlas3d.py), so the diamond face gradients, built
from node averages of values and of centres alike (numerics/gradients.py),
stay exact for linear fields there, and a uniform flow stays uniform.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import minimum_filter

from jaxrans.core.config import FlowConditions, NumericsConfig
from jaxrans.mesh.assemble import PatchRule
from jaxrans.mesh.assemble3d import assemble_3d
from jaxrans.mesh.atlas import build_atlas
from jaxrans.mesh.cgns import RawBlock3D
from jaxrans.mesh.mesh import assign_conditions
from jaxrans.numerics import boundary as bc
from jaxrans.numerics.gradients import face_gradients
from jaxrans.numerics.residual import make_residual

FLOWS = {
    "euler": FlowConditions(0.6, 4.0),
    "laminar": FlowConditions(0.6, 4.0, 1000.0),
    "sa": FlowConditions(0.6, 4.0, 1e6, sutherland_ratio=0.37, turbulence="sa_noft2"),
}


def warp(p: np.ndarray) -> np.ndarray:
    x, y, z = p
    return np.stack([x + 0.04 * np.sin(2.0 * y) * np.cos(z), y + 0.04 * np.sin(2.0 * z + x),
                     z + 0.05 * np.cos(1.5 * x) * np.sin(y)])


def prism(n: int = 6, layers: int = 6) -> list[RawBlock3D]:
    """Three blocks ``n x n x layers``: quadrilaterals (centre, v_2b, v_2b+1,
    v_2b+2) of the unit hexagon, bilinear, extruded in ``z``, warped."""
    v = np.array([(np.cos(t), np.sin(t)) for t in np.pi / 3 * np.arange(6)])
    s = np.linspace(0.0, 1.0, n + 1)[:, None, None]
    t = np.linspace(0.0, 1.0, n + 1)[None, :, None]
    z = np.linspace(0.0, 1.0, layers + 1)
    blocks = []
    for b in range(3):
        p00, p10, p11, p01 = np.zeros(2), v[2 * b], v[(2 * b + 1) % 6], v[(2 * b + 2) % 6]
        quad = (1 - s) * (1 - t) * p00 + s * (1 - t) * p10 + s * t * p11 + (1 - s) * t * p01   # (n+1, n+1, 2)
        nodes = np.stack([np.broadcast_to(quad[..., 0, None], quad.shape[:2] + z.shape),
                          np.broadcast_to(quad[..., 1, None], quad.shape[:2] + z.shape),
                          np.broadcast_to(z, quad.shape[:2] + z.shape)])
        blocks.append(RawBlock3D(warp(nodes)))
    return blocks


@pytest.fixture(scope="module")
def mesh():
    mesh, info = assemble_3d(prism(), (PatchRule("outer"),))
    assert info.blocks == 3 and info.interfaces == 6 and info.interface_mismatch < 1e-14
    return assign_conditions(mesh, {"outer": "far_field"})


@pytest.mark.parametrize("flow", list(FLOWS))
def test_a_uniform_flow_stays_uniform(mesh, flow) -> None:
    atlas = build_atlas(mesh)
    numerics = NumericsConfig(limiter="van_albada")
    distance = jnp.full(atlas.shape, 1e8) if FLOWS[flow].turbulence else None   # no wall: SA source zero
    operator = make_residual(atlas, FLOWS[flow], numerics, distance=distance)
    R = np.asarray(jax.jit(operator.residual)(operator.initial_state()))[:, atlas.interior]
    scale = np.abs(np.asarray(operator.initial_state())).max()
    assert np.max(np.abs(R)) / scale < 1e-13, np.max(np.abs(R)) / scale


def test_face_gradients_are_exact_for_a_linear_field_round_the_edge(mesh) -> None:
    atlas = build_atlas(mesh)
    slope = np.array([0.3, -1.2, 0.7])
    centres = np.asarray(atlas.centres)[(slice(None),) + (slice(1, -1),) * 3]
    phi = jnp.asarray(np.einsum("d,d...->...", slope, centres))[None]
    q = jnp.concatenate([jnp.ones_like(phi), jnp.zeros_like(phi).repeat(3, 0), phi])  # any primitives
    q_inf = jnp.asarray([1.0, 0.0, 0.0, 0.0, 1.0])
    filled = bc.fill_ghosts(q, atlas, bc.Conditions(q_inf, 1.4, bc.uniform_far_field(q, atlas, q_inf, 1.4)))
    value = np.asarray(filled[4])
    # Cells holding the linear field: block cells and first-layer interface
    # ghosts, edges included (not the conditions' ghosts, nor gaps). A face
    # gradient reads the 2 x 3 x 3 cells round the face's four nodes; every
    # face of a block cell whose box holds the linear field is checked.
    linear = np.abs(value - np.einsum("d,d...->...", slope, centres)) < 1e-12
    grads = face_gradients(jnp.pad(filled[4:5], ((0, 0),) + ((1, 1),) * 3, mode="edge"), atlas.centres)
    checked = []
    for a, G in enumerate(grads):
        across = minimum_filter(linear.astype(np.int8), size=[1 if b == a else 3 for b in range(3)],
                                mode="constant", cval=0) > 0
        widths = [(1, 1) if b == a else (0, 0) for b in range(3)]
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[a], hi[a] = slice(0, -1), slice(1, None)
        box, cell = np.pad(across, widths), np.pad(atlas.interior, widths)
        ok = box[tuple(lo)] & box[tuple(hi)] & (cell[tuple(lo)] | cell[tuple(hi)])     # face shape
        err = np.abs(np.asarray(G)[:, 0] - slope[:, None, None, None])[:, ok]
        assert err.max() < 1e-11, (a, err.max())
        checked.append(ok)
    # the faces of each block's cells on the axis (its corner i = j = 0), half way up
    for window in atlas.windows:
        i, j, k = (w.start for w in window)
        k += 3
        assert checked[0][i, j, k] and checked[1][i, j, k] and checked[2][i, j, k]
