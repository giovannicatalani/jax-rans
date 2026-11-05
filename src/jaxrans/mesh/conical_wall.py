"""Distance to the physical, radially extended conical wall, not a thin slab.

Each central wall segment a->b generates a planar sector. Write its points
as a + s E + z D_a, where E=b-a, D_a=(p-mu*(a_x-x_LE), -mu*a_y, 1).
The sector is s>=0, s+mu*z<=1. Its closest point is either the orthogonal
plane projection or a point on one of the two generator rays. This form
avoids constructing a remote apex as mu tends to zero and includes the
infinite-extrusion limit.
"""

import numpy as np

from ..geometry.conical import ConicalGeometry
from .mesh import WallFaces


def conical_panel_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray,
                           geometry: ConicalGeometry) -> np.ndarray:
    """Pairwise distances: points (P,3), segment endpoints (F,2) -> (P,F)."""
    a = np.column_stack([a, np.zeros(len(a))])
    b = np.column_stack([b, np.zeros(len(b))])
    p, mu = geometry.coefficients

    def generator(x):
        return np.stack([p - mu * (x[:, 0] - geometry.normal_leading_edge_x),
                         -mu * x[:, 1], np.ones(len(x))], axis=-1)

    e, d = b - a, generator(a)
    normal = np.cross(e, d)
    determinant = np.sum(normal**2, axis=-1)
    ee, ed, dd = np.sum(e * e, -1), np.sum(e * d, -1), np.sum(d * d, -1)
    r = points[:, None] - a
    re, rd = np.sum(r * e, -1), np.sum(r * d, -1)
    s = (re * dd - rd * ed) / determinant
    z = (rd * ee - re * ed) / determinant
    plane = np.abs(np.sum(r * normal, -1)) / np.sqrt(determinant)
    best = np.where((s >= 0) & (s + mu * z <= 1), plane, np.inf)
    for endpoint in (a, b):
        direction = generator(endpoint)
        delta = points[:, None] - endpoint
        z = np.sum(delta * direction, -1) / np.sum(direction**2, -1)
        if mu != 0.0:
            z = np.where(mu * z > 1.0, 1.0 / mu, z)
        residual = delta - z[..., None] * direction
        best = np.minimum(best, np.linalg.norm(residual, axis=-1))
    return best


def conical_wall_distance(points: np.ndarray, walls: WallFaces, geometry: ConicalGeometry,
                          chunk: int = 512) -> np.ndarray:
    """Minimum over all physical conical panels, without artificial span ends.

    points is (...,3), and the return value has shape (...). All panels are
    considered; no finite search radius or nearest-face shortlist is used.
    """
    if not len(walls.a):
        return np.full(points.shape[:-1], np.inf)
    flat = points.reshape(-1, 3)
    out = np.empty(len(flat))
    for start in range(0, len(flat), chunk):
        out[start:start + chunk] = conical_panel_distance(
            flat[start:start + chunk], walls.a, walls.b, geometry).min(axis=1)
    return out.reshape(points.shape[:-1])
