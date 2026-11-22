"""Conical section geometry and a specified, smooth far-field relaxation.

The geometric input is tau=-dc/ds and leading-edge sweep. In the fixed
normal-section frame, generators satisfy dx/dz=p-mu*(x-x_LE), dy/dz=-mu*y.
Both p and mu vanish with tau; no distant apex is formed numerically.

Multiplying both coefficients by w moves the apex parallel to the mean
sweep direction: full conicity at w=1, translation at w=0. The C2 quintic
distance transition is an explicit choice here, not a recovered FLOWer law.
"""

from dataclasses import dataclass

import numpy as np

from ..core.sectional import SectionalSpec
from ..mesh.block import Grid
from ..mesh.mesh import Mesh, WallFaces, as_mesh, wall_faces


def distance_to_section(points: np.ndarray, walls: WallFaces, chunk: int = 2048) -> np.ndarray:
    """Exact nearest-segment distance of points (2, ...) to the section."""
    if not len(walls.a):
        raise ValueError("conical geometry needs wall faces to define its chord and relaxation")
    flat = np.moveaxis(points, 0, -1).reshape(-1, 2)
    edge = walls.b - walls.a
    length2 = np.sum(edge * edge, axis=1)
    out = np.empty(len(flat))
    for start in range(0, len(flat), chunk):
        delta = flat[start:start + chunk, None] - walls.a
        t = np.clip(np.sum(delta * edge, axis=-1) / length2, 0.0, 1.0)
        out[start:start + chunk] = np.sqrt(np.min(np.sum((delta - t[..., None] * edge) ** 2, axis=-1), axis=1))
    return out.reshape(points.shape[1:])


def relaxation_weight(distance: np.ndarray, start: float, end: float) -> np.ndarray:
    """One through start; zero from end; C2 transition between them."""
    t = np.clip((distance - start) / (end - start), 0.0, 1.0)
    return np.where(t >= 1.0, 0.0, np.where(t <= 0.0, 1.0, 1.0 - t**3 * (10.0 - 15.0 * t + 6.0 * t**2)))


@dataclass(frozen=True)
class ConicalGeometry:
    spec: SectionalSpec
    chord: float
    leading_edge_x: float

    @property
    def angle(self) -> float:
        return float(np.deg2rad(self.spec.reference_sweep_deg))

    @property
    def normal_leading_edge_x(self) -> float:
        return self.leading_edge_x * np.cos(self.angle)

    @property
    def coefficients(self) -> tuple[float, float]:
        a, c, s = np.tan(np.deg2rad(self.spec.leading_edge_sweep_deg)), np.cos(self.angle), np.sin(self.angle)
        denominator = c + s * a
        return float((c * a - s) / denominator), float(self.spec.taper_rate / (self.chord * denominator))

    @classmethod
    def of(cls, mesh: Mesh, spec: SectionalSpec, section: str) -> "ConicalGeometry":
        walls = wall_faces(mesh)
        if not len(walls.a):
            raise ValueError("conical geometry needs a wall to define its chord")
        # As for the far-field reference, use the longest section body.
        bodies = [np.concatenate([walls.a[walls.body == body, 0], walls.b[walls.body == body, 0]])
                  for body in range(walls.bodies)]
        x = max(bodies, key=np.ptp)
        leading, chord = float(x.min()), float(np.ptp(x))
        if section == "normal":
            angle = np.deg2rad(spec.reference_sweep_deg)
            chord *= np.cos(angle) + np.sin(angle) * np.tan(np.deg2rad(spec.trailing_edge_sweep_deg))
            leading /= np.cos(angle)
        if not np.isfinite(chord) or chord <= 0:
            raise ValueError("conical geometry needs a positive streamwise chord")
        return cls(spec, chord, leading)

    def normal_section(self, mesh: Mesh, start: float, end: float) -> Mesh:
        """Intersect the streamwise section's generators with the normal plane.

        Relax away from the wall so the mesh transformation is nonsingular
        in the far field. At zero weight this is the usual cosine scaling.
        """
        mesh = as_mesh(mesh)
        walls = wall_faces(mesh)
        p, mu = self.coefficients
        c, s = np.cos(self.angle), np.sin(self.angle)
        out = []
        for block in mesh:
            xy = np.asarray(block.nodes)
            w = relaxation_weight(distance_to_section(xy, walls) / self.chord, start, end)
            x = xy[0] - self.leading_edge_x
            denominator = 1.0 - w * mu * s * x
            if np.any(denominator <= 0.0):
                raise ValueError("conical normal-section mapping crosses the apex; reduce taper or relaxation extent")
            nodes = np.stack([self.normal_leading_edge_x + x * (c - w * p * s) / denominator,
                              xy[1] / denominator])
            out.append(Grid.from_node_array(nodes, block.sides))
        return tuple(out)

    def map_points(self, points: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Corresponding points on a plane z, from their central x/y foot."""
        p, mu = self.coefficients
        x, y, z = points
        return np.stack([x + z * weight * (p - mu * (x - self.normal_leading_edge_x)),
                         y * (1.0 - z * weight * mu), z])

    def streamwise_points(self, xy: np.ndarray) -> np.ndarray:
        """Wall generator intersections with the original streamwise plane."""
        x = xy[0] - self.normal_leading_edge_x
        a = np.tan(np.deg2rad(self.spec.leading_edge_sweep_deg))
        c, s = np.cos(self.angle), np.sin(self.angle)
        kappa = self.spec.taper_rate / self.chord
        denominator = 1.0 + kappa * s * x
        return np.stack([self.leading_edge_x + x * (c + s * a) / denominator, xy[1] / denominator])

    def surface_vectors(self, walls: WallFaces) -> tuple[np.ndarray, np.ndarray]:
        """Streamwise wall midpoints and Cartesian area vectors per unit span.

        The tangent along the span is (a-kappa*X, -kappa*Y, 1).
        These area vectors include the side component of the conical wall.
        """
        a = self.streamwise_points(walls.a.T)
        b = self.streamwise_points(walls.b.T)
        midpoint = 0.5 * (a + b)
        dx, dy = b - a
        sx, sy = -dy, dx
        slope = np.tan(np.deg2rad(self.spec.leading_edge_sweep_deg))
        kappa = self.spec.taper_rate / self.chord
        sz = -(slope - kappa * (midpoint[0] - self.leading_edge_x)) * sx + kappa * midpoint[1] * sy
        return midpoint, np.stack([sx, sy, sz])

    def streamwise_gradient_scale(self, xy: np.ndarray) -> np.ndarray:
        """Gradient/traction scale from the normal cut to the streamwise cut.

        Primitives (including nu_tilde) are copied along the generators.
        A homothety of factor lambda leaves viscosity unchanged and scales
        Cartesian gradients, Newtonian stress and QCR stress by 1/lambda.
        """
        return 1.0 + self.spec.taper_rate / self.chord * np.sin(self.angle) * (
            xy[0] - self.normal_leading_edge_x)
