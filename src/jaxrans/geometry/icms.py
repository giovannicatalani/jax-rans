"""ICMS sections in aircraft axes (X aft, Y outboard, Z up).

No aerodynamic angle correction is performed here. Coordinates are rigidly
flattened and scaled; the original geometry and inverse transform are retained.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .airfoil import Airfoil, from_points


@dataclass(frozen=True)
class Section:
    index: int
    points: np.ndarray                 # original order, metres
    source_lines: tuple[int, int]       # one-based, inclusive
    leading_edge_index: int
    leading_edge: np.ndarray
    trailing_edge: np.ndarray
    chord: float
    basis: np.ndarray                  # columns: chord, thickness, plane normal
    normalized: np.ndarray             # original order, x/c and z/c
    airfoil: Airfoil                    # lower TE -> LE -> upper TE
    plane_error: float                  # max distance from the fitted plane / chord
    chord_pitch_deg: float              # TE-up positive, relative to projected aircraft X
    sweep_deg: float = 0.0
    sweep_segment: tuple[int, int] = (0, 1)

    @property
    def quarter_chord(self) -> np.ndarray:
        return self.leading_edge + 0.25 * (self.trailing_edge - self.leading_edge)

    def reconstruct(self) -> np.ndarray:
        return self.leading_edge + self.chord * self.normalized @ self.basis[:, :2].T

    def record(self) -> dict:
        return {
            "index": self.index, "source_lines": list(self.source_lines),
            "points": len(self.points), "leading_edge_index": self.leading_edge_index,
            "leading_edge_m": self.leading_edge.tolist(),
            "trailing_edge_midpoint_m": self.trailing_edge.tolist(),
            "quarter_chord_m": self.quarter_chord.tolist(), "chord_m": self.chord,
            "basis_columns_chord_thickness_normal": self.basis.tolist(),
            "plane_error_over_chord": self.plane_error,
            "reconstruction_error_m": float(np.linalg.norm(self.reconstruct() - self.points, axis=1).max()),
            "chord_pitch_deg_te_up_positive": self.chord_pitch_deg,
            "sweep_deg": self.sweep_deg, "sweep_segment": list(self.sweep_segment),
            "contour_reversed": bool(np.array_equal(np.column_stack((self.airfoil.x, self.airfoil.y)),
                                                     self.normalized[::-1])),
            "trailing_edge_gap_over_chord": self.airfoil.trailing_edge_gap,
        }


def normalize_section(points, index: int = 0, source_lines=(1, 1),
                      plane_tolerance: float = 1e-7) -> Section:
    """LE is the supplied point farthest from the midpoint of the TE corners.

    ``basis`` is orthonormal. Its thickness axis points upwards. A finite TE
    gap is retained: only its midpoint, not both corners, lands at (1, 0).
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 5 or not np.isfinite(p).all():
        raise ValueError(f"section {index}: expected at least 5 finite XYZ points")
    if np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) == 0):
        raise ValueError(f"section {index}: repeated consecutive points")
    te = (p[0] + p[-1]) / 2
    k = int(np.argmax(np.linalg.norm(p - te, axis=1)))
    le, chord = p[k], float(np.linalg.norm(p[k] - te))
    if chord == 0 or k in (0, len(p) - 1):
        raise ValueError(f"section {index}: degenerate chord or leading edge at a TE corner")
    _, singular, vh = np.linalg.svd((p - p.mean(axis=0)) / chord, full_matrices=False)
    if singular[1] < 1e-10:
        raise ValueError(f"section {index}: collinear points")
    normal = vh[-1]
    error = float(np.abs((p - p.mean(axis=0)) @ normal).max() / chord)
    if error > plane_tolerance:
        raise ValueError(f"section {index}: non-planar points ({error:.3e} chord); check ICMS count/header")
    ec = (te - le) / chord
    ez = np.cross(ec, normal)
    ez /= np.linalg.norm(ez)
    if ez[2] < 0:
        ez = -ez
    if ez[2] < 1e-8:
        raise ValueError(f"section {index}: thickness axis cannot be oriented upwards")
    normal = np.cross(ec, ez)
    basis = np.column_stack((ec, ez, normal))
    xy = (p - le) @ basis[:, :2] / chord
    foil = from_points(xy[:, 0], xy[:, 1], f"section_{index:03d}")
    if len(foil.x) != len(p):
        raise ValueError(f"section {index}: coincident points after flattening")
    ex = np.array([1., 0., 0.]) - normal[0] * normal
    if np.linalg.norm(ex) < 1e-8:
        raise ValueError(f"section {index}: aircraft X is normal to the section")
    ex /= np.linalg.norm(ex)
    up = np.cross(normal, ex)
    if up[2] < 0:
        up = -up
    pitch = float(np.degrees(np.arctan2(ec @ up, ec @ ex)))
    return Section(index, p, tuple(source_lines), k, le, te, chord, basis, xy, foil, error, pitch)


def _xyz(line: str):
    words = line.split()
    if len(words) != 3:
        return None
    try:
        return [float(w.replace("D", "E").replace("d", "e")) for w in words]
    except ValueError:
        return None


def read_icms(path: str | Path, count: int, units: str, header_lines: int | None = None,
              plane_tolerance: float = 1e-7) -> list[Section]:
    """Read a final XYZ block, split equally into ``count`` profiles.

    Auto header detection takes the *entire* final numeric block. It never
    discards coordinate-like header rows to make a point count fit; specify
    ``header_lines`` for such headers. Blank lines do not count as points.
    File order must run from root to tip; it is never silently sorted.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 2:
        raise ValueError("sections.count must be an integer >= 2 (needed for local sweep)")
    if units not in ("m", "mm"):
        raise ValueError("sections.units must be 'm' or 'mm'")
    if not np.isfinite(plane_tolerance) or plane_tolerance <= 0:
        raise ValueError("sections.plane_tolerance must be finite and positive")
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    rows = [(i + 1, line) for i, line in enumerate(lines) if line.strip()]
    if header_lines is None:
        start = len(rows)
        while start and _xyz(rows[start - 1][1]) is not None:
            start -= 1
        rows = rows[start:]
    else:
        if not isinstance(header_lines, int) or isinstance(header_lines, bool) or header_lines < 0:
            raise ValueError("sections.header_lines must be a nonnegative integer or null")
        rows = [(i, line) for i, line in rows if i > header_lines]
    if len(rows) < 5 * count or len(rows) % count:
        raise ValueError(f"ICMS: {len(rows)} coordinate rows cannot form {count} equal profiles; "
                         "check sections.count and sections.header_lines")
    values = []
    for i, line in rows:
        xyz = _xyz(line)
        if xyz is None:
            raise ValueError(f"ICMS line {i}: expected three coordinates")
        values.append(xyz)
    points = np.asarray(values) * (0.001 if units == "mm" else 1.0)
    n = len(points) // count
    sections = [normalize_section(points[i*n:(i+1)*n], i,
                                  (rows[i*n][0], rows[(i+1)*n-1][0]), plane_tolerance)
                for i in range(count)]
    q = np.array([s.quarter_chord for s in sections])
    if np.any(np.diff(q[:, 1]) <= 0):
        raise ValueError("ICMS profiles must run root to tip with strictly increasing quarter-chord Y")
    # Incoming secants everywhere; no stencil straddles a kink.
    delta = np.diff(q, axis=0)
    sweep = np.degrees(np.arctan2(delta[:, 0], np.hypot(delta[:, 1], delta[:, 2])))
    return [replace(s, sweep_deg=float(sweep[max(0, i-1)]),
                    sweep_segment=(max(0, i-1), max(0, i-1)+1)) for i, s in enumerate(sections)]


def write_hug(section: Section, path: str | Path) -> None:
    foil = section.airfoil
    with Path(path).open("w") as stream:
        stream.write(f"1\n{len(foil.x)}\n")
        np.savetxt(stream, np.column_stack((foil.x, foil.y)), fmt="%.17g")
