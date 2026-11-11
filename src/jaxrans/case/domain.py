"""The ``domain`` section of a case: which grid, and how to obtain it.

One grid per case: generated round an airfoil (``o_grid``, ``c_grid``:
mesh/generate.py, the airfoil a NACA 4-digit name or a coordinate file),
generated round a multi-element section (``multi_element``: mesh/highlift.py,
the elements from one file each or from one multi-curve file, optionally
re-placed), or read from any structured grid file (``file``: Plot3D or CGNS,
single- or multi-block, 2D or 3D: mesh/assemble.py, mesh/assemble3d.py).
Paths are resolved against the case directory first, the working directory
second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..geometry.airfoil import Airfoil, airfoil_source
from ..geometry.elements import Element, place, read_element, read_elements
from ..mesh.assemble import PatchRule, RawBlock, assemble, load_grid
from ..mesh.assemble3d import is_3d, load_grid_3d
from ..mesh.generate import Spacing, blunt_c_grid, c_grid, o_grid
from ..mesh.highlift import Settings, multi_element
from ..mesh.mesh import Mesh
from ..post.forces import ForceReference


@dataclass(frozen=True)
class GridSpec:
    """How to obtain the grid (``docs/case_reference.md`` lists every key)."""

    grid: str = "c_grid"
    airfoil: str | None = None
    airfoil_smoothing: float = 0.0
    airfoil_smoothing_method: str = "parametric"
    airfoil_smoothing_points: int = 96
    airfoil_smoothing_radius_tol: float = 0.1
    cells_airfoil: int | None = None
    cells_wake: int | None = None
    cells_normal: int | None = None
    wall_spacing: float | None = None
    far_field: float = 100.0
    trailing_edge: str = "as_given"
    leading_edge_spacing: float = Spacing.leading_edge
    trailing_edge_spacing: float = Spacing.trailing_edge
    file: str | None = None
    format: str = "auto"
    coordinate_scale: float = 1.0
    patches: tuple[dict[str, Any], ...] = field(default=())
    elements: tuple[dict[str, Any], ...] = field(default=())
    cells_per_chord: float | None = None
    layer_cells: int = 32
    layer_thickness: float = 0.02
    gap_cells: int = 16
    even_cells: int = 4
    reference_length: float = 1.0
    reference_area: float | None = None
    moment_centre: tuple[float, ...] = (0.25, 0.0, 0.0)
    section: str = "streamwise"
    base_blocks: bool = False

    def __post_init__(self):
        if not np.isfinite(self.airfoil_smoothing) or self.airfoil_smoothing < 0:
            raise ValueError("airfoil_smoothing must be finite and nonnegative")
        if self.airfoil_smoothing_method not in ("parametric", "bspline", "curvature"):
            raise ValueError("unknown airfoil smoothing method")
        if (isinstance(self.airfoil_smoothing_points, bool)
                or not isinstance(self.airfoil_smoothing_points, int) or self.airfoil_smoothing_points < 12):
            raise ValueError("airfoil_smoothing_points must be an integer >= 12")
        if not np.isfinite(self.airfoil_smoothing_radius_tol) or not 0 <= self.airfoil_smoothing_radius_tol < 1:
            raise ValueError("airfoil_smoothing_radius_tol must lie in [0, 1)")

    def force_reference(self) -> ForceReference:
        return ForceReference(self.reference_length, self.reference_area, tuple(self.moment_centre))

    def build(self, base: Path, dtype=np.float64) -> Mesh:
        """The grid's blocks, their patches named but not yet given conditions."""
        return GRIDS[self.grid](self, base, dtype)

    def rules(self) -> tuple[PatchRule, ...]:
        return tuple(PatchRule(**rule) for rule in self.patches)

    def describe(self, base: Path | None = None) -> str:
        if self.grid == "file":
            return f"file {Path(self.file).name}"
        if self.grid == "multi_element":
            roles = [e["role"] for e in self.elements]
            return (f"multi_element {'+'.join(roles)}: {self.cells_per_chord:g} cells per chord, wall spacing "
                    f"{self.wall_spacing:g}, layers {self.layer_cells} cells to {self.layer_thickness:g}, "
                    f"far field {self.far_field:g}")
        wake = f" + 2x{self.cells_wake} wake" if self.grid == "c_grid" else ""
        text = (f"{self.grid} round {self.airfoil}: {self.cells_airfoil}{wake} x {self.cells_normal} cells, "
                f"wall spacing {self.wall_spacing:g}, far field {self.far_field:g}")
        if base is not None:
            text += "".join(f"; {note}" for note in self.airfoil_shape(base)[1])
        return text

    def airfoil_shape(self, base: Path) -> tuple[Airfoil, list[str]]:
        """The airfoil the grid is built round, and what was done to it."""
        given = airfoil_source(self.airfoil, base)
        airfoil, notes = given.unit_chord(), []
        if abs(given.chord - 1.0) > 1e-6:
            notes.append(f"airfoil chord {given.chord:g} scaled to 1")
        if self.trailing_edge == "close" and not airfoil.sharp:
            notes.append(f"trailing-edge gap {airfoil.trailing_edge_gap:.3g} closed")
            airfoil = airfoil.closed()
        if self.airfoil_smoothing:
            from ..geometry.regularize import regularize_airfoil

            airfoil, report = regularize_airfoil(airfoil, self.airfoil_smoothing,
                                                method=self.airfoil_smoothing_method,
                                                control_points=self.airfoil_smoothing_points,
                                                radius_tolerance=self.airfoil_smoothing_radius_tol)
            notes.append(f"airfoil regularization ({self.airfoil_smoothing_method}): "
                         f"budget {self.airfoil_smoothing:g} c, "
                         f"continuous displacement bound {report['deviation_bound']:.6g} c, "
                         f"LE radius {report['before']['leading_edge_radius']:.6g} -> "
                         f"{report['after']['leading_edge_radius']:.6g} c")
            if self.airfoil_smoothing_method == "curvature":
                notes.append(f"curvature fit: {self.airfoil_smoothing_points} control points, "
                             f"nose-radius relative bound {self.airfoil_smoothing_radius_tol:g}, "
                             f"RMS displacement {report['rms_deviation']:.6g} c, "
                             "trailing-edge tangents preserved")
        if self.grid == "c_grid" and not airfoil.sharp:
            notes.append("blunt trailing edge: base and wake blocks")
        return airfoil, notes

    def placed_elements(self, base: Path) -> list[Element]:
        """The elements, front to back, read and (where asked) re-placed against the main element."""
        roles = [e["role"] for e in self.elements]
        if self.file:
            elements = read_elements(self.resolve(base), roles, self.coordinate_scale)
        else:
            elements = [read_element(self.resolve(base, e["file"]), e["role"], self.coordinate_scale)
                        for e in self.elements]
        main = elements[roles.index("main")]
        out = []
        for given, element in zip(self.elements, elements):
            moves = {k: given[k] for k in ("deflection_deg", "gap", "overlap") if k in given}
            if moves:
                element = place(element, main, moves.get("deflection_deg", 0.0), moves.get("gap"),
                                moves.get("overlap"))
            out.append(element)
        return out

    def settings(self) -> Settings:
        return Settings(self.wall_spacing, self.cells_per_chord, self.layer_cells, self.layer_thickness,
                        self.far_field, self.leading_edge_spacing, self.trailing_edge_spacing, self.gap_cells,
                        self.even_cells, self.base_blocks)

    def resolve(self, base: Path, name: str | None = None) -> Path:
        """A file, relative to ``base`` (the case directory) or the working directory."""
        name = name or self.file
        for root in (base, Path.cwd()):
            path = Path(name) if Path(name).is_absolute() else root / name
            if path.is_file():
                return path
        raise FileNotFoundError(f"{name!r} not found in {base} or {Path.cwd()}")


def _generated(spec: GridSpec, base: Path, dtype) -> Mesh:
    airfoil, _ = spec.airfoil_shape(base)
    spacing = Spacing(spec.leading_edge_spacing, spec.trailing_edge_spacing)
    common = dict(cells_normal=spec.cells_normal, wall_spacing=spec.wall_spacing,
                  far_field=spec.far_field, spacing=spacing)
    if spec.grid == "o_grid":
        raw = [RawBlock(*o_grid(airfoil, spec.cells_airfoil, **common))]
    elif airfoil.sharp:
        raw = [RawBlock(*c_grid(airfoil, spec.cells_airfoil, spec.cells_wake, **common))]
    else:
        blocks = blunt_c_grid(airfoil, spec.cells_airfoil, spec.cells_wake, **common)
        raw = [RawBlock(x, y, name=name) for name, (x, y) in blocks.items()]
    mesh, _ = assemble(raw, dtype=dtype, source=spec.describe())
    return mesh


def _file(spec: GridSpec, base: Path, dtype) -> Mesh:
    path = spec.resolve(base)
    if is_3d(path, spec.format):
        mesh, _ = load_grid_3d(path, spec.format, spec.coordinate_scale, spec.rules(), dtype)
        return mesh
    mesh, _ = load_grid(path, spec.format, spec.coordinate_scale, spec.rules(), dtype)
    return mesh


def _multi_element(spec: GridSpec, base: Path, dtype) -> Mesh:
    raw, _ = multi_element(spec.placed_elements(base), spec.settings())
    mesh, _ = assemble(raw, dtype=dtype, source=spec.describe())
    return mesh


GRIDS: dict[str, Callable[[GridSpec, Path, object], Mesh]] = {
    "o_grid": _generated, "c_grid": _generated, "file": _file, "multi_element": _multi_element,
}
"""Grid sources by ``domain.grid``."""

PATCH_RULE_KEYS = {"name", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
ELEMENT_KEYS = {"role", "file", "deflection_deg", "gap", "overlap", "notes"}
