"""CGNS grid files (HDF5 storage), structured zones, read with h5py.

Per zone: the node coordinates (of a 3D zone one cell thick, the plane of
its thin index, as for Plot3D; with ``dim=3``, the whole 3D zone), the
boundary faces of every ``BC_t`` given by a ``PointRange``, named by the BC's
family, else by its type (``BCWall*``: ``wall``, ``BCFarfield``:
``farfield``, ...), else by the node's name; and the 1-to-1 interfaces, which
the assembly checks its face matching against. Files in the older ADF
storage are refused with the conversion to use.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
ADF_SIGNATURE = b"ADF Database"

BC_TYPE_PATCHES = {
    "BCWall": "wall", "BCWallViscous": "wall", "BCWallViscousHeatFlux": "wall",
    "BCWallViscousIsothermal": "wall", "BCWallInviscid": "wall",
    "BCFarfield": "farfield", "BCSymmetryPlane": "symmetry",
    "BCInflow": "inflow", "BCInflowSubsonic": "inflow", "BCTunnelInflow": "inflow",
    "BCOutflow": "outflow", "BCOutflowSubsonic": "outflow", "BCTunnelOutflow": "outflow",
    "BCExtrapolate": "outflow",
}
"""Patch names for the standard CGNS boundary types, when a BC has no family."""


@dataclass(frozen=True)
class Declared:
    """A face range a file declares: a boundary ``name``, or (``name`` None)
    a 1-to-1 interface to zone ``donor``. Cells ``[start, stop)`` along
    ``side``, in the file's own index order: integers on a 2D block, pairs in
    the side's two running indices (mesh/block3d.py) on a 3D one."""

    side: str
    start: int | tuple[int, int]
    stop: int | tuple[int, int]
    name: str | None = None
    donor: int | None = None


@dataclass(frozen=True)
class RawBlock:
    """A block as read: node coordinates ``(n_i, n_j)`` and what the file declares."""

    x: np.ndarray
    y: np.ndarray
    declared: tuple[Declared, ...] = ()
    name: str = ""


@dataclass(frozen=True)
class RawBlock3D:
    """A 3D block as read: nodes ``(3, n_i, n_j, n_k)`` and what the file declares."""

    nodes: np.ndarray
    declared: tuple[Declared, ...] = ()
    name: str = ""


def _bytes(path: Path) -> bytes:
    data = Path(path).read_bytes()
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def is_cgns(path: Path) -> bool:
    head = _bytes(path)[:64]
    return head.startswith(HDF5_SIGNATURE) or ADF_SIGNATURE in head


def _label(node) -> str:
    value = node.attrs.get("label", b"")
    return (value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)).strip("\x00 ")


def _data(node):
    return node[" data"][()] if " data" in node else None


def _text(node) -> str:
    d = _data(node)
    return "" if d is None else d.tobytes().decode(errors="replace").strip("\x00 ")


def _children(node, label: str):
    return [(k, node[k]) for k in node if not k.startswith(" ") and _label(node[k]) == label]


def _zones(path: Path):
    """The open file and the ``Zone_t`` nodes of its first base."""
    import h5py

    data = _bytes(path)
    if ADF_SIGNATURE in data[:64]:
        raise ValueError(f"{path.name} is CGNS in the old ADF storage; convert it to HDF5 with the CGNS "
                         f"tools: adf2hdf {path.name} out.cgns (or cgnsconvert -h)")
    f = h5py.File(io.BytesIO(data), "r")
    bases = _children(f, "CGNSBase_t")
    if not bases:
        raise ValueError(f"{path.name}: no CGNSBase_t")
    return f, _children(bases[0][1], "Zone_t")


def _vertices(zone) -> np.ndarray:
    return np.atleast_2d(_data(zone))[0]                       # vertex counts per index


def is_3d_cgns(path: str | Path) -> bool:
    """True for a CGNS file whose zones are 3D with more than two nodes in
    every direction (not thin extrusions of a 2D grid)."""
    f, zones = _zones(Path(path))
    with f:
        return all(len(_vertices(z)) == 3 and _vertices(z).min() > 2 for _, z in zones)


def read_cgns(path: str | Path, dim: int = 2) -> list[RawBlock] | list[RawBlock3D]:
    """Every structured zone of the first base, as a 2D block (``dim=2``) or
    as a whole 3D block (``dim=3``).

    Raises:
        ValueError: ADF storage, an unstructured zone, a BC without a
            ``PointRange``, a 3D zone that is not a thin extrusion (2D), or a
            zone that is not 3D (3D).
    """
    path = Path(path)
    f, zones = _zones(path)
    with f:
        names = [name for name, _ in zones]
        return [_zone(name, zone, names, path, dim) for name, zone in zones]


def _zone(name: str, zone, zone_names: list[str], path: Path, dim: int) -> RawBlock | RawBlock3D:
    types = _children(zone, "ZoneType_t")
    if types and _text(types[0][1]) != "Structured":
        raise ValueError(f"{path.name}: zone {name!r} is {_text(types[0][1])}; only structured zones are read")
    size = _vertices(zone)
    (_, coords), = _children(zone, "GridCoordinates_t")[:1]
    xyz = [np.asarray(_data(coords[f"Coordinate{c}"]), dtype=np.float64).T if f"Coordinate{c}" in coords
           else None for c in "XYZ"]
    if dim == 3:
        if len(size) != 3 or any(c is None for c in xyz):
            raise ValueError(f"{path.name}: zone {name!r} is not a 3D zone")
        face = lambda point_range: _face_3d(point_range, size)  # noqa: E731
    elif len(size) == 2:
        x, y, axes = xyz[0], xyz[1], (0, 1)
        face = lambda point_range: _face(point_range, axes)  # noqa: E731
    else:
        x, y, axes = _thin_plane(xyz, name, path)
        face = lambda point_range: _face(point_range, axes)  # noqa: E731
    declared = []
    for _, zbc in _children(zone, "ZoneBC_t"):
        for bc_name, bc in _children(zbc, "BC_t"):
            ranges = _children(bc, "IndexRange_t")
            if not ranges:
                raise ValueError(f"{path.name}: BC {bc_name!r} of zone {name!r} has no PointRange")
            cells = face(np.asarray(_data(ranges[0][1])))
            if cells is None:
                continue                                       # a face across the thin index
            family = _children(bc, "FamilyName_t")
            patch = (_text(family[0][1]) if family else None) or BC_TYPE_PATCHES.get(_text(bc)) or bc_name.strip()
            declared.append(Declared(*cells, name=patch))
    for _, zgc in _children(zone, "ZoneGridConnectivity_t"):
        for _, conn in _children(zgc, "GridConnectivity1to1_t"):
            ranges = dict(_children(conn, "IndexRange_t"))
            cells = face(np.asarray(_data(ranges["PointRange"])))
            if cells is not None:
                donor = _text(conn)
                declared.append(Declared(*cells, donor=zone_names.index(donor) if donor in zone_names else None))
    if dim == 3:
        return RawBlock3D(np.stack(xyz), tuple(declared), name)
    return RawBlock(x, y, tuple(declared), name)


def _thin_plane(xyz, name, path):
    """The plane of a 3D zone's thin index, in its two in-plane coordinates."""
    coords = [c for c in xyz if c is not None]
    scale = max(float(np.abs(c).max()) for c in coords) or 1.0
    for axis in range(3):
        if coords[0].shape[axis] > 2:
            continue
        first = [np.take(c, 0, axis) for c in xyz]
        last = [np.take(c, -1, axis) for c in xyz]
        keep = [k for k in range(3) if xyz[k] is not None
                and np.abs(first[k] - last[k]).max() <= 1e-10 * scale and np.ptp(first[k]) > 1e-10 * scale]
        if len(keep) == 2:
            return first[keep[0]], first[keep[1]], tuple(a for a in range(3) if a != axis)
    raise ValueError(f"{path.name}: zone {name!r} is 3D and not a thin extrusion of a 2D plane")


def _face_3d(point_range: np.ndarray, size) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """``(side, start, stop)`` cells of a node range ``(2, 3)`` (begin, end;
    1-based) on a side of a 3D block, in the side's running indices."""
    begin, end = np.asarray(point_range, dtype=np.int64) - 1
    constant = [a for a in range(3) if begin[a] == end[a]]
    if len(constant) != 1 or begin[constant[0]] not in (0, size[constant[0]] - 1):
        raise ValueError(f"the node range {point_range.tolist()} is not a face of a block side")
    n = constant[0]
    side = "ijk"[n] + ("_min" if begin[n] == 0 else "_max")
    running = [a for a in range(3) if a != n]
    lo = tuple(int(min(begin[a], end[a])) for a in running)
    hi = tuple(int(max(begin[a], end[a])) for a in running)
    return side, lo, hi


def _face(point_range: np.ndarray, axes: tuple[int, int]) -> tuple[str, int, int] | None:
    """``(side, start, stop)`` cells of a node range ``(2, ndim)`` (begin,
    end; 1-based) in the 2D block of index axes ``axes``; None for a face
    across the dropped index."""
    begin, end = point_range[0], point_range[1]
    a, b = axes
    for constant, along, sides in ((a, b, ("i_min", "i_max")), (b, a, ("j_min", "j_max"))):
        if begin[constant] == end[constant] and begin[along] != end[along]:
            side = sides[0] if begin[constant] == 1 else sides[1]
            lo, hi = sorted((int(begin[along]), int(end[along])))
            return side, lo - 1, hi - 1
    return None
