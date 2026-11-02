"""The counted .hug geometry and its finite-base C-grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxrans.case import parse_case
from jaxrans.case.domain import GridSpec
from jaxrans.geometry.airfoil import Airfoil, naca4
from jaxrans.geometry.elements import read_element, read_elements, read_hug
from jaxrans.mesh.coarsen import coarsen
from jaxrans.mesh.mesh import assign_conditions, interface_node_mismatch, patches, wall_faces
from jaxrans.mesh.quality import quality


def _hug(path, foils):
    rows = [str(len(foils))]
    for foil in foils:
        rows.append(str(len(foil.x)))
        rows += [f"{x:.16g} {y:.16g}" for x, y in zip(foil.x, foil.y)]
    path.write_text("\n".join(rows) + "\n")


def test_hug_reads_open_bases_without_changing_endpoints(tmp_path) -> None:
    source = [naca4("0012", sharp=False), naca4("2412", sharp=False)]
    path = tmp_path / "section.hug"
    _hug(path, source)
    got = read_hug(path)
    assert len(got) == 2
    for a, b in zip(got, source):
        np.testing.assert_allclose(np.c_[a.x, a.y], np.c_[b.x, b.y], atol=1e-15)
        assert not a.sharp
        assert a.trailing_edge_gap == pytest.approx(b.trailing_edge_gap)


@pytest.mark.parametrize("content,match", [
    ("2\n5\n1 0\n", "fewer than 5 points"),
    ("1\n5\n1 0\n0 0\n0 1\n1 1\n1 2\nextra\n", "unexpected data"),
    ("1\n5\n1 0\n0 0\n0 1\n1 nan\n1 2\n", "non-finite"),
])
def test_hug_rejects_invalid_counts_and_coordinates(tmp_path, content, match) -> None:
    path = tmp_path / "bad.hug"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        read_hug(path)


def test_hug_refuses_a_nonzero_base_below_the_sharp_edge_tolerance(tmp_path) -> None:
    path = tmp_path / "tiny.hug"
    foil = naca4("0012", sharp=False)
    y = foil.y.copy()
    y[0], y[-1] = -1e-8, 1e-8
    _hug(path, [Airfoil("tiny", foil.x, y)])
    with pytest.raises(ValueError, match="below the geometry tolerance"):
        read_hug(path)


def test_a_single_hug_element_keeps_its_explicit_corners(tmp_path) -> None:
    path = tmp_path / "flap.hug"
    foil = naca4("2412", sharp=False)
    _hug(path, [foil])
    got = read_element(path, "flap", 0.3)
    np.testing.assert_allclose(got.airfoil.x, 0.3 * foil.x, atol=1e-15)
    np.testing.assert_allclose(got.airfoil.y, 0.3 * foil.y, atol=1e-15)


def test_blunt_c_grid_preserves_the_base_and_resolves_its_wall_layer(tmp_path) -> None:
    foil = naca4("0012", sharp=False)
    path = tmp_path / "clean.hug"
    _hug(path, [foil])
    spec = GridSpec(grid="c_grid", airfoil=path.name, trailing_edge_topology="blunt_c",
                    cells_airfoil=128, cells_wake=64, cells_normal=64,
                    wall_spacing=1e-5, far_field=20.0)
    mesh = spec.build(tmp_path)
    assert len(mesh) == 3
    assert set(patches(mesh)) == {"wall", "farfield"}
    assert interface_node_mismatch(mesh) < 1e-12
    wall = assign_conditions(mesh, {"wall": "no_slip_wall", "farfield": "far_field"})
    faces = wall_faces(wall)
    np.testing.assert_allclose((faces.b - faces.a).sum(axis=0), 0.0, atol=1e-12)
    q = quality(wall)
    assert faces.bodies == 1 and not q.folded
    assert q.min_angle_deg > 20.0 and q.max_growth < 1.5
    assert q.wall_spacing[0] == pytest.approx(1e-5, rel=0.1)
    assert q.wall_spacing[1] == pytest.approx(1e-5, rel=0.1)
    base = (np.isclose(faces.a[:, 0], 1.0, atol=1e-12)
            & np.isclose(faces.b[:, 0], 1.0, atol=1e-12))
    assert base.sum() >= 2
    assert sum(np.hypot(*(faces.b[base] - faces.a[base]).T)) == pytest.approx(foil.trailing_edge_gap)
    assert len(coarsen(coarsen(wall))) == 3


def test_blunt_c_option_never_closes_the_hug_base(tmp_path) -> None:
    path = tmp_path / "clean.hug"
    _hug(path, [naca4("0012", sharp=False)])
    spec = GridSpec(grid="c_grid", airfoil=path.name, trailing_edge_topology="blunt_c",
                    trailing_edge="close", cells_airfoil=64, cells_wake=32,
                    cells_normal=32, wall_spacing=1e-4)
    with pytest.raises(ValueError, match="preserves the trailing-edge base"):
        spec.build(tmp_path)


def test_blunt_c_grid_also_handles_an_asymmetric_base(tmp_path) -> None:
    spec = GridSpec(grid="c_grid", airfoil="naca2412_blunt", trailing_edge_topology="blunt_c",
                    cells_airfoil=128, cells_wake=64, cells_normal=64,
                    wall_spacing=1e-5, far_field=20.0)
    mesh = spec.build(tmp_path)
    mesh = assign_conditions(mesh, {"wall": "no_slip_wall", "farfield": "far_field"})
    report = quality(mesh)
    assert not report.folded and report.min_angle_deg > 20.0
    assert report.wall_spacing[1] == pytest.approx(1e-5, rel=0.1)


def test_hug_elements_keep_their_order_and_bases(tmp_path) -> None:
    reference = Path(__file__).resolve().parents[2] / "validation/references/jaxa_30p30n/2010_30p30n_thik_te.crv"
    original = read_elements(reference, ["slat", "main", "flap"], 1.0 / 18.0)
    path = tmp_path / "highlift.hug"
    _hug(path, [e.airfoil for e in original])
    spec = GridSpec(grid="multi_element", file=path.name,
                    elements=tuple({"role": r} for r in ("slat", "main", "flap")),
                    wall_spacing=1.5e-5, cells_per_chord=150)
    loaded = spec.placed_elements(tmp_path)
    assert [e.role for e in loaded] == ["slat", "main", "flap"]
    for got, source in zip(loaded, original):
        np.testing.assert_allclose(np.c_[got.airfoil.x, got.airfoil.y],
                                   np.c_[source.airfoil.x, source.airfoil.y], atol=1e-14)
        assert got.airfoil.trailing_edge_gap == pytest.approx(source.airfoil.trailing_edge_gap)


def test_hug_blunt_c_grid_is_usable_as_a_swept_section(tmp_path) -> None:
    path = tmp_path / "clean.hug"
    _hug(path, [naca4("0012", sharp=False)])
    domain = {"grid": "c_grid", "airfoil": path.name, "trailing_edge_topology": "blunt_c",
              "cells_airfoil": 64, "cells_wake": 32, "cells_normal": 32,
              "wall_spacing": 1e-4, "far_field": 20.0}
    case = parse_case({"general": {"case_name": "swept_hug"}, "domain": domain,
                       "freestream": {"mach": 0.5, "sweep_deg": 30.0}}, {}, tmp_path)
    atlas = case.grid()
    assert atlas.components == 3 and len(atlas.mesh) == 3


@pytest.mark.slow
def test_three_blunt_hug_elements_build_with_three_closed_walls(tmp_path) -> None:
    reference = Path(__file__).resolve().parents[2] / "validation/references/jaxa_30p30n/2010_30p30n_thik_te.crv"
    elements = read_elements(reference, ["slat", "main", "flap"], 1.0 / 18.0)
    path = tmp_path / "highlift.hug"
    _hug(path, [e.airfoil for e in elements])
    spec = GridSpec(grid="multi_element", file=path.name,
                    elements=tuple({"role": r} for r in ("slat", "main", "flap")),
                    trailing_edge_topology="blunt_c",
                    wall_spacing=1.5e-5, cells_per_chord=150)
    mesh = spec.build(tmp_path)
    wall = assign_conditions(mesh, {"wall": "no_slip_wall", "farfield": "far_field"})
    faces = wall_faces(wall)
    assert faces.bodies == 3 and not quality(wall).folded
    assert interface_node_mismatch(mesh) < 1e-12
    coarsen(coarsen(wall))
    for k, e in enumerate(elements):
        a = faces.a[faces.body == k]
        b = faces.b[faces.body == k]
        np.testing.assert_allclose((b - a).sum(axis=0), 0.0, atol=1e-12)
        assert len(set(faces.block[faces.body == k])) == 2
        start = np.array([e.airfoil.x[0], e.airfoil.y[0]])
        stop = np.array([e.airfoil.x[-1], e.airfoil.y[-1]])
        direction = (stop - start) / np.linalg.norm(stop - start)
        normal = np.array([-direction[1], direction[0]])
        base = (np.abs((a - start) @ normal) < 1e-12) & (np.abs((b - start) @ normal) < 1e-12)
        assert base.sum() >= 2
        assert np.linalg.norm(b[base] - a[base], axis=1).sum() == pytest.approx(e.airfoil.trailing_edge_gap)
