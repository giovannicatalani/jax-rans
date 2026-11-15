"""The .hug geometry, the C-grid round a blunt trailing edge, and base blocks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxrans.case import parse_case
from jaxrans.case.domain import GridSpec
from jaxrans.geometry.airfoil import naca4, read_hug
from jaxrans.geometry.elements import read_element, read_elements
from jaxrans.mesh.coarsen import coarsen
from jaxrans.mesh.generate import blunt_c_grid
from jaxrans.mesh.mesh import assign_conditions, interface_node_mismatch, patches, wall_faces
from jaxrans.mesh.quality import interface_growth, quality

CRV = Path(__file__).resolve().parents[2] / "validation/references/jaxa_30p30n/2010_30p30n_thik_te.crv"
ROLES = ("slat", "main", "flap")


def _hug(path, foils):
    rows = [str(len(foils))]
    for foil in foils:
        rows.append(str(len(foil.x)))
        rows += [f"{x:.16g} {y:.16g}" for x, y in zip(foil.x, foil.y)]
    path.write_text("\n".join(rows) + "\n")
    return path


def _blunt_wall(mesh):
    return assign_conditions(mesh, {"wall": "no_slip_wall", "farfield": "far_field"})


def test_hug_keeps_the_given_corners(tmp_path) -> None:
    source = [naca4("0012", sharp=False), naca4("2412", sharp=False)]
    got = read_hug(_hug(tmp_path / "section.hug", source))
    assert len(got) == 2
    for a, b in zip(got, source):
        np.testing.assert_allclose(np.c_[a.x, a.y], np.c_[b.x, b.y], atol=1e-15)
        assert not a.sharp


@pytest.mark.parametrize("content,match", [
    ("2\n5\n1 0\n", "fewer than 5 points"),
    ("1\n5\n1 0\n0 0\n0 1\n1 1\n1 2\n5\n1 0\n0 0\n0 1\n1 1\n1 2\n", "holds 2 elements"),
    ("1\n5\n1 0\n0 0\n0 1\n1 nan\n1 2\n", "all finite"),
    ("1\n5\n1 2\n0 1\n0 0\n1 0\n1 -1\n", "must run from the lower"),
])
def test_hug_refuses_a_bad_file(tmp_path, content, match) -> None:
    path = tmp_path / "bad.hug"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        read_hug(path)


def test_hug_elements_take_their_roles_front_to_back(tmp_path) -> None:
    slat, main, flap = read_elements(CRV, ROLES, 1.0 / 18.0)
    path = _hug(tmp_path / "section.hug", [main.airfoil, flap.airfoil, slat.airfoil])
    got = read_elements(path, ROLES)
    np.testing.assert_allclose(got[1].airfoil.x, main.airfoil.x, atol=1e-14)
    assert got[1].role == "main" and read_element(_hug(tmp_path / "one.hug", [flap.airfoil]), "flap").role == "flap"
    with pytest.raises(ValueError, match="the case lists 2"):
        read_elements(path, ROLES[:2])


@pytest.mark.parametrize("airfoil", ["naca0012_blunt", "naca2412_blunt"])
def test_blunt_c_grid_keeps_the_base_and_is_conforming_and_smooth(tmp_path, airfoil) -> None:
    spec = GridSpec(grid="c_grid", airfoil=airfoil, cells_airfoil=128, cells_wake=64, cells_normal=64,
                    wall_spacing=1e-5, far_field=20.0)
    mesh = spec.build(tmp_path)
    assert len(mesh) == 3
    assert set(patches(mesh)) == {"wall", "farfield"} and interface_node_mismatch(mesh) < 1e-12
    wall = _blunt_wall(mesh)
    faces = wall_faces(wall)
    np.testing.assert_allclose((faces.b - faces.a).sum(axis=0), 0.0, atol=1e-12)
    report = quality(wall)
    assert faces.bodies == 1 and not report.folded
    assert report.interface_growth_max == pytest.approx(interface_growth(wall).max())
    assert report.interface_growth_location.startswith("block ")
    assert report.interface_growth_max < 12.0
    assert report.wall_spacing == pytest.approx((1e-5, 1e-5), rel=0.15)
    assert all(g.ni >= 2 and g.nj >= 2 for g in coarsen(coarsen(wall)))


def test_blunt_c_grid_keeps_the_surface_resolution_of_a_sharp_one() -> None:
    foil = naca4("0012", sharp=False).unit_chord()
    x, y = blunt_c_grid(foil, 256, 64, 96, 5e-6)["body"]
    wall = x[:, 0] < 1.0 - 1e-12
    upper = wall & (y[:, 0] > 0.0)
    assert np.count_nonzero((x[upper, 0] > 0.4) & (x[upper, 0] < 0.8)) >= 30


def test_blunt_wake_interface_jump_is_reduced_on_the_reference_grid(tmp_path) -> None:
    spec = GridSpec(grid="c_grid", airfoil="naca0012_blunt", cells_airfoil=256,
                    cells_wake=64, cells_normal=96, wall_spacing=5e-6, far_field=100)
    mesh = _blunt_wall(spec.build(tmp_path))
    report = quality(mesh)
    assert report.interface_growth_max < 10.0
    assert report.min_area > 0.0
    assert all(b.nj % 4 == 0 for b in mesh)


def test_a_closed_blunt_section_gets_the_one_block_c_grid(tmp_path) -> None:
    spec = GridSpec(grid="c_grid", airfoil=_hug(tmp_path / "a.hug", [naca4("0012", sharp=False)]).name,
                    trailing_edge="close", cells_airfoil=64, cells_wake=32, cells_normal=32, wall_spacing=1e-4)
    assert len(spec.build(tmp_path)) == 1


def test_a_blunt_c_grid_is_usable_as_a_swept_section(tmp_path) -> None:
    domain = {"grid": "c_grid", "airfoil": "naca0012_blunt", "cells_airfoil": 64, "cells_wake": 32,
              "cells_normal": 32, "wall_spacing": 1e-4, "far_field": 20.0}
    case = parse_case({"general": {"case_name": "swept"}, "domain": domain,
                       "freestream": {"mach": 0.5, "sweep_deg": 30.0}}, {}, tmp_path)
    assert case.grid().components == 3


@pytest.mark.slow
def test_base_blocks_split_each_blunt_layer_on_the_same_wall(tmp_path) -> None:
    elements = read_elements(CRV, ROLES, 1.0 / 18.0)
    path = _hug(tmp_path / "section.hug", [e.airfoil for e in elements])
    plain, split = (GridSpec(grid="multi_element", file=path.name, elements=tuple({"role": r} for r in ROLES),
                             wall_spacing=1.5e-5, cells_per_chord=150, base_blocks=flag).build(tmp_path)
                    for flag in (False, True))
    assert len(split) == len(plain) + 3 and interface_node_mismatch(split) < 1e-12
    assert sum(g.ni * g.nj for g in split) == sum(g.ni * g.nj for g in plain)
    faces = wall_faces(_blunt_wall(split))
    assert faces.bodies == 3 and not quality(_blunt_wall(split)).folded
    for k in range(3):
        assert len(set(faces.block[faces.body == k])) == 2
    coarsen(coarsen(_blunt_wall(split)))
