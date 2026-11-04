"""Geometric invariants and known piecewise wing geometry, independent of CFD."""

import numpy as np
import pytest

from jaxrans.geometry.airfoil import naca4, read_hug
from jaxrans.geometry.icms import normalize_section, read_icms, write_hug
from scripts.generate_icms_example import generate


def transformed(sharp=False, reverse=False, pitch=57., dihedral=30.):
    foil = naca4("0018", n=51, sharp=sharp)
    a, d = np.radians([pitch, dihedral])
    chord = np.array([np.cos(a), -np.sin(a)*np.sin(d), np.sin(a)*np.cos(d)])
    up = np.array([-np.sin(a), -np.cos(a)*np.sin(d), np.cos(a)*np.cos(d)])
    basis = np.column_stack((chord, up))
    points = np.array([12., 7., 3.]) + 6.*np.column_stack((foil.x, foil.y)) @ basis.T
    return foil, points[::-1] if reverse else points


@pytest.mark.parametrize("sharp", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
def test_rigid_transform_normalization_and_inverse(sharp, reverse, tmp_path):
    foil, points = transformed(sharp, reverse)
    s = normalize_section(points)
    np.testing.assert_allclose(s.reconstruct(), points, atol=1e-13)
    np.testing.assert_allclose(s.basis.T @ s.basis, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(s.normalized[s.leading_edge_index], [0, 0], atol=1e-14)
    np.testing.assert_allclose((s.normalized[0]+s.normalized[-1])/2, [1, 0], atol=1e-14)
    np.testing.assert_allclose(s.airfoil.x, foil.x, atol=1e-14)
    np.testing.assert_allclose(s.airfoil.y, foil.y, atol=1e-14)
    assert s.chord == pytest.approx(6)
    assert s.chord_pitch_deg == pytest.approx(57)
    assert s.basis[2, 1] > 0
    assert s.leading_edge_index != np.argmin(points[:, 0])
    path = tmp_path / "foil.hug"
    write_hug(s, path)
    got, = read_hug(path)
    np.testing.assert_allclose(got.x, s.airfoil.x)
    np.testing.assert_allclose(got.y, s.airfoil.y)
    assert got.sharp == sharp


@pytest.mark.parametrize("cut", ["dihedral_normal", "line_of_flight"])
def test_synthetic_wing_and_incoming_sweep_at_kink(tmp_path, cut):
    path = tmp_path / "wing.icms"
    generate(path, cut=cut, n=31)
    sections = read_icms(path, 10, "m")
    y = np.r_[np.linspace(0, 13, 9), 15]
    heights = np.minimum(y, 13)*np.tan(np.radians(5)) + np.maximum(y-13, 0)*np.tan(np.radians(30))
    np.testing.assert_allclose([s.chord for s in sections], 6-4*y/15)
    np.testing.assert_allclose([s.leading_edge for s in sections], np.c_[np.zeros(10), y, heights], atol=1e-14)
    assert sections[8].sweep_segment == (7, 8)  # Y=13 takes the inboard segment
    assert sections[9].sweep_segment == (8, 9)
    inner = np.degrees(np.arctan((-1/15)*np.cos(np.radians(5))))
    outer = np.degrees(np.arctan((-1/15)*np.cos(np.radians(30))))
    np.testing.assert_allclose([s.sweep_deg for s in sections[:9]], inner, atol=1e-13)
    assert sections[9].sweep_deg == pytest.approx(outer)
    assert sections[0].sweep_segment == (0, 1)
    assert sections[0].airfoil.trailing_edge_gap / sections[-1].airfoil.trailing_edge_gap == pytest.approx(1.8)
    for s, yi in zip(sections, y):
        expected = naca4("0010", n=31, sharp=False)
        np.testing.assert_allclose(s.airfoil.y, expected.y*(0.18-0.08*yi/15)/0.10, atol=1e-14)


def test_units_and_header_lines(tmp_path):
    m, mm = tmp_path / "m.icms", tmp_path / "mm.icms"
    generate(m, n=21)
    generate(mm, units="mm", n=21)
    a, b = read_icms(m, 10, "m"), read_icms(mm, 10, "mm", header_lines=5)
    for sa, sb in zip(a, b):
        np.testing.assert_allclose(sa.points, sb.points, atol=1e-14)
        np.testing.assert_allclose(sa.normalized, sb.normalized, atol=1e-14)
    mm.write_text("1 2 3\n" + mm.read_text().split("\n", 5)[5])
    with pytest.raises(ValueError, match="header_lines"):
        read_icms(mm, 10, "mm")
    assert len(read_icms(mm, 10, "mm", header_lines=1)) == 10


def test_bad_geometry_and_order_are_rejected(tmp_path):
    _, p = transformed()
    p[12] += np.array([0, 0.1, 0])
    with pytest.raises(ValueError, match="non-planar"):
        normalize_section(p)
    with pytest.raises(ValueError, match="collinear"):
        normalize_section(np.c_[np.r_[1., .5, 0., .5, 1.], np.zeros((5, 2))])
    path = tmp_path / "bad.icms"
    profiles = generate(path, n=21)
    with path.open("w") as stream:
        np.savetxt(stream, np.concatenate(profiles[::-1]))
    with pytest.raises(ValueError, match="root to tip"):
        read_icms(path, 10, "m")
    profiles[0][10, 0] = np.nan
    np.savetxt(path, np.concatenate(profiles))
    with pytest.raises(ValueError, match="finite"):
        read_icms(path, 10, "m")


def test_wrong_profile_count_is_not_silently_repaired(tmp_path):
    path = tmp_path / "wing.icms"
    generate(path, n=21)
    with pytest.raises(ValueError):
        read_icms(path, 9, "m")
