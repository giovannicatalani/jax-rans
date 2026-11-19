"""Physical wall distance, independently checked against finite triangles."""

import numpy as np
import pytest

from jaxrans.core.sectional import SectionalSpec
from jaxrans.geometry.conical import ConicalGeometry
from jaxrans.mesh.conical_wall import conical_panel_distance
from jaxrans.mesh.wall3d import triangle_distance


@pytest.mark.parametrize("taper", [-0.3, 0.3])
def test_distance_matches_the_whole_positive_cone_sector(taper):
    geom = ConicalGeometry(SectionalSpec(leading_edge_sweep_deg=30, taper_rate=taper), 1., 0.)
    a, b = np.array([[.2, .1]]), np.array([[.9, .3]])
    p, mu = geom.coefficients
    apex = np.array([p / mu, 0., 1 / mu])
    # A large triangle covers the relevant part of the unbounded sector.
    A = apex + 100 * (np.r_[a[0], 0.] - apex)
    B = apex + 100 * (np.r_[b[0], 0.] - apex)
    points = np.random.default_rng(42).uniform(-4, 4, (100, 3))
    points = np.concatenate([points, [apex + [0, 0, 1], apex]])
    expected = triangle_distance(points, apex, A, B)
    np.testing.assert_allclose(conical_panel_distance(points, a, b, geom)[:, 0], expected,
                               rtol=2e-12, atol=2e-12)


def test_zero_taper_is_an_infinite_swept_extrusion_not_a_thin_slab():
    geom = ConicalGeometry(SectionalSpec(leading_edge_sweep_deg=30, taper_rate=0.), 1., 0.)
    points = np.array([[.5, .3, 1000], [-.2, .3, -1000], [1.2, .3, 0]])
    got = conical_panel_distance(points, np.array([[0., 0.]]), np.array([[1., 0.]]), geom)
    np.testing.assert_allclose(got[:, 0], [.3, np.hypot(.2, .3), np.hypot(.2, .3)], atol=1e-12)


def test_small_taper_distance_is_continuous_without_a_remote_apex_subtraction():
    points = np.array([[.5, .3, 0.], [-.2, .3, 0.], [1.2, .3, 0.]])
    a, b = np.array([[0., 0.]]), np.array([[1., 0.]])
    values = []
    for taper in [0., 1e-6, 1e-9, 1e-12]:
        geom = ConicalGeometry(SectionalSpec(leading_edge_sweep_deg=30, taper_rate=taper), 1., 0.)
        values.append(conical_panel_distance(points, a, b, geom))
    assert np.max(abs(values[-1] - values[0])) < 1e-12
    assert np.max(abs(values[2] - values[0])) < np.max(abs(values[1] - values[0]))
