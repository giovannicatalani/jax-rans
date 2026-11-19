"""Variable-density manufactured diffusion for the conservative SA row."""

import jax.numpy as jnp
import numpy as np

from jaxrans.mesh.block import Grid, rectangle_sides
from jaxrans.numerics.turbulence import sa_residual
from jaxrans.physics import spalart_allmaras as sa
from jaxrans.physics.navier_stokes import Viscosity


def error(n: int) -> float:
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    grid = Grid.from_nodes(xx, yy, rectangle_sides(n, n, j_min="test", j_max="test",
                                                     i_min="test", i_max="test"))
    xc = (np.arange(-1, n + 1) + 0.5) / n
    X = np.broadcast_to(xc[:, None], (n + 2, n + 2))
    rho = 1 + 0.2 * X
    h = 1 + 0.3 * X + 0.1 * X**2
    p1 = jnp.asarray(np.stack([rho, np.zeros_like(X), np.zeros_like(X),
                               np.full_like(X, 1 / 1.4), h]))
    mu, nu_inf = 0.04, 0.02
    zeros_i = jnp.zeros((n + 1, n))
    zeros_j = jnp.zeros((n, n + 1))
    xi = np.broadcast_to(x[:, None], (n + 1, n))
    grad_i = jnp.stack([jnp.asarray(0.3 + 0.2 * xi), zeros_i])
    grad_j = jnp.stack([zeros_j, zeros_j])
    mass = (zeros_i, zeros_j)
    R = np.asarray(sa_residual(p1, mass, (grad_i, grad_j), grid,
                               Viscosity(mu), 1.4, sa.SAModel(nu_inf),
                               jnp.ones((n, n))))
    Xc = (np.arange(n) + 0.5) / n
    rho_c = 1 + 0.2 * Xc
    h_c = 1 + 0.3 * Xc + 0.1 * Xc**2
    h_prime = 0.3 + 0.2 * Xc
    nu = mu / rho_c
    P, D = sa.source(jnp.asarray(nu_inf * h_c), jnp.asarray(nu),
                     jnp.zeros(n), jnp.ones(n))
    source = rho_c * ((np.asarray(P - D)) / nu_inf +
                      sa.SA.c_b2 / sa.SA.sigma * nu_inf * h_prime**2)
    A = mu / rho_c + nu_inf * h_c
    A_prime = -mu * 0.2 / rho_c**2 + nu_inf * h_prime
    expected = -rho_c * (A_prime * h_prime + A * 0.2) / sa.SA.sigma
    observed = R / np.asarray(grid.volume) + source[:, None]
    return float(np.sqrt(np.mean((observed[2:-2] - expected[2:-2, None]) ** 2)))


def test_variable_density_sa_diffusion_refines_at_second_order() -> None:
    errors = [error(n) for n in (16, 32, 64)]
    assert errors[0] / errors[1] > 3.0
    assert errors[1] / errors[2] > 3.0
