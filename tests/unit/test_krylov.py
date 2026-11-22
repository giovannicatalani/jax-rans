"""Right-preconditioned GMRES: exact on small systems, and never worse than
the zero vector -- the property left preconditioning lacks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.solvers.krylov import gmres_right


def system(n: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = np.eye(n) * 4.0 + rng.standard_normal((n, n))
    return A, rng.standard_normal(n)


def test_full_subspace_solves_exactly() -> None:
    """With ``restart = n`` the Krylov space is everything: the exact solution."""
    A, b = system()
    x = gmres_right(lambda v: jnp.asarray(A) @ v, jnp.asarray(b), lambda r: r,
                    restart=24, maxiter=1, tol=1e-14)
    assert np.allclose(A @ np.asarray(x), b, atol=1e-10)


def test_an_exact_preconditioner_converges_in_one_step() -> None:
    A, b = system()
    A_inv = jnp.asarray(np.linalg.inv(A))
    x = gmres_right(lambda v: jnp.asarray(A) @ v, jnp.asarray(b), lambda r: A_inv @ r,
                    restart=1, maxiter=1, tol=1e-14)
    assert np.allclose(A @ np.asarray(x), b, atol=1e-10)


@pytest.mark.parametrize("restart", [1, 3, 6])
def test_the_true_residual_never_grows(restart: int) -> None:
    """Even with a bad preconditioner: GMRES minimizes |b - A x| over a space
    containing zero."""
    A, b = system(seed=3)
    bad = jnp.asarray(np.random.default_rng(9).standard_normal((24, 24)))
    x = gmres_right(lambda v: jnp.asarray(A) @ v, jnp.asarray(b), lambda r: bad @ r,
                    restart=restart, maxiter=2, tol=1e-14)
    assert np.linalg.norm(b - A @ np.asarray(x)) <= np.linalg.norm(b) * (1 + 1e-12)


def test_array_shaped_unknowns_and_zero_rhs() -> None:
    """Fields are ``(4, ni, nj)``, not vectors; a zero right-hand side gives zero."""
    rng = np.random.default_rng(1)
    d = jnp.asarray(1.0 + rng.random((4, 5, 3)))
    b = jnp.asarray(rng.standard_normal((4, 5, 3)))
    x = gmres_right(lambda v: d * v, b, lambda r: r, restart=8, maxiter=10, tol=1e-12)
    assert np.allclose(np.asarray(d * x), np.asarray(b), atol=1e-10)
    zero = gmres_right(lambda v: d * v, jnp.zeros_like(b), lambda r: r, restart=4, maxiter=2, tol=1e-12)
    assert not np.any(np.asarray(zero))


def test_givens_least_squares_matches_lstsq() -> None:
    from jaxrans.solvers.krylov import hessenberg_lstsq

    rng = np.random.default_rng(4)
    for m in (1, 4, 10):
        H = np.triu(rng.standard_normal((m + 1, m)), -1)
        e1 = np.zeros(m + 1)
        e1[0] = 1.7
        want = np.linalg.lstsq(H, e1, rcond=None)[0]
        got = np.asarray(hessenberg_lstsq(jnp.asarray(H), jnp.asarray(1.7), 1e-15 * (m + 1)))
        np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-13)


def test_givens_least_squares_at_a_krylov_breakdown() -> None:
    """``H[k, k-1] = 0``: the exact solution lies in the first ``k`` directions."""
    from jaxrans.solvers.krylov import hessenberg_lstsq

    rng = np.random.default_rng(5)
    H = np.triu(rng.standard_normal((9, 8)), -1)
    H[4, 3] = 0.0
    H[4:, 4:] = 0.0
    e1 = np.zeros(9)
    e1[0] = 1.0
    y = np.asarray(hessenberg_lstsq(jnp.asarray(H), jnp.asarray(1.0), 1e-15 * 9))
    assert np.all(np.isfinite(y)) and np.linalg.norm(e1 - H @ y) < 1e-14


@pytest.mark.parametrize("arnoldi", [False, True])
def test_returned_action_is_the_operator_on_the_solution(arnoldi: bool) -> None:
    A, b = system(seed=6)
    x, action = gmres_right(lambda v: jnp.asarray(A) @ v, jnp.asarray(b), lambda r: r / 4.0,
                            restart=6, maxiter=3, tol=1e-14, return_action=True,
                            action_from_arnoldi=arnoldi)
    np.testing.assert_allclose(np.asarray(action), A @ np.asarray(x), rtol=0, atol=1e-12 * np.linalg.norm(b))
    plain = gmres_right(lambda v: jnp.asarray(A) @ v, jnp.asarray(b), lambda r: r / 4.0,
                        restart=6, maxiter=3, tol=1e-14)
    np.testing.assert_allclose(np.asarray(x), np.asarray(plain), rtol=1e-12, atol=1e-14)
