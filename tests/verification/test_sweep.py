"""A batched sweep is, point for point, the sequential runs it stands for:
a cold point is ``run``, a continued point ``run(initial=previous)``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxrans import load_case, run
from jaxrans.sweep import Point, at_point, chains_of, run_sweep

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ("case.domain.cells_airfoil=32", "case.domain.cells_normal=16",
            "numerical.time_integration.solver=implicit", "numerical.convergence.max_iters=200",
            "numerical.convergence.block_iters=10", "numerical.convergence.tol=1e-3",
            "numerical.convergence.forces_tol=null")


@pytest.fixture(scope="module")
def case():
    return load_case(ROOT / "examples/naca0012_euler", SETTINGS)


def sequential(case, grid, chain, continuation=True):
    out, previous = [], None
    for point in chain:
        r = run(at_point(case, point), grid, initial=previous if continuation else None)
        out.append(r)
        previous = r.U if r.history.reason in ("converged", "settled") else None
    return out


@pytest.mark.parametrize("continuation", [True, False])
def test_sweep_matches_sequential_runs(case, continuation) -> None:
    """Two chains of unequal length (alpha at M 0.5, Mach at alpha 1) in one
    batch: each point equals its own sequential run, iterations included."""
    grid = case.grid()
    chains = [[Point(0.0, 0.5), Point(1.0, 0.5), Point(2.0, 0.5)], [Point(1.0, 0.45), Point(1.0, 0.55)]]
    sweep = run_sweep(case, chains, grid, continuation=continuation)
    assert sweep.points == tuple(p for chain in chains for p in chain)
    k = 0
    for chain in chains:
        for point, single in zip(chain, sequential(case, grid, chain, continuation)):
            batched = sweep.results[k]
            assert batched.history.reason == single.history.reason
            assert batched.history.n_iters == single.history.n_iters
            np.testing.assert_allclose(batched.U, single.U, rtol=1e-9, atol=1e-12)
            assert batched.forces.cl == pytest.approx(single.forces.cl, rel=1e-9, abs=1e-12)
            assert sweep.continued[k] == (continuation and point is not chain[0])
            k += 1


def test_fewer_slots_than_chains_queue_the_rest(case) -> None:
    chains = [[Point(a, 0.5)] for a in (0.0, 1.0, 2.0)]
    one = run_sweep(case, chains, slots=1)
    three = run_sweep(case, chains)
    assert one.timing.slots == 1 and three.timing.slots == 3
    for a, b in zip(one.results, three.results):
        np.testing.assert_allclose(a.U, b.U, rtol=1e-9, atol=1e-12)


def test_cli_arrangement() -> None:
    assert chains_of([0, 1, 2, 3], [0.5]) == [[Point(a, 0.5) for a in (0, 1, 2, 3)]]
    assert chains_of([1], [0.5, 0.6]) == [[Point(1, 0.5), Point(1, 0.6)]]
    assert chains_of([0, 1], [0.5, 0.6]) == [[Point(0, 0.5), Point(1, 0.5)], [Point(0, 0.6), Point(1, 0.6)]]
    assert chains_of([0, 1, 2, 3], [0.5], split=2) == [[Point(0, 0.5), Point(1, 0.5)], [Point(2, 0.5), Point(3, 0.5)]]
