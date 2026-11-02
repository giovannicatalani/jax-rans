"""The validation layer (validation/validate.py): every validation case loads,
its levels build the grid family it documents, and the judging helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from jaxrans import load_case
from jaxrans.post.forces import Forces
from jaxrans.solvers.driver import ConvergenceHistory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation"))
import validate as v  # noqa: E402

NAMES = sorted(d.name for d in (ROOT / "validation" / "cases").iterdir() if d.is_dir())


@pytest.mark.parametrize("name", NAMES)
def test_every_validation_case_and_level_parses(name) -> None:
    val = v.load_validation(name)
    assert val.levels and val.suite
    for level in val.levels:
        case = load_case(val.directory, v.level_overrides(level))
        assert case.name == name


def test_suites_are_ordered() -> None:
    assert [c.name for c in v.suite("rans")] == ["tmr_flatplate_sa", "tmr_naca0012_sa_a10"]
    with pytest.raises(KeyError, match="rans"):
        v.suite("nope")


def test_levels_are_selected_by_index_or_name() -> None:
    levels = v.load_validation("tmr_naca0012_sa_a10").levels
    assert v.select_levels(levels, "0,2") == [levels[0], levels[2]]
    assert v.select_levels(levels, "225-65") == [levels[1]]
    assert v.select_levels(levels, None) == list(levels)


def test_richardson_recovers_order_and_limit() -> None:
    f_star, c, p = 0.0553, 0.4, 1.7
    fit = v.richardson(tuple(f_star + c * h**p for h in (1.0, 0.5, 0.25)))
    assert fit.monotone and fit.order == pytest.approx(p, rel=1e-12)
    assert fit.extrapolated == pytest.approx(f_star, rel=1e-12)
    assert np.isnan(v.richardson((1.0, 0.9, 0.95)).order)


def judged(cl_history):
    from jaxrans.simulation import Result

    case = load_case(ROOT / "validation" / "cases" / "tmr_naca0012_sa_a10")
    result = Result(case, None, None, Forces(cl=1.091, cd=0.012273, cm=0.0, fx=0.0, fy=0.0),
                    ConvergenceHistory(residual=np.ones(3), reason="converged", n_iters=3),
                    np.asarray(cl_history), 0.0)
    return v.Judged(result, v.load_validation("tmr_naca0012_sa_a10").reference, 1e-4)


def test_a_short_run_never_counts_as_settled() -> None:
    assert not judged([1.091, 1.091]).settled
    assert judged([1.091] * 16).settled
    assert not judged([1.091] * 12 + [1.091, 1.1, 1.091, 1.1]).settled
    assert judged([1.091] * 16).failures() == []
