"""Shared convergence semantics and diagnostics on tiny synthetic canvases."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans.case import CaseError, parse_case
from jaxrans.convergence import (
    convergence_record, diagnostics_report, equation_labels,
    equation_thresholds, equations_settled,
)
from jaxrans.core.config import ImplicitConfig, NumericsConfig
from jaxrans.numerics.residual import ResidualOperator
from jaxrans.simulation import run


def case_doc():
    return {"general": {"case_name": "test"},
            "domain": {"grid": "o_grid", "airfoil": "naca0012", "cells_airfoil": 32,
                       "cells_normal": 16, "wall_spacing": 1e-3},
            "freestream": {"mach": 0.5}}


def test_device_convergence_preserves_block_history_and_force_gate():
    case = parse_case(case_doc(), {"convergence": {
        "max_iters": 40, "block_iters": 2, "diagnostics": True,
        "forces_tol": 1.0, "equation_abs_tol": 1.0,
    }})
    grid = case.grid()
    host = run(case, grid)
    resident = run(case, grid, device_convergence=True)
    assert host.history.reason == resident.history.reason == "settled"
    assert host.history.n_iters == resident.history.n_iters == 26
    assert resident.convergence_history[-1]["force_tail_samples"] == 4
    assert "equations" in resident.convergence_history[-1]
    np.testing.assert_allclose(resident.history.residual, host.history.residual, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(resident.U, host.U, rtol=1e-10, atol=1e-12)


def test_config_roundtrip_named_floors_and_implicit_options():
    numerical = {"convergence": {"diagnostics": True, "equation_abs_tols": {"sa": 2e-5, "mass": 0}},
                 "implicit": {"nonlinear_merit": "equations", "backtracking_steps": 3}}
    case = parse_case(case_doc(), numerical)
    assert case.numerics.diagnostics
    assert dict(case.numerics.equation_abs_tols) == {"sa": 2e-5, "mass": 0.0}
    assert hash(case.numerics)
    assert case.implicit.nonlinear_merit == "equations"
    assert case.implicit.backtracking_steps == 3
    defaults = parse_case(case_doc(), {})
    assert not defaults.numerics.diagnostics
    assert defaults.implicit.nonlinear_merit == "mass"
    assert defaults.implicit.backtracking_steps == 0


@pytest.mark.parametrize("settings", [{"backtracking_steps": -1}, {"backtracking_steps": 17},
                                     {"backtracking_steps": 1.5}, {"backtracking_steps": True},
                                     {"nonlinear_merit": "energy"}])
def test_implicit_controls_reject_invalid_values(settings):
    with pytest.raises(ValueError):
        ImplicitConfig(**settings)


@pytest.mark.parametrize("overrides", [{"typo": 1e-5}, {"sa": -1}, {"sa": np.nan},
                                     {"sa": np.inf}, {"sa": True}])
def test_invalid_equation_floors(overrides):
    with pytest.raises(ValueError):
        NumericsConfig(equation_abs_tols=overrides)


@pytest.mark.parametrize("section,key,value", [("convergence", "diagnostics", 1),
                                              ("convergence", "equation_abs_tols", []),
                                              ("implicit", "backtracking_steps", True)])
def test_schema_checks_new_types(section, key, value):
    with pytest.raises(CaseError, match=key):
        parse_case(case_doc(), {section: {key: value}})


def test_threshold_overrides_with_ambiguous_five_variable_states():
    numerics = NumericsConfig(equation_abs_tol=1e-8, equation_abs_tols={"sa": 1e-3, "momentum_z": 2e-3})
    peak = jnp.ones(5) * 1e-6
    sa = np.asarray(equation_thresholds(peak, numerics, names=equation_labels(5, 2)))
    euler = np.asarray(equation_thresholds(peak, numerics, labels=equation_labels(5, 3)))
    assert sa[-1] == 1e-3
    assert euler[3] == 2e-3
    assert euler[-1] == 1e-8
    with pytest.raises(ValueError, match="explicit"):
        equation_thresholds(peak, numerics)


def test_strict_mode_ignores_all_absolute_floors_and_equation_tol():
    numerics = NumericsConfig(forces_tol=None, tol=1e-6, equation_tol=0.1,
                              equation_abs_tol=1.0, equation_abs_tols={"sa": 1000.0})
    peak = jnp.ones(5)
    np.testing.assert_allclose(equation_thresholds(peak, numerics), 1e-6)
    assert not equations_settled(jnp.ones(5) * 1e-5, peak, numerics)
    assert equations_settled(jnp.zeros(5), peak, numerics)


def test_decisions_are_jittable_batched_and_reject_nonfinite_peaks():
    numerics = NumericsConfig()
    current = jnp.array([[0, 0, 0, 0], [0, 0, 0, 2e-4]])
    decide = jax.jit(jax.vmap(lambda row, peak: equations_settled(row, peak, numerics)))
    np.testing.assert_array_equal(decide(current, jnp.ones_like(current)), [True, False])
    assert not equations_settled(jnp.zeros(4), jnp.full(4, jnp.inf), numerics)
    assert not equations_settled(jnp.full(4, jnp.nan), jnp.ones(4), numerics)


def test_record_distinguishes_sa_force_and_insufficient_sample_blockers():
    numerics = NumericsConfig()
    labels = equation_labels(5, 2)
    current = np.array([0, 0, 0, 0, 1e-2])
    record = convergence_record(current, np.ones(5), np.zeros((16, 3)), numerics, names=labels)
    assert not record["settled"] and record["blockers"] == ["sa"]
    assert record["forces_settled"] and record["force_tail_samples"] == 4
    record = convergence_record(np.zeros(5), np.ones(5), np.zeros((3, 3)), numerics, names=labels)
    assert record["equations_settled"] and "forces_insufficient_samples" in record["blockers"]
    assert not record["force_passed"]["cm"]
    coefficients = np.zeros((16, 3))
    coefficients[-1, 2] = 1
    record = convergence_record(np.zeros(5), np.ones(5), coefficients, numerics, names=labels)
    assert record["blockers"] == ["force_cm"]
    record = convergence_record(np.zeros(5), np.ones(5), [], replace(numerics, forces_tol=None), names=labels)
    assert record["settled"] and record["blockers"] == []


def toy_operator(dtype, components=2, sa=False):
    volume = jnp.array([[1, 3], [0, 0]], dtype=dtype)
    centres = jnp.array([[[1, 1 + 1e-9], [0, 0]], [[2, 3], [0, 0]]], dtype=jnp.float64)
    grid = SimpleNamespace(shape=(2, 2), dim=2, components=components,
                           volume=volume, centres=centres.astype(dtype))
    atlas = SimpleNamespace(grid=grid, interior=jnp.array([[True, True], [False, False]]),
                            centres=jnp.pad(centres, ((0, 0), (1, 1), (1, 1))),
                            windows=((slice(0, 1), slice(0, 2)),))
    q = jnp.array([1, 0.5, *([0] * (components - 1)), 1, *([0.01] if sa else [])], dtype=dtype)
    residual = jnp.broadcast_to(jnp.array([[1e20, -2e20], [jnp.nan, jnp.inf]], dtype=dtype),
                               (len(q), 2, 2))
    return ResidualOperator(lambda U: residual, lambda U: volume, atlas, q, 1.4), residual


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("components,sa", [(2, False), (2, True), (3, False), (3, True)])
def test_diagnostics_share_norms_scales_and_ignore_ghosts(dtype, components, sa):
    operator, residual = toy_operator(dtype, components, sa)
    diagnostics = jax.jit(operator.equation_diagnostics)(operator.initial_state())
    scales = np.asarray(operator.equation_scales())
    rates = np.asarray(residual, dtype=np.float64)[:, 0, :] / np.array([1, 3]) / scales[:, None]
    expected = np.sqrt(np.sum(rates**2 * np.array([1, 3]), axis=1) / 4)
    np.testing.assert_allclose(diagnostics["l2_volume"], expected, rtol=1e-14)
    # JIT fusion can round differently from eager evaluation.
    np.testing.assert_allclose(operator.equation_norms(operator.initial_state()),
                               diagnostics["l2_volume"], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(operator.equation_norms_from_residual(residual),
                               diagnostics["l2_volume"], rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(diagnostics["linf"], np.max(np.abs(rates), axis=1), rtol=1e-14)
    np.testing.assert_allclose(diagnostics["balance"], np.asarray(residual, dtype=np.float64)[:, 0, :].sum(axis=1))
    np.testing.assert_allclose(diagnostics["normalized_balance"], np.sum(rates * [1, 3], axis=1) / 4)
    np.testing.assert_array_equal(diagnostics["max_index"], np.tile([0, 0], (len(scales), 1)))
    assert np.all(np.isfinite(diagnostics["l2_volume"]))
    report = diagnostics_report(operator, diagnostics)
    assert report["mass"]["l2_volume"] == float(diagnostics["l2_volume"][0])
    assert report["mass"]["max_location"]["cell_index"] == [0, 0]


def test_location_uses_preserved_fp64_positions_and_zero_ties_are_interior():
    operator, residual = toy_operator(jnp.float32)
    changed = residual.at[:, 0, 0].set(0)
    operator = replace(operator, residual=lambda U: changed)
    diagnostics = operator.equation_diagnostics(operator.initial_state())
    assert float(diagnostics["max_coordinates"][0, 0]) == 1 + 1e-9
    assert float(operator.grid.centres[0, 0, 1]) == 1.0
    zero = replace(operator, residual=lambda U: jnp.zeros_like(residual))
    diagnostics = zero.equation_diagnostics(zero.initial_state())
    np.testing.assert_array_equal(diagnostics["max_index"], np.tile([0, 0], (4, 1)))
    np.testing.assert_array_equal(diagnostics["l2_volume"], 0)
