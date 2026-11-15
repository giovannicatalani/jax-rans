"""Equation convergence decisions and host reports, independent of solvers.

Arrays returned by the decision helpers can be jitted or vmapped. Reports
convert those arrays to ordinary Python values only at the host boundary.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def equation_labels(nvar: int, components: int) -> tuple[str, ...]:
    """Conservative state order, distinguishing 2D SA from 3D Euler."""
    if components not in (2, 3) or nvar not in (components + 2, components + 3):
        raise ValueError("expected flow equations with optional SA and 2 or 3 velocity components")
    labels = ("mass", *(f"momentum_{axis}" for axis in "xyz"[:components]), "energy")
    return labels + (("sa",) if nvar == components + 3 else ())


def equation_names(components: int, n_equations: int) -> tuple[str, ...]:
    """Alias with component count first for residual consumers."""
    return equation_labels(n_equations, components)


def _labels(nvar, labels, names, *, required=False):
    if labels is not None and names is not None:
        raise ValueError("pass labels or names, not both")
    result = labels if labels is not None else names
    if result is None:
        if nvar == 5:
            if required:
                raise ValueError("five equations require explicit labels/names for per-equation floors")
            return tuple(f"equation_{i}" for i in range(nvar))
        result = equation_labels(nvar, 2 if nvar == 4 else 3)
    result = tuple(result)
    if len(result) != nvar or len(set(result)) != nvar:
        raise ValueError("equation labels must be unique and match the equation count")
    return result


def equation_thresholds(peak, numerics, *, labels=None, names=None):
    """Per-equation ``max(relative * peak, absolute_floor)``.

    Strict mode (``forces_tol=None``) uses ``tol * peak`` exclusively.
    The last axis contains equations; leading axes may contain batch cases.
    """
    peak = jnp.asarray(peak)
    if numerics.forces_tol is None:
        return numerics.tol * peak
    overrides = dict(numerics.equation_abs_tols)
    labels = _labels(peak.shape[-1], labels, names, required=bool(overrides))
    floors = jnp.asarray([overrides.get(name, numerics.equation_abs_tol) for name in labels], dtype=peak.dtype)
    return jnp.maximum(floors, numerics.equation_tol * peak)


def equations_settled(current, peak, numerics, *, labels=None, names=None):
    """Whether every equation is finite, nonnegative and below its threshold."""
    current, peak = jnp.asarray(current), jnp.asarray(peak)
    threshold = equation_thresholds(peak, numerics, labels=labels, names=names)
    valid = jnp.isfinite(current) & jnp.isfinite(peak) & (current >= 0) & (peak >= 0)
    return jnp.all(valid & (current <= threshold), axis=-1)


def cl_spread(history) -> float:
    """Range over the final quarter; at least four tail samples are needed."""
    history = np.asarray(history)
    tail = history[3 * len(history) // 4:]
    return float(np.ptp(tail)) if tail.size >= 4 else float("nan")


def convergence_record(current, peak, coefficients, numerics, *, labels=None, names=None) -> dict:
    """Host decision and evidence for one case.

    ``coefficients`` has shape ``(blocks, 3)`` in CL/CD/CM order. Empty
    histories cannot satisfy a force criterion. Equation names are optional
    except for ambiguous five-variable states with absolute overrides.
    """
    current, peak = np.asarray(current), np.asarray(peak)
    labels = _labels(current.shape[-1], labels, names,
                     required=bool(numerics.equation_abs_tols) and numerics.forces_tol is not None)
    threshold = np.asarray(equation_thresholds(peak, numerics, labels=labels))
    passed = np.isfinite(current) & np.isfinite(peak) & (current >= 0) & (peak >= 0) & (current <= threshold)
    coefficients = np.asarray(coefficients)
    if coefficients.size == 0:
        coefficients = np.empty((0, 3))
    if coefficients.ndim != 2 or coefficients.shape[1] != 3:
        raise ValueError("coefficients must have shape (blocks, 3) in CL/CD/CM order")
    spreads = [cl_spread(coefficients[:, i]) for i in range(3)]
    force_names = ("cl", "cd", "cm")
    force_passed = {name: bool(numerics.forces_tol is None or
                               (np.isfinite(spread) and spread < numerics.forces_tol))
                    for name, spread in zip(force_names, spreads)}
    forces_settled = all(force_passed.values())
    equations_ok = bool(np.all(passed))
    blocking_equations = [name for name, ok in zip(labels, passed) if not ok]
    tail_samples = len(coefficients) - 3 * len(coefficients) // 4
    blockers = list(blocking_equations)
    if numerics.forces_tol is not None:
        if tail_samples < 4:
            blockers.append("forces_insufficient_samples")
        blockers.extend(f"force_{name}" for name, ok in force_passed.items() if not ok)
    return {
        "settled": bool(equations_ok and forces_settled),
        "equations_settled": equations_ok,
        "forces_settled": bool(forces_settled),
        "current": current.tolist(),
        "peak": peak.tolist(),
        "thresholds": threshold.tolist(),
        "equation_passed": passed.tolist(),
        "names": list(labels),
        "force_spreads": dict(zip(force_names, spreads)),
        "force_passed": force_passed,
        "force_tail_samples": tail_samples,
        "forces_tol": numerics.forces_tol,
        "blocking_equations": blocking_equations,
        "blockers": blockers,
    }


def diagnostics_report(operator, diagnostics) -> dict:
    """Name-keyed JSON-ready equation diagnostics, with block/cell locations."""
    arrays = {key: np.asarray(value) for key, value in diagnostics.items()}
    labels = equation_labels(len(arrays["scales"]), operator.components)
    report = {}
    for row, name in enumerate(labels):
        index = arrays["max_index"][row]
        location = {"canvas_index": index.tolist(), "coordinates": arrays["max_coordinates"][row].tolist()}
        for block, window in enumerate(operator.atlas.windows):
            if all(s.start <= i < s.stop for i, s in zip(index, window)):
                location.update(block=block, cell_index=[int(i - s.start) for i, s in zip(index, window)])
                break
        report[name] = {
            "scale": float(arrays["scales"][row]),
            "l2_volume": float(arrays["l2_volume"][row]),
            "linf": float(arrays["linf"][row]),
            "balance": float(arrays["balance"][row]),
            "normalized_balance": float(arrays["normalized_balance"][row]),
            "max_location": location,
        }
    return report
