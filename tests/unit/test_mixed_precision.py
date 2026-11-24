"""Mixed precision is a case option and leaves the nonlinear problem in FP64."""

import numpy as np
import pytest

from jaxrans import load_case
from jaxrans.core.config import NumericsConfig


@pytest.mark.parametrize("name", ["mixed", "mixt"])
def test_mixed_case_keeps_double_fields_and_geometry(name):
    case = load_case("examples/rae2822_sensitivity", (f"case.general.precision={name}",))
    assert case.numerics.mixed_precision
    assert case.numerics.precision == "mixed"
    assert case.numerics.dtype == np.float64
    assert case.grid().grid.volume.dtype == np.float64
    reference = load_case("examples/rae2822_sensitivity")
    assert case.implicit == reference.implicit
    for key in ("tol", "equation_tol", "equation_abs_tol", "forces_tol"):
        assert getattr(case.numerics, key) == getattr(reference.numerics, key)


def test_mixed_alias_is_canonical_and_other_modes_keep_their_meaning():
    assert NumericsConfig(precision="mixt", solver="implicit") == NumericsConfig(precision="mixed", solver="implicit")
    assert NumericsConfig(precision="fp32").dtype == np.float32
    assert NumericsConfig().dtype == np.float64
    assert not NumericsConfig().mixed_precision


def test_robust_defaults_mixed_implicit_and_chain_thresholds():
    """Defaults that converge transonic SA polars past CL max (CHANGELOG)."""
    from jaxrans.core.config import ImplicitConfig
    assert NumericsConfig(solver="implicit").precision == "mixed"
    assert NumericsConfig(solver="explicit").precision == "fp64"
    assert NumericsConfig(solver="implicit", precision="fp64").precision == "fp64"
    assert NumericsConfig().equation_peak == "chain"
    config = ImplicitConfig()
    assert (config.krylov_subspace, config.krylov_restarts, config.krylov_early_exit) == (30, 3, True)
    case = load_case("examples/rae2822_sensitivity")      # implicit, no precision key
    assert case.numerics.mixed_precision


def test_mixed_does_not_silently_run_explicit_in_double():
    with pytest.raises(ValueError, match="requires the implicit"):
        NumericsConfig(precision="mixed", solver="explicit")
