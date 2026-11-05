"""Input, sequencing and runnable integration of the zero-taper sectional mode."""

from dataclasses import replace
import json

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrans import CaseError, run
from jaxrans.case import apply_override, parse_case
from jaxrans.mesh.coarsen import prolong
from jaxrans.mesh.sectional import SectionalAtlas
from jaxrans.numerics.residual import make_residual


def document(**sectional):
    return {"general": {"case_name": "sectional"},
            "domain": {"grid": "o_grid", "airfoil": "naca0012", "cells_airfoil": 32,
                       "cells_normal": 16, "wall_spacing": 1e-3, "far_field": 20.0,
                       "sectional": {"model": "conical", "leading_edge_sweep_deg": 30.0,
                                     "taper_rate": 0.0, **sectional}},
            "freestream": {"mach": 0.5, "alpha_deg": 2.0}}


@pytest.mark.parametrize("values, message", [
    ({"taper_rate": float("nan")}, "finite"),
    ({"leading_edge_sweep_deg": 90.0}, "leading_edge_sweep_deg"),
    ({"leading_edge_sweep_deg": float("inf")}, "finite"),
    ({"leading_edge_sweep_deg": True}, "must be a number"),
    ({"model": "unknown"}, "model"),
    ({"surprise": 1.0}, "unknown key"),
])
def test_invalid_or_unimplemented_geometry_is_refused(values, message):
    with pytest.raises(CaseError, match=message):
        parse_case(document(**values), {})


def test_the_legacy_and_sectional_sweep_inputs_cannot_conflict():
    doc = document()
    doc["freestream"]["sweep_deg"] = 30.0
    with pytest.raises(CaseError, match="mutually exclusive"):
        parse_case(doc, {})


@pytest.mark.parametrize("depth", [0.0, -1.0, float("inf")])
def test_invalid_numerical_depth_is_refused(depth):
    with pytest.raises(ValueError, match="depth"):
        parse_case(document(), {"sectional": {"depth": depth}})


def test_a_sectional_case_builds_one_independent_plane_and_normal_section():
    case = parse_case(document(), {"sectional": {"depth": 0.3}})
    atlas = case.grid()
    assert isinstance(atlas, SectionalAtlas) and atlas.dim == 3
    assert atlas.cells == atlas.reference.cells == 32 * 16
    assert atlas.depth == 0.3 and atlas.spec == case.grid_spec.sectional
    np.testing.assert_allclose(np.ptp(np.asarray(atlas.mesh[0].nodes)[2]), 0.3)
    np.testing.assert_allclose(np.ptp(np.asarray(atlas.mesh[0].nodes)[0, :, 0, 0]),
                               np.cos(np.deg2rad(30.0)), rtol=1e-3)


def test_sequencing_refines_the_section_without_refining_the_artificial_direction():
    case = parse_case(document(), {})
    coarse, fine = case.grid(coarsen=1), case.grid()
    assert fine.cells == 4 * coarse.cells
    assert coarse.mesh[0].shape[-1] == fine.mesh[0].shape[-1] == 1
    U = jnp.asarray(np.random.default_rng(1).normal(size=(5,) + coarse.shape))
    start = jnp.full((5,) + fine.shape, 7.0)
    projected = prolong(U, coarse, fine, start)
    child = np.asarray(fine.unpack(projected)[0])
    parent = np.asarray(coarse.unpack(U)[0])
    for di in (0, 1):
        for dj in (0, 1):
            np.testing.assert_array_equal(child[:, di::2, dj::2], parent)


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
@pytest.mark.parametrize("physics", ["euler", "sa"])
def test_sectional_case_runs_and_writes_sectional_outputs(tmp_path, solver, physics):
    doc = document()
    if physics == "sa":
        doc["freestream"]["reynolds"] = 1e6
        doc["turbulence"] = {"model": "sa_noft2"}
    case = parse_case(doc, {
        "time_integration": {"solver": solver},
        "convergence": {"max_iters": 2, "block_iters": 1, "forces_tol": None},
        "implicit": {"krylov_subspace": 3, "krylov_restarts": 1},
    }, tmp_path)
    case = replace(case, numerics=replace(case.numerics, limiter="van_albada"))
    result = run(case)
    assert result.history.n_iters == 2 and np.isfinite(result.forces.cl)
    assert np.isfinite(np.asarray(result.U)).all()
    assert "x_streamwise" in result.surface()
    out = result.write()
    assert (out / "sectional_surface.dat").is_file()
    assert (out / "sectional_block0.vts").is_file()
    metadata = json.loads((out / "sectional_metadata.json").read_text())
    assert metadata["extra"]["sectional"]["axes"] == ["normal_chord", "up", "sweep"]
    assert metadata["extra"]["sectional"]["taper_rate"] == 0.0


def test_flow_sweep_and_sectional_geometry_must_match():
    case = parse_case(document(), {})
    with pytest.raises(ValueError, match="flow frame"):
        make_residual(case.grid(), replace(case.flow, sweep_deg=0.0), case.numerics)


def test_nested_sectional_overrides_are_recorded():
    docs = {"case": document(), "numerical": {}}
    apply_override(docs, "case.domain.sectional.leading_edge_sweep_deg=-20.0")
    case = parse_case(docs["case"], docs["numerical"])
    assert case.grid_spec.sectional.leading_edge_sweep_deg == -20.0


def test_numerical_sectional_options_require_the_sectional_model():
    doc = document()
    del doc["domain"]["sectional"]
    with pytest.raises(CaseError, match="requires 'domain.sectional'"):
        parse_case(doc, {"sectional": {"depth": 0.3}})


def test_spatial_decomposition_cannot_drop_the_sectional_closure():
    from jaxrans.numerics.slabs import SlabAtlas

    atlas = parse_case(document(), {}).grid()
    with pytest.raises(ValueError, match="sectional atlas"):
        SlabAtlas.of(atlas, [object(), object()])
