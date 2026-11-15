"""Precision policies must survive the sectional geometry integration."""
import jax.numpy as jnp
import numpy as np
import pytest
from jaxrans.case import parse_case
from jaxrans.numerics.residual import make_residual
from tests.unit.test_sectional import document


@pytest.mark.parametrize('precision', ['fp32', 'mixed', 'fp64'])
@pytest.mark.parametrize('taper', [0., .03])
def test_sectional_operator_preserves_precision_and_geometry(precision, taper):
    doc = document(taper_rate=taper)
    doc['general']['precision'] = precision
    case = parse_case(doc, {'time_integration': {'solver': 'implicit'}})
    atlas = case.grid()
    op = make_residual(atlas, case.flow, case.numerics)
    expected = jnp.float32 if precision == 'fp32' else jnp.float64
    u = op.initial_state()
    residual = op.residual(u)
    assert u.dtype == residual.dtype == expected
    assert atlas.mesh[0].nodes.dtype == np.float64
    assert atlas.spec.taper_rate == taper
    assert np.isfinite(residual).all()
    assert np.all(np.asarray(atlas.grid.volume)[atlas.interior] > 0)


@pytest.mark.parametrize('taper', [0., .03])
def test_mixed_sectional_sa_runs_and_preserves_output_policy(tmp_path, taper):
    import json
    from jaxrans import run

    doc = document(taper_rate=taper)
    doc['general']['precision'] = 'mixed'
    doc['freestream']['reynolds'] = 1e6
    doc['turbulence'] = {'model': 'sa_noft2'}
    case = parse_case(doc, {
        'time_integration': {'solver': 'implicit'},
        'convergence': {'max_iters': 2, 'block_iters': 1, 'forces_tol': None},
        'implicit': {'krylov_subspace': 3, 'krylov_restarts': 1},
    }, tmp_path)
    result = run(case)
    assert result.history.n_iters == 2
    assert result.U.dtype == jnp.float64
    assert np.isfinite(result.U).all()
    assert np.isfinite(result.forces.cl)
    directory = result.write(tmp_path / f'taper_{taper}')
    metadata = json.loads((directory / 'sectional_metadata.json').read_text())
    assert metadata['extra']['sectional']['taper_rate'] == taper
    assert metadata['extra']['precision_policy']['state'] == 'float64'
    assert metadata['extra']['precision_policy']['krylov_basis'] == 'float32'
    assert metadata['extra']['precision_policy']['preconditioner'] == 'float32'
