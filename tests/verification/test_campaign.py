"""Real CFD steps: batched geometry and warm starts match standalone runs."""

import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from jaxrans import load_case, run
from jaxrans.campaign import PolarRequest, run_campaign
from jaxrans.polar import at_alpha

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ("case.domain.cells_airfoil=32", "case.domain.cells_normal=16",
            "case.freestream.mach=0.3", "numerical.convergence.max_iters=200",
            "numerical.convergence.block_iters=10", "numerical.convergence.tol=0.8",
            "numerical.convergence.forces_tol=null")


@pytest.fixture(scope="module")
def requests():
    return [PolarRequest(name, load_case(ROOT / "examples/naca0012_euler",
                         SETTINGS + (f"case.domain.airfoil={name}",)), [0, 1])
            for name in ("naca0012", "naca2412")]


@pytest.fixture(scope="module")
def campaign(requests):
    return run_campaign(requests)


def test_real_batch_matches_standalone_continuation(requests, campaign):
    assert campaign.successful and len(campaign.batches) == 1
    for request in requests:
        grid, previous = request.case.grid(), None
        polar = campaign.polars[request.name]
        assert len(polar.results) == 2
        for alpha, batched in zip(request.alphas, polar.results):
            single = run(at_alpha(request.case, alpha), grid, initial=previous)
            assert single.history.n_iters == batched.history.n_iters
            np.testing.assert_allclose(batched.U, single.U, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(batched.cl_history, single.cl_history, rtol=1e-9, atol=1e-12)
            assert batched.forces.cl == pytest.approx(single.forces.cl, rel=1e-9, abs=1e-12)
            previous = single.U


def test_profile_batch_on_two_devices(requests, campaign, tmp_path):
    """Three entries force padding; the inactive copy must never count as a profile."""
    script = textwrap.dedent(f"""
        import jax
        import numpy as np
        from jaxrans import load_case
        from jaxrans.campaign import PolarRequest, run_campaign
        assert len(jax.devices()) == 2
        requests = [PolarRequest(name, load_case({str(ROOT / 'examples/naca0012_euler')!r},
                      {SETTINGS!r} + (f'case.domain.airfoil={{name}}',)), [0, 1])
                    for name in ('naca0012', 'naca2412')]
        requests.append(PolarRequest('duplicate', requests[0].case, [0, 1]))
        campaign = run_campaign(requests, devices=2)
        assert campaign.successful and len(campaign.batches) == 1
        for name, polar in campaign.polars.items():
            assert polar.timing.devices == 2
            np.save(name + '.npy', np.stack([np.asarray(r.U) for r in polar.results]))
    """)
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    for request in requests:
        expected = np.stack([np.asarray(r.U) for r in campaign.polars[request.name].results])
        np.testing.assert_allclose(np.load(tmp_path / f"{request.name}.npy"), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("solver", ["explicit", "implicit"])
def test_rans_batched_geometry_matches_single_steps(solver):
    settings = SETTINGS + (f"numerical.time_integration.solver={solver}",
                           "numerical.time_integration.cfl=0.01",
                           "numerical.convergence.max_iters=4", "numerical.convergence.block_iters=2",
                           "numerical.convergence.tol=1e-12")
    requests = []
    for name in ("naca0012", "naca2412"):
        case = load_case(ROOT / "examples/naca0012_euler", settings + (f"case.domain.airfoil={name}",))
        case = replace(case, flow=replace(case.flow, reynolds=5000., turbulence="sa_noft2"),
                       conditions={**case.conditions, "wall": "no_slip_wall"})
        requests.append(PolarRequest(name, case, [0, 1]))
    campaign = run_campaign(requests)
    assert len(campaign.batches) == 1
    for request in requests:
        batched = campaign.polars[request.name].results[0]
        single = run(at_alpha(replace(request.case, sequencing=0), 0))
        assert np.isfinite(batched.U).all() and np.isfinite(single.U).all()
        np.testing.assert_allclose(batched.U, single.U, rtol=1e-9, atol=1e-11)
        assert batched.forces.cd == pytest.approx(single.forces.cd, rel=1e-9, abs=1e-12)
