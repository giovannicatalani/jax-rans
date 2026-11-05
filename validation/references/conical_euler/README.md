# Conical Euler numerical checks

These are verification and sensitivity results for the new sectional
operator. They are **not** external reference data, a comparison to FLOWer,
or validation against a finite 3D wing or experiment.

The case is `examples/naca0012_conical_euler`: NACA 0012, M=0.5,
streamwise incidence 2 degrees, leading-edge sweep 30 degrees, an O-grid
with 32 x 32 cells and far field at 50 chords. The taper rate is -dc/ds.
The mesh is deliberately coarse. In particular its absolute drag should
not be interpreted as an accurate inviscid-wing drag prediction.

Measured on CPU in float64; exact coefficients and solver settings are in
[sensitivity.json](sensitivity.json). Every run converged by more than ten
residual orders. The depth is an artificial numerical length in grid units.

| taper | depth | full conicity through / translation from (chords) | CL | CD | CY |
|---|---|---|---|---|---|
| 0 | 0.1 | 1 / 5 | 0.236314519257 | 0.005206840431 | 0.001757217056 |
| 0.2 | 0.2 | 1 / 5 | 0.305256548165 | 0.000932789137 | 0.009809999753 |
| 0.2 | 0.1 | 1 / 5 | 0.305256548165 | 0.000932789137 | 0.009809999753 |
| 0.2 | 0.05 | 1 / 5 | 0.305256548166 | 0.000932789137 | 0.009809999753 |
| 0.2 | 0.1 | 0.5 / 3 | 0.291772691717 | 0.001603593775 | 0.009138370456 |
| 0.2 | 0.1 | 2 / 8 | 0.322292438950 | 0.000081529200 | 0.010664975818 |

The depth study changes CL by less than 1e-12 on this case. The relaxation
study changes CL by -4.42% and +5.58% relative to the 1/5 choice. This is a
material model sensitivity, not a nonlinear convergence error. The chosen
quintic relaxation law and distances require calibration against a 3D wing
before claims of aerodynamic accuracy can be made.

Reproduce the depth study from the repository root:

```bash
JAX_PLATFORMS=cpu python -m validation.conical_euler \
  --depths 0.2 0.1 0.05 \
  --set numerical.implicit.krylov_subspace=12 \
  --set numerical.convergence.block_iters=100 \
  --out /tmp/conical-depth.json
```

Repeat with `--depths 0.1` and overrides
`numerical.sectional.relaxation_start=0.5` and
`numerical.sectional.relaxation_end=3` (or 2 and 8) for the relaxation study.
The baseline uses `--set case.domain.sectional.taper_rate=0`.

The automated checks also compare the conical discretization to independently
integrated 3D Cartesian Euler fluxes for a smooth manufactured field, and
check uniform-flow preservation and the continuous zero-taper limit. Their
implementation is in `tests/verification/test_conical.py`.

The manufactured-field L2 truncation errors on 12, 24 and 48 cells per
section direction were 6.95873e-5, 1.82468e-5 and 4.66150e-6, giving observed
orders 1.93 and 1.97. Spanwise depth is refined with the section in this test;
the flux reference uses 3x3 Gauss integration of the analytic Cartesian field.
For taper rates 1e-3, 1e-4 and 1e-5, the relative residual differences from
zero taper were 2.04753e-4, 2.04673e-5 and 2.04665e-6: a continuous linear
approach to the existing 2.5D operator.
