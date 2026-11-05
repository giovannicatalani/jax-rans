# Conical viscous numerical checks

These results verify the sectional discretization and its numerical depth
sensitivity. They are not external aerodynamic reference data or validation
against FLOWer, a finite wing or an experiment. The frozen primitive closure
at fixed reference viscosity and the uncalibrated relaxation window are
described in [sectional calculations](../../../docs/sectional.md).

The example is `examples/naca0012_conical_sa`: NACA 0012, M=0.5, streamwise
incidence 2 degrees, Re=1e6, SA-noft2, constant molecular viscosity,
leading-edge sweep 30 degrees, taper rate 0.2. It uses a generated 64 x 48
O-grid, wall spacing 2e-5, far field 50 chords, full conicity through one
chord and translation from five chords. This is a coarse numerical check;
the aerodynamic coefficients have not been established by mesh refinement.

Reproduce from the repository root (the module retains its stage B name):

```bash
JAX_PLATFORMS=cpu python -m validation.conical_euler \
  examples/naca0012_conical_sa --depths .1 .05 --progress \
  --out /tmp/conical-sa-depth.json
```

CPU, float64. Detailed coefficients, stopping reasons and residual reductions
are in [depth.json](depth.json). Each run starts from uniform freestream.
The stopping tolerance is a ten-order residual decrease; force-based early
termination is disabled.

| depth | iterations | residual orders dropped | CL | CD | friction CD |
|---|---|---|---|---|---|
| .1 | 640 | 10.25 | .265435184727 | .007773835956 | .008554300321 |
| .05 | 1100 | 10.37 | .265434964986 | .007773968634 | .008554405120 |

Halving the depth changes CL by 8.3e-7 relative, CD by 1.7e-5 relative and
friction drag by 1.2e-5 relative. The extra iterations at smaller depth are
consistent with the retained conservative 3D pseudo-time estimate; iteration
histories are not expected to be identical.

The automated checks in `tests/verification/test_conical_viscous.py` compare
laminar, SA and SA-QCR residuals, including the turbulence equation, to the
middle of a separately built five-layer 3D grid on the same perturbed section
state. They also verify no-slip adiabatic walls, exactly zero wall eddy
viscosity, differentiated closure, multiblock equivalence and the continuous
zero-taper limit of residuals, forces and friction. A separate refinement of
the numerical depth (.2, .1, .05, .025) gives second-order residual convergence
and wall friction independent of depth to round-off on the same state.

The laminar manufactured test in `tests/verification/test_conical.py` compares
the discrete residual against independent analytic Cartesian Navier-Stokes
fluxes integrated on all six faces with 3x3 Gauss quadrature. Re=10, taper
rate .3, unrelaxed conicity, section refinements 12, 24, 48, depth=1/n:

| section resolution | L2 truncation error |
|---|---|
| 12 | 6.98710436e-5 |
| 24 | 1.83139523e-5 |
| 48 | 4.67773242e-6 |

The observed orders are 1.932 and 1.969. This checks the geometric closure
and viscous discretization, rather than the validity of frozen conical
profiles at changing local Reynolds number.
