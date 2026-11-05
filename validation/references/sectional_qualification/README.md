# Stage D, points 1 and 2: numerical qualification

This directory records independently converged zero-taper limit comparisons
and finite-taper mesh sensitivity. These are internal numerical checks, not
external reference data or validation against a finite wing.

## Findings

All 24 solves converged by at least 10.06 residual orders. All eight limit
comparisons pass: the largest absolute coefficient difference is 1.84e-9,
the largest Cp difference 1.46e-8, and the largest normalized primitive
field difference 1.27e-7 (including SA). This validates point 1 for the
cases and tolerances in the campaign.

Point 2 reveals remaining mesh sensitivity. From 128x96 to 256x192, CL
changes by 2.54%, CD by 3.70% and friction drag by .237%, relative to the
finer result. The observed orders on the three finest levels are .612 for
CL and 1.834 for CD; friction-drag differences do not shrink, so no
Richardson extrapolation is given for that quantity. The fields and forces
have **not** established mesh independence. In particular, the 64x48 example
is not an aerodynamic reference solution.

At the 64x48 anchor, halving the first source spacing changes CL by .072%
and halving it again by .073%. Moving the minimum outer distance from 45.14
to 102.93 chords changes CL by .448%; moving it to 225.39 changes CL by
another .197%. Both outer extensions preserve every node of the original
inner mesh. Global refinement is the dominant measured sensitivity here.
The wall/domain studies were performed on the 64x48 anchor; their interaction
with refinement of the complete mesh has not been quantified.

Further global refinement is needed before claiming a tight CL/CD accuracy
target. The extrapolations are indicative fits, not validated continuum
reference values. No calibration or 3D-wing comparison is implied.

See [summary.md](summary.md) for measured comparisons and [summary.json](summary.json)
for the complete numerical records, tolerances, job settings, code hashes,
devices, iteration counts, residual reductions and final residuals per equation.
The [refinement figure](mesh_refinement.png) is also available as
[PDF](mesh_refinement.pdf).

The [method and reproduction guide](../../../docs/sectional-qualification.md)
explains the source mesh, nested subsampling, separate first-wall-spacing
study and far-field extensions that preserve the entire common inner mesh.

From the repository root:

```bash
JAX_PLATFORMS=cuda python -u -m validation.sectional_qualification \
  --study all --out results/sectional_d_qualification
python -m validation.sectional_report results/sectional_d_qualification --plot
```

On the WSL2 system used for these measurements the additional setting
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` avoids the documented JAX
Triton compilation issue. Solves use float64 on CUDA device 0. All sixteen
limit solves start independently from freestream; the eight mesh-sensitivity
solves do likewise. No force-based early stopping is used.

The zero-taper comparisons cover Euler, laminar, SA and SA-QCR at 0 and 30
degrees of sweep. The mesh studies use conical SA at M=.5, incidence 2
degrees, Re=1e6, leading-edge sweep 30 degrees, taper rate .2, numerical
depth .1 and relaxation window 1/5 chords. They do not establish accuracy
outside that case or calibrate the relaxation window.

Full per-run inputs, physical solved nodes, primitive fields, wall arrays
and residual histories are produced under `results/sectional_d_qualification`.
Those generated fields are not duplicated in this reference directory.
