# Changelog

## 0.2.0

- **Renamed to JAX-RANS**: distribution `jax-rans`, import `jaxrans`, command
  `jaxrans` (formerly `jax-aero`, `jaxaero`; nothing had been released under
  the old name, so there is no alias).
- **Derivatives of the forces** (`jaxrans.sensitivity(result, outputs,
  parameters)`): dCL, dCD, dCM and the friction drag with respect to the
  angle of attack and the Mach number (at fixed Reynolds number), exact for
  the discrete solution, one linear solve per parameter on the exact Jacobian
  of the converged residual, iterated with the implicit solver's own Newton
  step (docs/gradients.md). Verified against central finite differences to
  4e-6 (laminar NACA 0012) and 6e-7 (SA Clark Y) in dCL/dalpha;
  `validation/check_sensitivity.py` repeats the check on any case. The force
  coefficients are one JAX function now (`force_coefficients`),
  `integrate_forces` its float conversion: the same values to 1e-15.
  New limiter `van_albada_smooth` (van Albada's original form, no cut at
  extrema): differentiable, for derivatives of a flow with a shock.
- **Quickstart notebook** (`notebooks/quickstart.ipynb`, with an Open in
  Colab badge): a transonic NACA 0012 solved on a generated grid, its Mach
  field and Cp, and its derivatives checked against finite differences. CI
  executes it on every push. Committed without outputs.
- **Multi-block 3D grids**: any number of point-matched blocks, from Plot3D or
  CGNS (HDF5; 3D zones read whole, with their families, BC types and 1-to-1
  interfaces). Interfaces are found from coincident faces in any relative
  orientation of the blocks, partial faces included; three or five blocks may
  meet along an edge. The line preconditioner runs normal to each block's
  wall, whatever its index; the ghost fill repeats its three stages where an
  interface permutes index directions (docs/dev/notes.md). Single-block 3D
  results unchanged bitwise. Verified by splitting the ONERA M6 grid into
  re-indexed blocks (the same residual to 4e-15) and on a three-block prism.
  New validation case `adodg_crm_wing` (the NASA CRM wing, 26 blocks);
  `scripts/adf2hdf.py` converts ADF CGNS files.
- **Generated multi-element grids** (`domain.grid = "multi_element"`): a slat,
  main element and flap meshed from their coordinates, optionally re-placed by
  deflection, gap and overlap (docs/multi-element.md). New example
  `main_flap_sa`; validation case `gen_30p30n_a5p5` (the 30P30N on generated
  grids); `scripts/plot_blocks.py` draws a grid's blocks.
- **One case over several GPUs** (`jaxrans CASE --devices N`, `run(case,
  devices=...)`): the grid split in slabs along its first index. The
  residual's stencils exchange two rows with each neighbour per evaluation
  (`shard_map`), the rest of the step is partitioned by XLA. ONERA M6 1.08M:
  1.8x faster on four V100s than on one, a quarter of the memory each; the
  8.65M grid fits on four 32 GB GPUs. All of it in `numerics/slabs.py` and
  behind `--devices`: a one-GPU run executes the code it did before. Works on
  multi-block grids too: the NASA CRM wing on 3.6M cells (26 blocks) runs on
  two A100s (24 GB each; 35 GB of temporaries alone on one), 2.2 s a step.
- **Faster implicit solver on GPUs.** The line preconditioner solves its
  block-tridiagonal chains by cyclic reduction instead of a sequential Thomas
  sweep: on an A100 a Newton step takes 21.5 ms instead of 136 on RAE 2822,
  127 instead of 186 on the ONERA M6 135k grid. Same solutions to round-off.
- **Runs stop when their forces have settled** (`convergence.forces_tol`,
  default 1e-4; `null` for the old behaviour): RAE 2822 in 940 iterations
  instead of 6000 (57 s instead of 841 on an A100), with the same forces. New
  exit reasons `settled` and `stalled` (no step accepted in a whole block).
  Polars are unchanged (every angle to `tol` or `max_iters`).
- **Grid sequencing** (`convergence.grid_sequencing`): solve first on the case's
  grid coarsened by taking every other node, then start the finer grid from
  it. On the ONERA M6 1.08M grid, 2.8x fewer fine-grid iterations to settled
  forces. Single runs only; the grid must coarsen evenly.

- `domain.trailing_edge = "close"` closes an open trailing edge so that a
  C-grid can be built round a downloaded coordinate file.
- Airfoils are scaled to unit chord with the leading edge at the origin, so
  coordinates in other units give correct coefficients.
- `jaxrans CASE --check` builds the grid, prints its patches, conditions and
  bodies, writes `grid.png` and stops.
- New examples: `clarky_sa` (a UIUC coordinate file as downloaded) and
  `30p30n_sa` (a multi-block grid from file).
- Documentation: walkthroughs for your own airfoil and for a multi-element
  airfoil.
- **3D**: one code for 2D and 3D -- the dimension is the
  grid's. Single-block 3D grids from Plot3D files (O seams and poles found
  from the coordinates, patches named by box rules with z bounds), Euler,
  laminar and SA in 3D, 3D forces (`domain.reference_area`,
  `domain.moment_centre`), ParaView output (`.vts`); `muscl_kappa` (the
  unlimited kappa = 1/3 reconstruction, as CFL3D runs the M6). Validated on the M6,
  Euler and SA (CL +0.19%, CD -1.74% vs CFL3D on the same 1.08M grid).
  Example `onera_m6_sa` (its grid in the example directory) and the
  walkthrough [docs/wing-3d.md](docs/wing-3d.md).
  2D results unchanged (to 5e-15). Limits: single-block 3D grids from Plot3D;
  `jaxrans.plot` is 2D only (3D wings: `scripts/plot_wing.py` or ParaView).
- Internal API (Python users of library internals; JSON cases unaffected):
  Riemann fluxes take an area vector `S (d, ...)`; state conversions take
  `dim`; `Grid` stores `nodes`, `centres`, `volume`, `faces`.

## 0.1.0 -- first release

2D steady compressible flow on structured grids, run from two JSON files.

- **Physics**: Euler, laminar Navier-Stokes (constant or Sutherland
  viscosity), Spalart-Allmaras RANS (SA-noft2, SA-neg).
- **Numerics**: cell-centred finite volume; MUSCL with minmod, van Albada or
  no limiter; HLLC and Roe fluxes; diamond face gradients. Explicit
  multistage Runge-Kutta, or pseudo-transient Newton-Krylov with
  right-preconditioned GMRES and a line-implicit preconditioner.
- **Grids**: O- and C-grids generated round any airfoil (NACA 4-digit or a
  coordinate file: Selig, Lednicer, plain); any structured grid from Plot3D
  (every variant) or CGNS/HDF5, single- or multi-block, interfaces found from
  the coordinates; a quality report before every run.
- **Boundary conditions per named patch**: slip and no-slip walls, symmetry,
  characteristic far field (optionally with the point vortex), total-condition
  inflow, static-pressure outflow.
- **Polars**: every angle in one batched run, split over GPUs.
- **Case files**: strict schema; unknown keys, wrong types and unimplemented
  options are errors naming the key; the reference page is generated from the
  schema.
- **Validation** against NASA TMR (NACA 0012, flat plate), NPARC/AGARD
  AR-138 (RAE 2822) and JAXA (30P30N high-lift) in `validation/`.
