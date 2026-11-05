# Changelog

## Unreleased

- **Sectional qualification, stage D points 1–2**: reproducible independent
  converged 2D/2.5D limit comparisons for Euler, laminar, SA and SA-QCR;
  nested conical SA mesh refinement and separate wall-spacing/far-field
  studies. Comparisons cover primitive fields, Cp, friction and forces,
  with strict convergence and provenance checks. Per-run arrays and logs,
  JSON/Markdown summaries and optional refinement plots are provided by
  `validation.sectional_qualification` and `validation.sectional_report`.
- **Conical viscous sections, stage C**: finite taper supports laminar
  Navier-Stokes, SA and SA-QCR with the full 3D kernels. Physical ghost centres
  use conical wall reflections and partner-block centres; SA uses distance to
  the radially extended physical wall. Viscous section loads transport wall
  traction to the streamwise cut. The closure copies Cartesian primitives,
  including the SA variable, at fixed reference viscosity. Verified against
  a five-layer 3D block, a laminar manufactured flux integral and the
  continuous 2.5D limit. Example `naca0012_conical_sa`; see `docs/sectional.md`
  for the local-similarity assumption and uncalibrated relaxation window.
- **Conical Euler sections, stage B**: finite `domain.sectional.taper_rate`,
  a mean-sweep frame, geometric intersection of streamwise profiles and a C2
  wall-distance relaxation to periodic translation. All face metrics and
  wall normals come from the conical 3D cells; section pressure loads include
  the conical side component. Verified by uniform flow, the zero-taper limit,
  depth refinement and a second-order manufactured Cartesian Euler flux
  integral check. New example
  `naca0012_conical_euler` and `validation.conical_euler` depth study.
- **Sectional 3D stencil, stage A** (`domain.sectional`): one independent
  spanwise cell with translated ghost geometry, using the full 3D flux and
  gradient operators. Zero taper reproduces 2.5D (and 2D at zero sweep),
  including laminar, SA and SA-QCR, both far fields and sectional outputs.
  Nonzero taper is explicitly rejected until the conical geometry and
  relaxation are implemented. Example `naca0012_sectional_euler` and
  `docs/sectional.md`; nested case overrides are supported.
- **Blunt trailing edges on a C-grid** (contributed by Xavier Bertrand): a C-grid
  round a blunt section now keeps its base, with a block off the base and one in
  the wake (example `naca0012_blunt_sa`). `.hug` coordinate files give the two
  corners explicitly, and `domain.base_blocks` splits a multi-element grid's
  blunt layers at their bases. Sharp sections and closed trailing edges unchanged.
- **Swept wing sections (2.5D)** (`freestream.sweep_deg`, `domain.section`):
  an infinite swept wing on any 2D grid, the spanwise velocity a third
  component with no derivative along the span (docs/swept.md). The section is
  given streamwise (meshed normal to the sweep line, `x` scaled by
  `cos(sweep)`) or normal; forces in the wing frame, per unit span, with the
  spanwise force `CY`; the surface file adds `x_streamwise` and `cf_z`. Every
  model (Euler, laminar, SA, SA-QCR), both far fields, the implicit and
  explicit paths, polars, derivatives in alpha and Mach, several devices.
  Verified against the 3D operator on a periodic extrusion (to 1e-11, every
  model), and inviscid against the 2D solution at the in-plane Mach number
  (to 4e-13, converged, transonic). Unswept 2D and 3D results unchanged
  bitwise. New example `rae2822_swept_sa`.
- Validation reduced to six main cases (flat plate, NACA 0012 at 10 deg, RAE 2822
  case 6, the 30P30N on JAXA's and on the generated grid, ONERA M6), with one
  page of results, `validation/README.md`. The gallery and `validation/report.py`
  are gone; `compare_surface.py --plot` draws Cp against the references.
- The Karman-Trefftz grid generator moved to `tests/`, its only user.
- `docs/multi-element.md` explains the multi-block template and what it takes.

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
