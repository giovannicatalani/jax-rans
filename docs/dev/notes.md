# Design notes

Why some things are done the way they are, with the measurements behind
each choice, grouped by module.

## solvers/implicit.py

- **Defect correction is the default Jacobian.** On 112x32 over 400 steps, a
  limited second-order residual dropped 0.06 orders with the exact Jacobian
  and 2.8 with the first-order one; an unlimited residual gave 4.9 orders
  exact. The exact Jacobian of a limited scheme carries the limiter's kinks.
- **The CFL ramp keys off the linear solve, not the residual history.** A
  stagnated GMRES returns a null update, which leaves the residual unchanged
  and looks like success to a residual-based test; a freestream start makes
  the residual rise through the transient, which looks like failure to
  textbook SER (Mulder and van Leer, J. Comput. Phys. 59 (1985)). One extra
  matvec, about 1% of a step, measures the linear reduction instead.
- **`max_relative_update`.** Rejection alone only stops steps that leave the
  physical domain. On TMR NACA 0012 at 15 deg, 449x129, accepted steps built
  an odd-even checkerboard in nu~ and rho within 600 steps, after which every
  step was rejected; bounding the relative change (0.2) cured it.

## solvers/driver.py

- **Force-based stopping is the default** (`convergence.forces_tol` 1e-4,
  the gallery's settledness bar, applied to CL, CD and CM). On the M6 1.08M
  level from freestream it stops at 2540 steps with the forces within 3e-6 of
  the 3000-step values; 1e-5 did not stop within 3000 (the last-quarter
  window still holds the early drift). RANS residuals stall at
  round-off 10-11 orders down, and some cases at 2-3 (30P30N L1, NACA 0012
  at 0 deg) with the forces settled to six digits long before. On RAE 2822
  the run stops at 940 of 6000 steps with the 6000-step forces to every
  printed digit; on the M6 135k level at 1600, the same forces to six digits
  whatever the Krylov settings.
- **A frozen run stops as `stalled`, never `settled`.** When every step is
  rejected the state does not move, so the forces are flat and would pass
  the settledness test (a mis-started 1.08M M6 run stopped as settled after
  260 steps that way). A block whose residual is bitwise constant is no
  progress, whatever the equations.

- **Convergence is judged from the running peak.** Started from uniform
  freestream, the first residual is near zero (every cell away from the wall
  is in equilibrium), so normalising by it reports a residual that appears
  to grow through the transient.

## core/config.py

- **FP64 by default; FP32 is opt-in.** `case.general.precision` selects field
  arithmetic without changing JAX's process-wide x64 support. Verification
  references remain in FP64. Source mesh metrics and gradient geometry are
  constructed in FP64 before rounding; FP32 fields, Krylov vectors and line
  factors stay single precision. Norm sums and the tiny GMRES least-squares
  solve use FP64; CGS2 and full FP32 matrix-product precision are retained.
  Guards must be representable in the field dtype. Tolerances are never
  silently relaxed: FP32 has a higher rounding floor, particularly on stretched
  meshes, and aerodynamic coefficients must be checked against an FP64 run.
  SA's `f_w` must also have a finite directional derivative: its original
  sixth-power quotient can overflow in FP32 AD while the value remains finite.
  The FP32 branch uses reciprocal powers for large `g`, an algebraically
  equivalent form that avoids this overflow.
  The cap in `r_of` must guard its inactive quotient before division: clipping
  an overflowing JVP afterwards multiplies infinity by zero and produces NaN.
  Scalar Newton acceptance compares FP64 norm results without an intervening
  FP32 cast; otherwise a real small decrease can round to exactly one.
- **HLLC by default**: it resolves a stationary contact exactly and needs no
  entropy fix, so the scheme carries no tunable dissipation.
- **Krylov sizes.** The Arnoldi loop (solvers/krylov.py) runs to full length
  whether or not it has converged, testing only between restarts, so a large
  subspace is paid in full every step: small subspace (10), two restarts.
  Inexact Newton tolerance 1e-2 (Eisenstat and Walker, SIAM J. Sci. Comput.
  17 (1996)).
- **CFL ramp defaults.** Start at 10 (five times what the explicit path
  tolerates); growth capped at 1.3 per step and earned by the residual
  reduction; `cfl_max` is a backstop, not a tuning knob: the ramp settles at
  whatever the preconditioner sustains, limited in practice by
  `krylov_success`.

## numerics/riemann.py

- **Why HLLC.** Roe's scheme is shock-unstable on high-aspect-ratio cells, and
  the TMR grids reach such ratios at the shock foot: on 448x128 at M 0.8 the
  transonic shock never settled (CL oscillating 0.04-0.28), curable only by
  an entropy-fix coefficient large enough to bias CL by 16%. HLLC takes its
  dissipation from wave-speed estimates, has no vanishing eigenvalue, and
  needs no fix. On the O-grid at M 0.8, alpha 1.25, 128 squared, the two agree
  to three digits (CL 0.33567 vs 0.33585, CD 0.025711 vs 0.025698, observed
  orders 0.94/1.37 vs 0.92/1.33), so Roe stays as the independent check, 15%
  cheaper per iteration.
- **Contact resolution is required**: a slip wall has `vn = 0` along its
  length, so a flux that smears contacts smears exactly where CL and CD are
  computed. A Rusanov entry was built and removed for that reason.

## numerics/reconstruction.py, numerics/limiters.py

- **Raw index differences, not spacing-weighted gradients.** Limiting a
  gradient over the real cell spacings is the formally correct second-order
  form on a graded grid; it was built, measured and removed. On the strongly
  graded TMR C-grids it lost badly (transonic CL 0.1854 against 0.2461,
  oscillation 4.5x larger): the TVD region of a limiter depends on the spacing
  ratio and these limiters ignore it. On the mildly graded conformal O-grids
  it lost marginally, for no gain in order (CD 0.003103 against 0.002831 at
  128 squared, observed order 1.16 against 1.13). The weighted variant is in
  the git history.
- **Limiters take differences `(a, b)`, not ratios**: no 0/0 on a flat field,
  no guard epsilon. Every limiter must return exactly `a` at `a == b` (second
  order in smooth flow); this is unit-tested.
- **minmod vs van Albada.** On the explicit path on TMR's C-grids minmod held
  a transonic shock where van Albada reached 2.29 orders against minmod's
  4.39 (later confounded by those grids' own limit cycle). Van Albada came
  back for RAE 2822: smooth where active, close to the central
  slope in smooth flow. Van Leer is in the git history.
- **Venkatakrishnan is deliberately absent**: it is a smooth Barth-Jespersen
  for unstructured grids, not a slope-ratio limiter; fed `r = a/b` it returns
  `phi(1) = 3/4`, quietly losing second order.

## numerics/slabs.py (one case over several GPUs)

- **Explicit halo exchange for the stencils, automatic partitioning for the
  rest.** Left to XLA (GSPMD), the split step made one exchange per shifted
  array, ~1000 per step body: on the M6 1.08M grid, four A100s were 3.2x
  slower than one (all four GPUs talk over PCIe and the socket link, no
  NVLink). `block_residual` inside `shard_map`, each device taking `g` rows
  from either neighbour once per evaluation, brought it to 32 exchanges and
  four V100s 1.83x faster than one (2196 -> 1201 ms per step), a quarter of
  the memory each. The residual's stencils reach `g` cells, which is why
  `g` rows suffice and the unsplit edge padding reproduces the canvas ends.
- **No whole-canvas gather on a split canvas**: a flat gather of a split
  array is an all-gather (a copy of the state to every device). So the split
  atlas fills ghosts and face-average neighbours by gathering only the cells
  that move, and the line chains of a single 3D block are slices, run over
  every canvas row (ghost rows inert) so the split axis is never sliced. The
  unsplit atlas keeps the flat gathers: one code path each, same values.
- **Only the case's own grid is split**; grid-sequencing levels are small and
  run on one device (with the default allocator device 0 then keeps their
  memory: `XLA_PYTHON_CLIENT_ALLOCATOR=platform` for a sequenced split run).

## mesh/coarsen.py

- **Grid sequencing coarsens the case's own grid** (every other node, the
  segments' indices halved), so it needs no second grid file and works for
  generated, file and multi-block grids alike. Injection (each child takes its
  parent's value) is enough: from the settled 135k M6 level, the 1.08M level
  had its forces within 1e-4 of the 3000-step values for good from step 620,
  against 1760 from freestream. The canvas padding cells must keep the
  freestream start (a zero density there fails every step's physicality test).

## core/linalg.py

- **The line solve is block cyclic reduction**, Thomas kept as the reference
  in the tests. A Thomas sweep is `2L` dependent steps per application
  (`L` the chain length), each a small kernel: on an A100 the preconditioner
  was 89% of an RAE 2822 Newton step (121.7 against 13.6 ms with full-length
  GMRES). Odd-even reduction is `ceil(log2 L)` levels for about twice the
  flops: per apply 3.52 -> 0.136 ms on RAE 2822 (flow chains of 128), 2.08 ->
  0.24 ms on M6 135k, and still faster at 1.08M cells (4.81 -> 2.89 ms) once
  the power-of-two padding was dropped (with it, 1.08M was no faster).
  `lax.scan(unroll=8)` gave only 2-4x and was dropped.

## solvers/krylov.py

- **Right preconditioning.** On TMR's 112x32 C-grid at CFL 1000, ten
  left-preconditioned GMRES iterations returned a true residual 3.1x larger
  than they started with; ten right-preconditioned ones 0.47x (0.18x with the
  lines continued through the wake cut).

## derivatives.py, solvers/sensitivity.py

- **Defect correction, not Krylov, for the tangent solves.** On the
  converged `naca0012_laminar` and a 128x64 `clarky_sa`, GMRES on the exact
  second-order Jacobian, right-preconditioned by the line solves, stalls:
  GMRES(50) at 4.0e-4 (laminar) and 5.8e-3 (SA) relative residual, flat
  from 4000 iterations on; 800 unrestarted vectors 4.1e-3; GCROT(30, 20) and
  LGMRES 5e-3 and 2e-2 after 6200 products; an outer flexible GMRES around
  the Newton solve 3e-4. The system is consistent (`A^T r` of the stalled
  residual is not small) and its Ritz values have positive real parts
  (smallest 0.008): the line preconditioner, a first-order upwind Jacobian,
  does not see what the second-order one adds. Iterating the implicit
  solver's own Newton solve, `x <- x + G (b - A x)`, converges instead,
  about half an order every 50 steps: 1e-10 in ~1100 steps on SA. The
  derivatives then match central differences to 6e-7 (SA dCL/dalpha) and
  4e-6 (laminar).
- **No adjoint yet.** The transposed iteration, `lambda <- lambda + G^T
  (g - A^T lambda)` with `G^T` GMRES on the transposed Newton operator (the
  line preconditioner's transpose from `jax.linear_transpose`, exact), stalls
  at 0.62 for dCL/dU on the SA case after two steps: its inner GMRES makes no
  progress on residuals concentrated in the second wall cell. A linear `G`
  (the line solve alone, whose transpose would converge exactly as the
  tangent) diverges at CFL 20. Reverse mode also needs the face-gradient
  solve (numerics/gradients.py) guarded on degenerate canvas faces, whose
  `1/det` is masked forward but gives `0 * inf` backward.
- **A smooth limiter for derivatives.** `van_albada` cuts the slope to 0 at
  `a*b <= 0`, a kink in its derivative. On a transonic NACA 0012 (Euler,
  M 0.75, 2 deg, 96x48) the tangent then misses central differences by
  2e-3 in dCL/dalpha whatever the step; with the unclipped form
  (`van_albada_smooth`, `eps^2 = 1e-12`) by 5e-7 against differences
  extrapolated from 0.005 and 0.0025 deg (7.7e-5 at 0.01, all truncation).
- **Forces through one function.** `integrate_forces` converts
  `force_coefficients` (JAX, traceable in the freestream); against the
  NumPy sums it replaced, 1.1e-15 relative at most (2D, multi-body, 3D).

## mesh/atlas.py

- **Line chains stop at other blocks.** Continued through every interface,
  30P30N L1 ran 2.1x slower per step (chains of 190 cells instead of 66) for
  no convergence gain; only a block's own j-cut (the C-grid wake) is crossed.

## mesh/highlift.py, mesh/blocking.py (multi-element grids)

Measured on the 30P30N's elements at 150 wall cells per chord (75.9k cells
main + flap, 113k all three), quality as `tests/unit/test_highlift.py`
measures it.

- **One chain template, not three.** An O-layer's outer contour is rounded at
  every trailing edge (and the 30P30N's are blunt), so a cut leaving the nose
  tip meets the contour tangentially: blocks cannot be split there. The
  families are compositions per element pair: a slot block over the
  downstream element (left side the gap cut and the upstream base, top the
  upstream wake), a cove block (upstream lip to base, gap cut, downstream
  nose, mouth), a drop from every lip. Fill blocks have `j` from a wall where
  they touch one.
- **The gap is a side, never a chain member.** With the gap cut in the
  chain of one block over everything, 80 i-lines packed into the 0.005 gap
  ran up to the middle loop and Winslow's `psi` lost authority there (cells
  of 0.004-0.007 against 3e-4 wanted, 95x across the cut). As a side of both
  the cove and the slot it is 5-6x at worst, median 3.
- **Wakes are offsets of the slot's floor.** A Hermite wake from a slat's
  trailing edge cannot follow the main element's surface; the floor (the
  downstream contour and wake) offset node for node, by the gap growing 10 %
  per unit length, does, and gives the slot straight near-normal lines. The
  normals are smoothed over two offsets and the curve over one, then the
  nodes put back at their arc fractions (averaging alone slid them: 12.7x
  at the main element's base); unsmoothed, the flap's base creased the main
  wake (17-24x across it).
- **Connectors fixed, each block smoothed alone.** Smoothing all blocks at
  once with the connectors' nodes free (stencils across blocks) let corners
  collapse where three blocks meet (the slat cusp folded); anchoring the
  cuts' ends folded where anchored met free. Fixed Hermite cuts with spacing
  matched at their ends: no fold in any family.
- **Spacing control, iterated.** Thomas-Middlecoff `psi` interpolated from a
  block's sides ignores the layer it sits on (first cells 10-20x the layer's
  last step mid-block). `psi` from the heights wanted off the bottom and top
  (per j line, from its end nodes alone) fixed the medians; three
  GRAPE-style rounds (heights rescaled by sqrt(wanted/got)) brought the
  concave-corner peak from 26.6x to 22.8x and medians to 1.1-1.4; more
  rounds skewed the cove and slot (12 rounds: 11 deg) and cost 100 s.
- **Near-wall layer.** Hyperbolic marching folds into a concave corner
  (the main element's cove step) at any thickness above about a wall cell:
  its smoothing is raised 31x there, ramped from zero at the wall so the
  first steps stay normal; a residual fold halves the thickness round it and
  marches again. A sharp convex corner (the slat cusp, 18 deg) twisted its
  first cells until the corner's two neighbours were placed equally far
  (the step is normal to the chord through them). The march's smoothing at
  the wall tilted the first step 5-8 deg on the slat and main noses; from
  zero at the wall (except within two nodes of a convex corner, whose fan
  needs it) the wall is orthogonal to 0.05 deg. The first cell is
  re-marched until within 1 % of the wall spacing (up to 6 passes; the
  marching's area smoothing moved it 3-12 %).
- **Sharp trailing edges.** A sharp trailing edge is the layer's seam node:
  the convex-corner test (heading jumps, unwrapped) missed it, the wall got
  no smoothing there and the layer folded, then was thinned to the floor
  and refused (every sharp main element with a flap). Turning is taken from
  each node's two edges instead, and the edge's two neighbours placed equally
  far, as at the slat cusp. NACA-based sections then build at 22-38 deg.
- **Targets not met yet.** Smallest angle 20 deg: 17.4 deg at the
  main element's concave cove corner (one layer cell), 12.9 deg where the
  cove narrows into the flap gap (66 cells), 16.4 deg where the slot's left
  side turns from the gap cut onto the upstream base. Interface growth 1.3:
  22.8x at the concave corner (the layer is thinned there and the cove's
  lines fan), 5-7x round the gap; medians 1.1-1.4. Tried and dropped:
  splitting the cove at its corner (to the downstream nose in proportion, or
  at its nearest point: slivers of 0.1-7.5 deg), and starting the slot where
  the channel narrows to three gaps (its left side then ran along the
  ceiling, parallel to its floor, and folded).

## 3D

- **Face vectors from the diagonals' cross product**, `S = (x_c - x_a) x
  (x_d - x_b) / 2`: the vector area of every surface the face's edges bound,
  so a cell's six vectors sum to zero exactly on warped faces and a
  collapsed face (a pole) is exactly zero. Volumes `(1/3) sum S . x_f`
  equal the trilinear hexahedron's (tested against Gauss quadrature).
- **Face gradients: the 2D diamond generalized** (centre-to-centre vector
  plus the face's two diagonals, node values and positions both averaged
  from the 8 surrounding centres), not Green-Gauss on a dual cell: exact
  for linear fields on any hexahedra, the same property and code as 2D.
- **The pole is an interface**, `i_min` facing itself half a turn round the
  O seam: the ghost across the axis is the diametrically opposite cell. Its
  faces have zero area, so it only feeds slopes and gradients.
- **Plane-reflection ghosts on skewed cells** (walls, symmetry; 2D and 3D):
  the face state is off by O(h tan theta) on a cell skewed by theta, an
  O(1) local truncation error. Measured with the 3D MMS on a warped cube:
  first order in the solution with Dirichlet ghosts placed that way, second
  order with ghosts at the cells of linearly extrapolated nodes (a point
  reflection through the face midpoint). Kept for the physical conditions
  (standard; the M6 wall is within 2 deg of orthogonal except at the sharp
  trailing edge and the leading-edge seam); the MMS uses the latter.
- **3D incidence is in x-z, span y** (TMR's wing grids): `freestream_primitive(flow, 3)`
  is `M (cos a, 0, sin a)`; a 2D grid extruded in z needs the freestream
  passed explicitly (tests/verification/test_extruded.py does).

## Multi-block 3D (mesh/assemble3d.py, mesh/atlas3d.py)

- **Interfaces from the geometry, orientation from the corners.** Each
  boundary quad is matched to the one whose four corners coincide under one
  of the eight orientations of a quadrilateral (KD-tree on centroids,
  tolerance 5% of the shorter of the face's shortest edge and its cell's
  width, as in 2D). The orientation gives `axes` and `flip` of the
  `InterfaceRect`, so a one-cell-wide interface is oriented too, and partial
  faces and C-cuts come out of the same code. The CGNS `Transform` is not
  read; the file's 1-to-1 ranges only cross-check the matching.
- **Blocks keep the file's index order.** The wall-normal index is chosen in
  the atlas (`line_axis`: the normal of the wall faces, else of the
  far-field faces), where the conditions are known; the preconditioner's
  links index the canvas faces of all families concatenated. Transposing
  blocks on reading would have needed the conditions at assembly time.
- **Fill order per block, wall-normal pair first.** A ghost between two
  physical sides is filled by one condition applied to the other's ghost,
  and its centre reflected through two planes; two reflections commute only
  for orthogonal planes. With the stages in fixed index order (`j, i, k`), a
  re-indexed M6 split differed from the single block by 5e-4 (laminar) at
  the wall-symmetry edge; with the wall pair first, then by 4e-7 at the far
  field of the outer blocks (no wall: still `j` of their own frame); with the
  far-field normal as their line axis, 4e-15. Corners between two physical
  sides that are both off the line axis (e.g. symmetry and inflow) still
  depend on the index order: none of the grids here has one.
- **Repeated fill passes.** A stage extends each side over the ghosts of the
  stages before it, reading the partner's ghosts there. Across an interface
  that maps my `k` to its `i`, that partner ghost belongs to the same stage
  and is not filled yet. The three stages are repeated until a pass reads
  filled cells only, found once in NumPy (`_passes`): 1 pass for a single
  block (bitwise as before), 2 for the CRM grid and the permuted M6 splits.
  The edge ghosts round a vertex of three blocks become copies of one of
  them, value and centre together, so the diamond gradients stay exact for
  linear fields there (tests/verification/test_irregular3d.py).
- **Packing**: tiles in the plane of two canvas axes (the 2D `_pack`), as
  deep as the deepest along the third, choosing the axis that gives the
  smallest canvas. CRM: 1.70x the cells at 450k, 1.41x at 3.6M (ghost rings
  and gaps). A better packer only if memory requires it.
- **Measured (CPU)**: CRM 450k, assembly 7 s, atlas plans 14 s, freestream
  residual 1e-13 off the wall.
- **CRM 450k, `max_relative_update` off** (`adodg_crm_wing`). With the M6's
  0.2 the run reached `max_iters` (6000 steps) at 1.2 orders, the forces
  still drifting: the CFL never left its starting 10. GMRES was not the
  cause (linear reduction 0.001 at CFL 10, against 0.5 to succeed); the
  global relaxation was: omega = 0.04 every step, set by nu~ in ten cells
  of the first layers at the wing tip's trailing edge (an odd-even
  checkerboard there, the smallest and most skewed cells of the grid), and
  the CFL grows only on omega = 1. Copying the three-block edge ghosts from
  another neighbour left the residual there unchanged to five digits: not
  the vertex treatment. Without the bound the same case settles in 2940
  steps (3.95 orders): CL 0.502865, CD 0.022494. A bound applied per cell
  (or to the SA row alone) would keep both; a solver change, not made.
- **Memory**: 9.7 GB on one A100 for the CRM 450k canvas (0.77M cells),
  12.6 kB per canvas cell (nvidia-smi's peak, preallocation off; the M6's
  7.8 kB was XLA's compile-time estimate of temporaries, a different
  measure). The 3.6M level's canvas (5.07M cells) would need ~64 GB.
