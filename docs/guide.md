# Running a case

A case is a directory with two files: `case.json` describes the
problem (grid, freestream, gas, turbulence model, boundary conditions) and
`numerical.json` the method (flux, limiter, solver, iterations). Every key is
listed in the [case reference](case_reference.md). To set up your own
geometry, see [your own airfoil](own-airfoil.md),
[a multi-element airfoil](multi-element.md) or [a 3D wing](wing-3d.md).

## Examples

| example | flow | grid |
|---|---|---|
| `naca0012_euler` | Euler, M 0.8, 1.25 deg | generated O-grid |
| `naca0012_laminar` | laminar, M 0.5, 1 deg, Re 5000 | generated C-grid |
| `clarky_sa` | SA RANS, M 0.2, 4 deg, Re 3e6 | C-grid round a downloaded coordinate file |
| `naca0012_sa` | SA RANS, M 0.15, 10 deg, Re 6e6 | NASA TMR C-grid from file |
| `rae2822_transonic_sa` | transonic SA RANS, M 0.729, 2.31 deg, Re 6.5e6 | NASA NPARC C-grid from file |
| `flatplate_sa` | SA RANS boundary layer, M 0.2, Re 5e6 | NASA TMR flat-plate grid |
| `30p30n_sa` | three-element high lift, M 0.17, 5.5 deg, Re 1.71e6 | JAXA 117-block grid from file |
| `main_flap_sa` | two-element high lift (30P30N main and flap), M 0.17, 5.5 deg, Re 1.71e6 | generated multi-element grid |

The grids read from files are not in the repository; `data/grids/README.md`
says where to download each. The first three examples and `main_flap_sa`
need nothing else.

## Running

```bash
jaxrans examples/naca0012_euler              # or: python -m jaxrans examples/naca0012_euler
jaxrans examples/naca0012_euler --check      # build and draw the grid, do not solve
jaxrans examples/naca0012_sa --set case.freestream.alpha_deg=12
jaxrans my_case --out /scratch/me/run1 --quiet
jaxrans my_wing --devices 4                  # one case split over 4 GPUs (too large for one)
```

`--set file.section.key=value` changes one key for this run, and can be
repeated. `--devices N` splits the case's grid in N slabs along its first index,
one per GPU; the first grid index (ghost cells included) must divide by N.
On four V100s (PCIe, no NVLink) the ONERA M6 1.08M grid runs 1.8x faster
than on one, with a quarter of the memory on each; the slabs exchange two
rows of cells per residual evaluation. From Python:

```python
from jaxrans import load_case, run

case = load_case("examples/naca0012_sa", overrides=("case.freestream.alpha_deg=12",))
result = run(case)
result.write()
print(result.forces.cl, result.forces.cd, result.history.orders_dropped)
```

The validation cases in `validation/cases/` run the same way.

A run stops at the first of: its forces settled (`settled`: CL, CD and CM each
vary by less than `convergence.forces_tol`, 1e-4, over the last quarter of the
iterations), the residual down by `tol` from its peak (`converged`), or
`max_iters`. It also stops, as `stalled`, if a whole block of iterations left
the state unchanged (every step rejected), and as `non_finite` on a NaN.
`convergence.grid_sequencing = n` solves first on the grid coarsened `n`
times (every other node each time) and starts each finer grid from the
coarser solution: fewer iterations on a fine grid. Every block's cell counts
and patch and interface limits must stay even (JAXA's 30P30N grids, for one,
are refused at the start). Polars (below) run every angle to `tol` or
`max_iters`, on the case's grid only.

## Output

Results go to `<case>/results/<case_name>/`, or to `--out`:

| file | content |
|---|---|
| `*_fields.npz` | the solution `U` and the grid nodes, per block |
| `*_surface.dat` | x, Cp and Cf on every wall face; with several bodies also y and a `body` column |
| `*_residual.dat`, `*_lift.dat` | convergence histories |
| `*_metadata.json` | forces, convergence, the case files as run, git hash, versions, device |

`python -m jaxrans.plot <result dir>` writes figures to `figures/` there:
Mach and Cp fields, wall Cp and Cf, convergence. It only reads these files,
so you can plot results copied back from a cluster without a GPU.

`U` is conservative, `(rho, rho u, rho v, E)`, plus `rho nu~ / nu_inf` with
SA. Everything is nondimensional, based on freestream density, freestream
speed of sound and the chord, so the freestream is `rho = 1`,
`(u, v) = M (cos a, sin a)`, `p = 1/gamma`.

## Polars

```bash
python -m jaxrans.polar examples/rae2822_transonic_sa --alpha -2 5.5 0.5
```

`--alpha` takes start, stop and step. All the angles run together as one
batch, split over every visible GPU. The output is `polar.dat` (one row per
angle), `polar_metadata.json` and a normal result directory per angle.

To initialize each angle from the previous solution, use continuation:

```bash
python -m jaxrans.polar examples/rae2822_transonic_sa --alpha -2 5.5 0.5 --mode continuation
```

Angles follow the requested order; `--alpha 5.5 -2 -0.5` follows a descending
branch. The first angle starts from freestream (with grid sequencing if
configured). Later angles inherit the conservative interior field, including
the turbulence variable, on the same mesh. Boundary conditions use the new
angle; CFL, residual reference and convergence histories start afresh.
The chain advances only after residual convergence or the configured force
settling criterion. A non-finite result, stalled solver or iteration limit
stops that profile's chain, retaining the failed point and earlier results.
The command writes partial output and exits with status 1 on such a failure.
Metadata records the requested angles, each point's status and its source angle.
The table's `settled` column remains a CL-spread diagnostic, distinct from the
solver's exit reason.

### Several profiles with continuation

```bash
python -m jaxrans.campaign examples/polar_campaign.json --batch-size 2 --out results/campaign
```

The manifest lists profiles; case paths are relative to the manifest:

```json
{
  "batch_size": 2,
  "profiles": [
    {"name": "naca0012", "case": "naca0012_euler", "alpha": [-2, 6, 1]},
    {"name": "naca2412", "case": "naca0012_euler", "alpha": [-2, 6, 1],
     "overrides": ["case.domain.airfoil=naca2412"]}
  ]
}
```

Each profile may instead provide an explicit `"alphas": [0, 1, 2, 1, 0]`
list, allowing different ranges and sweep directions. Profile names must be
unique directory names consisting of letters, digits, `_`, `-`, or `.` and
starting with a letter or digit. Profiles with the same topology, geometry
array shapes, physics and solver settings run in one `jax.vmap` batch, with
their own geometry and wall distance. Different Mach/Reynolds values, models,
solver settings or mesh connectivity form separate groups automatically;
arbitrary meshes are not padded into one common mesh. `--batch-size` limits
the profiles advanced together. `--devices` distributes the profile axis
over visible devices (default: as many as the batch can use).

Angles advance in synchronized rounds. A converged profile's complete solver
state is frozen between blocks while other profiles finish their angle.
Each profile retains its own stopping iteration and trimmed history. If one
profile fails, the other chains continue. A shorter chain becomes inactive
after its last angle. Optional coarse-grid startup runs individually before
the first final-grid batch; it is not repeated at later angles.

Output contains one polar directory per profile and `campaign_metadata.json`
with batch membership, wall time, requested/calculated angles and exit reasons.
Each polar's timing sums its active rounds' shared batch time; these times
must not be added to estimate campaign wall time. `case_iterations` counts
actual steps per profile, excluding frozen waits. Repeated angles or angles
that round to the same two-decimal directory name use indexed point directories.
Campaign exit status is 0 for success, 1 for an incomplete chain, and 2 for
invalid input. Measure wall time against the independent-angle batch on your
case: continuation can reduce iterations while changing GPU utilization.
Near stall, increasing and decreasing sweeps can follow different branches.

The Python interface is:

```python
from jaxrans import load_case
from jaxrans.campaign import PolarRequest, run_campaign

campaign = run_campaign([
    PolarRequest("profile_a", load_case("my_case_a"), [-2, 0, 2, 4]),
    PolarRequest("profile_b", load_case("my_case_b"), [4, 2, 0]),
], batch_size=2)
campaign.write("results/campaign")
```

For a single profile, `run_polar(case, alphas, mode="continuation")` uses
the same implementation. `run(case, grid, initial=previous.U)` also supports
a same-mesh restart and bypasses grid sequencing.

## Boundary conditions

Every grid has named patches on its boundaries, and `boundary_conditions`
gives each one a type:

```json
"boundary_conditions": {
    "wall": {"type": "no_slip_wall"},
    "farfield": {"type": "far_field", "model": "point_vortex"}
}
```

The types are `slip_wall`, `no_slip_wall` (adiabatic), `symmetry`,
`far_field` (characteristic; `model` is `freestream` or `point_vortex`),
`inflow_total` and `outflow_static`.

A patch you leave out gets the default for its name: `wall` is no-slip in a
viscous case and slip otherwise, `farfield` is `far_field`, `symmetry` is
`symmetry`, `inflow` is `inflow_total`, `outflow` is `outflow_static`. Any
other name must be given a type.

Where the patch names come from:

- **Generated grids** have `wall` and `farfield`.
- **CGNS files**: the family name of each boundary, else its BC type
  (`BCWall*` is `wall`, `BCFarfield` is `farfield`, and so on), else the
  name of the BC node.
- **Faces a file does not name** (all of them, in Plot3D) are named by
  `domain.patches` box rules if you give any, else by the loop they lie on:
  the outer boundary is `farfield`, each body is `wall`.

Block interfaces, the seam of an O-grid and the wake cut of a C-grid are
found from coincident faces. They are not patches, and you never declare
them.

CGNS files in the older ADF format are refused. Convert them with the CGNS
tools: `adf2hdf in.cgns out.cgns`.
