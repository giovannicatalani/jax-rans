# ICMS section campaigns

An ICMS campaign uses `case.json` and `numerical.json`, just like a single
case. The normal command detects its `campaign`/`sections` blocks:

```bash
# Generate HUG files, check meshes and write geometry figures without solving.
python -m jaxrans examples/wing_sections --check

# Run all sections x Mach x incidence in GPU batches.
python -m jaxrans examples/wing_sections
```

`--out DIR`, `--quiet` and `--set file.section.key=value` work as for simple
cases. For example, `--set='case.campaign.mach=[0.5]'` selects one Mach and
`--set=numerical.convergence.max_iters=100` overrides the iteration limit.
Campaigns currently use one GPU; `--devices` must stay at its default of 1.
`general.precision` selects `fp64` (default) or `fp32` for every calculation
in the campaign, also overridable with `--set case.general.precision=fp32`.
See [precision and convergence](guide.md#precision) for the FP32 safeguards
and rounding limits; reducing `batch_size` remains useful if VRAM is tight.

The Python API separates inspectable preprocessing from GPU calculations:

```python
from jaxrans.campaign import load_campaign, prepare_campaign, run_campaign

campaign = load_campaign("examples/wing_sections")  # or a case.json path
prepared = prepare_campaign(campaign)
# Inspect prepared.directory / "wing_geometry.png" and the section figures.
result = run_campaign(prepared)
print(result.directory / "campaign.csv")
```

Install the existing `plot` extra for the geometric figures:
`uv sync --extra plot` (or `pip install 'jax-rans[plot]'`).
Preprocessing creates and checks every mesh without advancing the flow.
`run_campaign` requires an accessible JAX GPU, and uses one GPU for each batch.
`backend="cpu"` is an explicit option for small verification runs.

## Configuration

[The complete example](../examples/wing_sections/case.json) keeps the familiar
`general`, `domain`, `freestream`, `material_properties`, `turbulence` and optional
`boundary_conditions` blocks. Its numerical settings remain in `numerical.json`.

Two additional blocks replace the single geometry and scalar Mach/incidence:

```json
{
    "sections": {
        "file": "wing.icms",
        "count": 10,
        "units": "m",
        "header_lines": null
    },
    "campaign": {
        "mach": {"start": 0.3, "stop": 0.8, "step": 0.1},
        "alpha_deg": [-2.0, 0.0, 2.0, 4.0],
        "batch_size": 8
    }
}
```

- `sections.file` is relative to the JSON file, as are normal case inputs.
- `count` is the **number of profiles**, at least two. All have equal point counts.
- `units` is required: `m` or `mm`. Geometry reports use metres; HUG files use unit chord.
- `header_lines` is the number of physical header lines, or `null` (default) to
  take the complete final block of three-column numeric rows. Blank lines are
  ignored. For coordinate-like numeric header lines, set the header length
  explicitly. The parser never trims rows merely to make their count fit.
- `plane_tolerance` optionally sets the maximum distance to a section's fitted
  plane divided by its chord (default `1e-7`). Invalid geometry is rejected.
- Mach and incidence accept nonempty lists without duplicates, or inclusive
  `start`/`stop`/`step` ranges. The positive step must divide `stop-start`.
- `batch_size` limits the total number of independent CFD cases in a device
  batch, including all profiles, Mach and incidence (default 8).

`domain` supplies the usual `c_grid` or `o_grid` parameters. Do not specify
`domain.airfoil`: each HUG is generated from the ICMS. Do not specify
`freestream.mach`, `alpha_deg` or `sweep_deg`: these are expanded from the
campaign and geometry. One positive `freestream.reynolds` is imposed unchanged
for every case. Unknown campaign/section keys are errors. There are no kink
settings.

## Exactly what preprocessing does

The aircraft axes are X aft, Y outboard and Z up. Profiles must be ordered
root to tip, with strictly increasing quarter-chord Y. They may run in either
direction between their two trailing-edge corners. No supplied points may lie
along the trailing-edge base.

For each section, the trailing edge `T` is the mean of the first and last
points. The leading edge `L` is the **supplied point farthest from T**, not the
minimum X. The chord is `c = norm(T-L)`. The fitted plane is checked for
coplanarity. The local chord axis is `(T-L)/c`; the thickness axis is perpendicular
to it in the section plane and has positive aircraft Z. Translation, rotation
and division by `c` give `LE=(0,0)` and the **TE midpoint** `(1,0)`. A finite TE
gap is retained. No smoothing or aerodynamic resampling occurs in HUG export;
meshing subsequently redistributes points using the normal grid generator.

The supported ICMS cuts are line-of-flight (XZ) or perpendicular to local
dihedral. Their planes contain aircraft X. Once unfolded and detwisted, their
sections are supplied as `domain.section="streamwise"`; the existing 2.5D
mesher applies its `cos(sweep)` chord-direction scaling once. A cut normal to
the sweep line is a different input convention, and is rejected here when its
plane does not contain aircraft X. See [the existing swept model](swept.md).

Quarter-chord points are computed in the **original 3D geometry**:
`Q = L + 0.25*(T-L)`. At station `i>0`, the local sweep uses only the incoming
secant `Q[i]-Q[i-1]`. The root uses `Q[1]-Q[0]`. For this secant `(dx,dy,dz)`:

```text
sweep_deg = degrees(atan2(dx, hypot(dy, dz)))
```

This includes dihedral and gives aft sweep a positive sign. A station exactly
at a kink uses its inboard segment, without averaging across the kink. On a
smooth curved wing, the reported angle is the incoming-segment approximation,
not a central derivative.

The incidence passed to the 2.5D model is **exactly the input aircraft incidence**.
Neither twist nor dihedral is added to or subtracted from that incidence.
`chord_pitch_deg_te_up_positive` is a geometric diagnostic, positive when the
TE lies above the LE relative to aircraft X projected into the section plane.
It is not an incidence correction.

## Audit files

`prepare_campaign` writes:

- `preprocessing.json`: input SHA256, expanded flow ranges, transformation and
  sweep conventions, section reports and mesh quality.
- `wing_geometry.png`: original sections, their indices and quarter-chord line.
- `sections/section_NNN/profile.hug`: one element, lower TE → LE → upper TE.
- `sections/section_NNN/geometry.png`: original 3D geometry with aircraft axes
  and local basis; in-plane geometry before detwist; normalized profile with
  axes, LE, TE midpoint and quarter chord. Geometric axes have equal scales.
- `sections/section_NNN/geometry.json`: original line numbers, LE/TE, dimensional
  chord, basis, plane/reconstruction errors, contour reversal, selected sweep
  segment and meshing operations, including an explicitly requested TE closure.
- A runnable `case.json` / `numerical.json` per section at the first flow point.

For a section's normalized rows `p`, the inverse transformation is
`L + c * p @ basis[:, :2].T`. The original input order is available in the
`Section.normalized` array. The HUG contour is reversed if necessary to meet
the solver's convention; it contains no extra trailing-edge points.

`run_campaign` adds a global `campaign.csv`, per-section `polar.csv` files and
`execution.json`, plus the ordinary result files and full resolved JSON pair
under `calculations/section_NNN/mach_NNN/alpha_NNN/`. Paths use range indices
so close Mach or incidence values cannot collide through rounded filenames.
The tables retain each case's exit reason: convergence, settled forces,
maximum iterations, non-finite state or stalled solver.

## GPU batches

The campaign is the Cartesian product of sections × Mach × incidence. Jobs
interleave sections within each flow point. Each bounded batch is compiled
into one on-device iteration loop. Mach/incidence lanes sharing a geometry
use `jax.vmap`; distinct geometry groups form a tuple in the same compiled
loop, so their meshes need not have identical block shapes. Python assembles
the graph and monitors blocks; it does not run one host-side solve per case.
The viscosity and SA reference viscosity vary with `M/Re` for each lane even
though Reynolds stays fixed.

Completed or failed cases are frozen after monitoring a block; other cases
continue. Each case keeps its own residual and force convergence history.
Fields are written and transferred to host memory after each batch. Adjust
`batch_size` to device memory. Compilation time can grow with the number of
distinct geometries; no throughput assumption is made. Grid sequencing is
explicitly unsupported by this campaign runner; use its default value zero.

## Synthetic example

Regenerate the committed text input with:

```bash
python scripts/generate_icms_example.py
```

The modelled **half-wing extends from Y=0 to 15 m** in aircraft coordinates.
Nine reference stations are evenly spaced from 0 to 13 m, and the tenth is
at 15 m. Chord varies linearly from 6 to 2 m and NACA 00xx relative thickness
from 18% to 10%, with continuous thickness values between the endpoints.
LE X is zero and twist is zero. Reference height is continuous:

```text
Z(Y) = min(Y,13)*tan(5 deg) + max(Y-13,0)*tan(30 deg)
```

The cuts are perpendicular to local dihedral: 5 degrees below Y=13 and
30 degrees at and above Y=13. The section contour therefore varies in Y as
well as X/Z; the quoted Y is the LE/reference station. Traversal direction
alternates between sections. The NACA polynomial has an open trailing edge.
The tapered wing's quarter-chord line has forward sweep even though the LE
has zero sweep; its angle is computed from the geometry, not assumed zero.

Tests cover both cutting planes, arbitrary twist/placement, units, contour
directions, open/closed TEs, exact transforms, kink secants, strict configuration,
audit outputs, and multi-geometry/multi-Mach batch equivalence to individual
laminar and SA runs.
