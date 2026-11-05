# A swept wing section (2.5D)

The [sectional 3D stencil](sectional.md) provides a separate zero-taper
verification path for the future conical solver, using the same flow and
section conventions.

One station of a swept wing, computed as an **infinite swept wing**: the
section is the same all along the sweep line, so nothing varies along it. The
grid is an ordinary 2D grid; the flow has three velocity components. It does
not capture taper, the root or the tip: a wing assembled from 2.5D stations
needs the stations' effective incidence from a lifting-line or vortex-lattice
model of the planform.

## 1. The case

Add `sweep_deg` to the freestream of any 2D case:

```json
"domain": {"grid": "c_grid", "airfoil": "rae2822.dat", "...": "...", "section": "streamwise"},
"freestream": {"mach": 0.842, "alpha_deg": 2.0, "sweep_deg": 30.0, "reynolds": 6.5e6, "temperature": 460.0}
```

- `mach` and `alpha_deg` are the freestream's: the Mach number of the whole
  velocity, and the incidence in the streamwise plane.
- `domain.section` says which section the airfoil (or grid file) is:
  `streamwise` (the default: the cut parallel to the freestream, as wings are
  usually defined) or `normal` (the cut normal to the sweep line). A
  streamwise section is meshed normal to the sweep line: the solver scales the
  grid's `x` by `cos(sweep)`, which keeps the thickness on a shorter chord
  (a 12 % streamwise section is 13.9 % thick normal to a 30 degree sweep line).
- Lengths stay in streamwise units: `reynolds` is per unit length as in 2D,
  and `reference_length` and `moment_centre` are streamwise.
- The far field needs the Mach number **in the plane of the grid** below 1,
  `M sqrt(cos^2(alpha) cos^2(sweep) + sin^2(alpha))`, so a supersonic
  freestream over a subsonic leading edge is accepted. Both far-field models
  (`freestream`, `point_vortex`) work.
- `sweep_deg: 0` runs the 2.5D path without sweep: the 2D solution, with a
  zero spanwise velocity.

`examples/rae2822_swept_sa` is a worked case.

## 2. The frames

```
  top view                                   the grid (normal section)
            S (span)
            ^     / sweep line, z                   y
            |   /                                    ^
            | /  sweep                               |  ___
            +--------> X (freestream)                | /   \______  airfoil, as in 2D
             \                                       +--------------> x
              \ x (normal chord)                    z: out of the page, along the sweep line
```

In the grid's frame (`x` along the normal chord, `y` up, `z` along the sweep
line) the freestream is

    V = M (cos(alpha) cos(sweep),  sin(alpha),  cos(alpha) sin(sweep))

an in-plane flow at the in-plane Mach number and incidence
`atan(tan(alpha) / cos(sweep))`, plus a uniform spanwise component. The
equations are the 3D ones with every derivative along `z` zero; the state
carries `rho w` as a fifth mean-flow row (`U = (rho, rho u, rho v, rho w, E[, rho nu~])`).

## 3. What comes out

- **Force coefficients** in the wing frame, per unit span, per `q_inf`
  (the whole freestream) times `reference_length`: `CL` normal to the
  freestream in the vertical plane, `CD` along it, `CM` about `moment_centre`
  in streamwise coordinates (pressure only, as in 2D), and `CY`, the spanwise
  force. Pressure acts normal to the swept generators, so it has a spanwise
  part, `-tan(sweep)` times its streamwise part; friction adds to it.
- **Surface** (`*_surface.dat`): `x` in the normal section, `x_streamwise`
  (`x / cos(sweep)`, the station's streamwise coordinate), `y`, `cp`, and
  viscous, `cf` along the chord and `cf_z` along the sweep line; all over the
  whole `q_inf`. `cp` is the pressure itself, so it compares directly with a
  3D wing's `cp` at the station, plotted at `x_streamwise`.
- **Fields** (`*_fields.npz`): `U` with its `rho w` row.


## 4. Limits

- Fully turbulent (SA, SA-QCR): no crossflow transition.
- An infinite wing: no taper, root or tip effects.
- 2D grids only; derivatives (`jaxrans.sensitivity`) are in `alpha` and
  `mach` at fixed sweep.
