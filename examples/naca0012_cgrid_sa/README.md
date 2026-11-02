# NACA 0012: blunt-base C-grid, SA RANS

This example uses `trailing_edge_topology: blunt_c` and the bundled
`naca0012_blunt.hug`. Its 401 points run from the lower trailing-edge corner
to the upper one, without points on the base. The geometry is a copy of
`tests/resources/naca0012_blunt.hug`: a unit-chord NACA 0012 with a base
thickness of 0.00252. The mesh preserves this base and fills the downstream
region with a base-layer block and a finite-width wake block.

The default conditions are Mach 0.8, incidence 1.25 degrees, sweep 25 degrees,
Reynolds number 6 million, and temperature 288.15 K. The geometry is given
in the streamwise section. The solver is implicit with SA-noft2, Sutherland
viscosity and no limiter, matching the companion O-grid example's current
settings. The mesh has three blocks and 38,176 cells at these settings.

From the repository root:

```bash
python -m jaxrans examples/naca0012_cgrid_sa --check
python -m jaxrans examples/naca0012_cgrid_sa
```

To draw the blocks and zoom in on the base (requires matplotlib):

```bash
python scripts/plot_blocks.py examples/naca0012_cgrid_sa --window -0.05 1.4 -0.2 0.2 --window 0.995 1.015 -0.004 0.004 --out /tmp/naca0012_cgrid_blocks.png
```

The block plot shows the input streamwise geometry; the swept solver scales
its x coordinates by cos(sweep) to work in the normal section.

The companion `examples/naca0012_ogrid_sa` uses the analytical
`naca0012_blunt` geometry with the same thickness formula and the legacy
O-grid. To compare results, keep the physical conditions, force references
and numerical settings identical in both cases, and check grid convergence.
See the root README for the tested WSL2/CUDA and Triton workarounds.
