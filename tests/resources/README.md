# Test geometry resources

`naca0012_blunt.hug` is a unit-chord NACA 0012 with a finite trailing-edge
base of thickness 0.00252 (0.252 % chord). It contains one element and 401
points: lower trailing-edge corner, lower surface, leading edge, upper
surface, upper trailing-edge corner. There are no points on the base.

The coordinates use 201 cosine-spaced chordwise stations and the standard
NACA half-thickness formula:

```text
x = (1 - cos(beta)) / 2, beta = 0 ... pi
y_t = 0.6 * (0.2969 sqrt(x) - 0.1260 x - 0.3516 x² + 0.2843 x³ - 0.1015 x⁴)
```

The coefficient -0.1015 preserves the open trailing edge. The static file is
read directly by the mesh-generation test in `tests/unit/test_hug.py`; it is
not regenerated during the test.

## Public high-lift section: 30P30N

`30p30n_blunt/30p30n_blunt.hug` contains the public JAXA/BANC 30P30N
**modified slat config F** geometry: slat, main wing, flap, all with finite
trailing-edge bases. The source is JAXA's
[APC-V geometry release](https://cfdws.chofu.jaxa.jp/apc/apc5/config.html),
archive `30P30N_modified_slat_configF.tar`, mirrored in this repository as
`validation/references/jaxa_30p30n/2010_30p30n_thik_te.crv`.
See that reference directory's README for the experimental configuration.

Conversion performed once using `geometry.elements.read_elements`: join the
source curves, open each contour at its two trailing-edge corners, order
slat/main/flap, and divide all coordinates by the 18-inch stowed chord.
The resulting file contains 201, 641 and 201 points respectively. It keeps
the deployed positions, has no points on the bases, and needs no further
coordinate scaling (`coordinate_scale = 1`). Tests read this static file
directly, without downloading or converting geometry at runtime.

From the repository root, check and draw the mesh without solving:

```bash
uv run --extra plot python -m jaxrans tests/resources/30p30n_blunt --check --out /tmp/30p30n_hug
uv run --extra plot python scripts/plot_blocks.py tests/resources/30p30n_blunt --window -0.13 1.2 -0.23 0.15 --window 0.82 0.94 -0.03 0.07 --out /tmp/30p30n_blocks.png
```

The case uses `trailing_edge_topology: blunt_c` and 150 cells per chord.
It produces 15 blocks and 113,248 cells, with no folded cells. At these
settings the quality report still flags locally skewed cells (minimum angle
about 9.7 degrees) and neighbouring-cell growth up to 3.84. The first-cell
height is checked away from corner fans and cove vertices, as in the existing
high-lift tests; the global nodal height includes displacements at those corners.
This resource tests geometry and mesh generation, not agreement with
experimental aerodynamic loads.
