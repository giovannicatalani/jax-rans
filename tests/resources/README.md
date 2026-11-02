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
