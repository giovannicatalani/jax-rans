"""Reproduce examples/wing_sections/wing.icms (no external geometry files).

    python scripts/generate_icms_example.py

15 m is the aircraft-Y extent of the modelled half-wing, not developed span.
Zero LE sweep, zero twist. Ten stations include the dihedral break at Y=13 m.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jaxrans.geometry.airfoil import naca4


def generate(path: Path, *, units="m", cut="dihedral_normal", n=101):
    if units not in ("m", "mm") or cut not in ("dihedral_normal", "line_of_flight"):
        raise ValueError("invalid units or cut")
    stations = np.concatenate([np.linspace(0., 13., 9), [15.]])
    base = naca4("0010", n=n, sharp=False)
    profiles = []
    for index, y in enumerate(stations):
        chord = 6. - 4.*y/15.
        relative_thickness = 0.18 - 0.08*y/15.
        dihedral = np.radians(5. if y < 13. else 30.)
        height = min(y, 13.)*np.tan(np.radians(5.)) + max(y-13., 0.)*np.tan(np.radians(30.))
        thickness_axis = np.array([0., -np.sin(dihedral), np.cos(dihedral)]) if cut == "dihedral_normal" else np.array([0., 0., 1.])
        points = np.array([0., y, height]) + np.column_stack((base.x*chord, np.zeros(len(base.x)), np.zeros(len(base.x))))
        points += (base.y*chord*relative_thickness/0.10)[:, None]*thickness_axis
        # Alternate orientation to exercise both supported contour directions.
        profiles.append(points[::-1] if index % 2 else points)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("Synthetic trapezoidal half-wing with winglet\n")
        stream.write(f"Aircraft axes: X aft, Y outboard, Z up; units {units}\n")
        stream.write("LE X=0; chord 6 to 2 m; relative thickness 0.18 to 0.10; zero twist\n")
        stream.write("Reference stations: Y=0..15 m; height continuous; dihedral 5 deg below 13 m, 30 deg above\n")
        stream.write(f"Cuts: {cut}; open trailing edges; alternating contour direction\n")
        for points in profiles:
            np.savetxt(stream, points*(1000 if units == "mm" else 1), fmt="%.16e")
    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path("examples/wing_sections/wing.icms"))
    generate(parser.parse_args().path)
