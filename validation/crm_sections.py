"""Wall pressure of a 3D wing result at span stations, from its surface file.

    python validation/crm_sections.py RESULT_DIR [--eta 0.1 0.25 ...]

Works on any 3D result, whatever its blocks: the wall faces whose centres
lie within a thin band about each station ``y = eta * semispan`` are taken,
and ``-Cp`` drawn against each face's chord fraction at its own ``y``. Writes
``figures/sections.png`` in the result directory. Reads files only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_surface(directory: Path) -> dict[str, np.ndarray]:
    (path,) = directory.glob("*_surface.dat")
    names = path.read_text().splitlines()[0].lstrip("# ").split()
    data = np.loadtxt(path)
    return dict(zip(names, data.T))


def chord_fraction(surface: dict[str, np.ndarray], bins: int = 60) -> np.ndarray:
    """Each wall face's ``x`` as a fraction of the local chord at its own
    ``y``: leading and trailing edges as the least and largest ``x`` of the
    wall in spanwise bins, interpolated (a swept wing's rows then collapse)."""
    x, y = surface["x_over_c"], surface["y_over_c"]
    edges = np.linspace(y.min(), y.max(), bins + 1)
    which = np.clip(np.digitize(y, edges) - 1, 0, bins - 1)
    centre = np.array([y[which == b].mean() for b in range(bins)])
    lead = np.array([x[which == b].min() for b in range(bins)])
    trail = np.array([x[which == b].max() for b in range(bins)])
    le, te = np.interp(y, centre, lead), np.interp(y, centre, trail)
    return (x - le) / (te - le)


def band(surface: dict[str, np.ndarray], fraction: np.ndarray, y: float, width: float):
    """Chord fraction and Cp of the wall faces within ``width`` of ``y``."""
    take = np.abs(surface["y_over_c"] - y) < width
    return fraction[take], surface["cp"][take]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result", type=Path)
    parser.add_argument("--eta", type=float, nargs="+", default=[0.131, 0.283, 0.502, 0.727, 0.846, 0.950])
    parser.add_argument("--width", type=float, default=0.02, help="half-width of the band, in semispans (about one row of wall faces)")
    args = parser.parse_args()
    surface = load_surface(args.result)
    semispan = surface["y_over_c"].max()
    fraction = chord_fraction(surface)
    fig, axes = plt.subplots(2, (len(args.eta) + 1) // 2, figsize=(4 * ((len(args.eta) + 1) // 2), 7),
                             sharey=True, constrained_layout=True)
    for ax, eta in zip(axes.ravel(), args.eta):
        x, cp = band(surface, fraction, eta * semispan, args.width * semispan)
        ax.plot(x, -cp, ".", ms=3, color="#2a5d8f")
        ax.set_title(f"eta = {eta:.3f}")
        ax.set_xlabel("x / c (local)")
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("-Cp")
    out = args.result / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "sections.png", dpi=130)
    print(out / "sections.png")


if __name__ == "__main__":
    main()
