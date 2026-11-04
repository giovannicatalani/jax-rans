"""Headless geometric audit figures; Matplotlib is the existing plot extra."""

from __future__ import annotations

import numpy as np


def plot_sections(sections, directory):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def figure(**kwargs):
        fig = Figure(**kwargs)
        FigureCanvasAgg(fig)
        return fig

    for section in sections:
        s = section
        fig = figure(figsize=(14, 4.5), layout="constrained")
        fig.get_layout_engine().set(wspace=0.12)
        ax = fig.add_subplot(131, projection="3d")
        ax.plot(*s.points.T)
        ax.scatter(*s.leading_edge, label="LE")
        ax.scatter(*s.trailing_edge, label="TE midpoint")
        for vector, name, color in zip(s.basis.T, ("chord", "thickness", "normal"), ("r", "g", "b")):
            end = s.leading_edge + 0.3*s.chord*vector
            ax.quiver(*s.leading_edge, *(0.3*s.chord*vector), color=color)
            ax.text(*end, name)
        ax.set(xlabel="X aircraft [m]", ylabel="Y aircraft [m]", zlabel="Z aircraft [m]",
               title="Original section and local axes")
        # Equal physical scale, with nonzero limits for a constant-Y plane.
        centre = (s.points.max(axis=0) + s.points.min(axis=0))/2
        radius = 0.6*s.chord
        ax.set(xlim=(centre[0]-radius, centre[0]+radius), ylim=(centre[1]-radius, centre[1]+radius),
               zlim=(centre[2]-radius, centre[2]+radius))
        ax.set_box_aspect((1, 1, 1))
        ax = fig.add_subplot(132)
        normal = s.basis[:, 2]
        ex = np.array([1., 0., 0.]) - normal[0]*normal
        ex /= np.linalg.norm(ex)
        up = np.cross(normal, ex)
        if up[2] < 0:
            up = -up
        local = (s.points-s.leading_edge) @ np.column_stack((ex, up))
        te = (s.trailing_edge-s.leading_edge) @ np.column_stack((ex, up))
        ax.plot(*local.T)
        ax.plot([0, te[0]], [0, te[1]], "r--", label="Chord")
        ax.scatter([0, te[0]], [0, te[1]])
        ax.set(xlabel="Projected aircraft X [m]", ylabel="In-plane up [m]",
               title=f"Before detwist; TE-up pitch {s.chord_pitch_deg:.3f} deg")
        ax.axis("equal")
        ax.grid(True)
        ax = fig.add_subplot(133)
        ax.plot(*s.normalized.T)
        ax.plot([0, 1], [0, 0], "r--")
        for x, z, label in ((0, 0, "LE (0,0)"), (1, 0, "TE midpoint (1,0)"), (0.25, 0, "c/4")):
            ax.scatter(x, z)
            ax.annotate(label, (x, z), xytext=(-85 if x == 1 else 0, 12),
                        textcoords="offset points", fontsize=8)
        ax.set(xlabel="x/c", ylabel="z/c (thickness)", title="Detwisted, unit chord")
        ax.axis("equal")
        ax.grid(True)
        fig.suptitle(f"Section {s.index:03d} | chord {s.chord:.5g} m | sweep {s.sweep_deg:.4f} deg")
        fig.savefig(directory / "sections" / f"section_{s.index:03d}" / "geometry.png", dpi=160)
        fig.clear()

    fig = figure(figsize=(10, 6), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    for s in sections:
        ax.plot(*s.points.T, linewidth=0.6)
        ax.text(*s.quarter_chord, str(s.index))
    q = np.array([s.quarter_chord for s in sections])
    ax.plot(*q.T, "ro-", label="Quarter-chord line; incoming secants")
    ax.set(xlabel="X aircraft [m]", ylabel="Y aircraft [m]", zlabel="Z aircraft [m]")
    all_points = np.concatenate([s.points for s in sections])
    ax.set_box_aspect(np.maximum(np.ptp(all_points, axis=0), 0.1))
    ax.legend()
    fig.savefig(directory / "wing_geometry.png", dpi=160)
    fig.clear()
