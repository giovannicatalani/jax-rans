"""Blunt-TE C-grid baseline and complete multiblock qualification archives."""

from dataclasses import asdict
import json

import numpy as np

BASELINE = "naca0012_blunt_c_grid_v1"
DEFAULT_OUTPUT = "results/sectional_c_blunt"


def require_baseline(case):
    if case.grid_spec.grid != "c_grid" or case.grid_spec.airfoil != "naca0012_blunt":
        raise ValueError("sectional qualification requires the naca0012_blunt C-grid baseline")


def normal_grid(case, stride):
    """Fixed contour/base/wake; vary only the main C-block's normal index.

    The two auxiliary blocks keep their level-1 resolution, matching the
    fixed tangential stride 2 on the main block's j_min interfaces.
    """
    from jaxrans.mesh.assemble import RawBlock, assemble
    from jaxrans.mesh.mesh import assign_conditions
    from jaxrans.mesh.sectional import build_sectional_atlas

    require_baseline(case)
    if stride not in (1, 2, 4):
        raise ValueError("normal stride must be 1, 2 or 4")
    fine = case.grid()
    if len(fine.reference.mesh) != 3:
        raise ValueError("expected the three-block blunt C-grid")
    raw = []
    for k, block in enumerate(fine.reference.mesh):
        nodes = np.asarray(block.nodes)[:, ::2, ::(stride if k == 0 else 2)]
        raw.append(RawBlock(*nodes))
    mesh, _ = assemble(raw)
    conditions = {seg.patch: seg.kind for block in fine.reference.mesh
                  for _, seg in block.sides.boundaries()}
    mesh = assign_conditions(mesh, conditions)
    return build_sectional_atlas(mesh, fine.spec, fine.g, fine.depth,
                                 fine.relaxation_start, fine.relaxation_end, fine.geometry)


def archive_arrays(result):
    """All blocks and all wall faces, including the trailing-edge base.

    `primitive` concatenates the independent cells of EVERY block. Explicit
    per-block arrays retain structured connectivity for field diagnostics.
    """
    from jaxrans.mesh.block import SIDES
    from jaxrans.numerics.residual import primitive_from_conservative

    grid, flow = result.grid, result.case.flow
    reference = grid.reference if hasattr(grid, "reference") else grid
    fields = []
    arrays = {}
    for k, block in enumerate(grid.unpack(primitive_from_conservative(result.U, flow.gamma, grid.components))):
        primitive = np.asarray(block)
        if grid.dim == 3:
            primitive = primitive[..., 0]
        if grid.components == 2:
            primitive = np.concatenate([primitive[:3], np.zeros_like(primitive[:1]), primitive[3:]])
        fields.append(primitive.reshape(len(primitive), -1))
        arrays[f"primitive_b{k}"] = primitive
        arrays[f"nodes_b{k}"] = np.asarray(grid.mesh[k].nodes)
    faces = reference.walls.faces
    a, b = faces.a.T, faces.b.T
    if getattr(grid, "geometry", None) is not None:
        a, b = (grid.geometry.streamwise_points(p) for p in (a, b))
    elif flow.swept:
        scale = np.array([[np.cos(flow.sweep_rad)], [1.]])
        a, b = a / scale, b / scale
    ds = np.linalg.norm(b - a, axis=0)
    # 0: lower, 1: upper, 2: blunt base. Unit-chord streamwise geometry.
    base = np.isclose(a[0], 1., atol=1e-10, rtol=0) & np.isclose(b[0], 1., atol=1e-10, rtol=0)
    region = np.where(base, 2, np.where((a[1] + b[1]) > 0, 1, 0))
    region_arc = np.empty(len(ds))
    for r in np.unique(region):
        mask = region == r
        region_arc[mask] = (np.cumsum(ds[mask]) - .5 * ds[mask]) / ds[mask].sum()
    arrays.update(primitive=np.concatenate(fields, axis=1), nodes=arrays["nodes_b0"],
                  block_shapes=np.array([p.shape[1:] for p in (arrays[f"primitive_b{k}"] for k in range(len(fields)))]),
                  wall_block=faces.block, wall_side=faces.side, wall_cell=faces.cell,
                  wall_region=region, region_arc=region_arc,
                  topology=np.array(json.dumps([asdict(b.sides) for b in reference.mesh])),
                  side_names=np.array(SIDES), baseline=np.array(BASELINE),
                  arc=(np.cumsum(ds) - .5 * ds) / ds.sum(), weight=ds,
                  residual_history=result.history.residual, **result.surface())
    return arrays
