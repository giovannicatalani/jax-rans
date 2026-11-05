# Blunt-TE C-grid sectional baseline checks

Active baseline: `naca0012_blunt_c_grid_v1`. These are numerical checks, not an external aerodynamic reference.

## Completed checks

- 104 targeted tests pass: 26 qualification tests, 15 boundary-layer diagnostics tests, and 63 generator/sectional/conical geometry tests.
- Tests compare zero-taper residuals for Euler, laminar, SA and QCR at 0 and 30 degrees of sweep on the actual generated three-block C-grid.
- Four independent SA solves converge (native 2D/sectional at zero sweep, legacy 2.5D/sectional at 30 degrees). Both pairs pass on coefficients, wall distributions and fields across all blocks.
- Maximum coefficient difference: 1.094e-12; maximum normalized primitive-field difference: 5.243e-10.
- Finite taper .2, LE sweep 30 deg, M=.5, alpha=2 deg, Re=1e6: 23216 independent cells, 2000 iterations, 10.028 residual orders, reason `converged`.
- Upper/lower wall profiles at five stations each are extracted successfully. Native centres below the estimated edge range from 17 to 41; their first-centre y+ ranges from 0.137 to 0.312. These counts and wall units alone do not establish profile convergence.

See [SA limits](limits_sa/summary.md), [finite-taper metadata](finite_taper_result.json), [finite-taper profiles](finite_taper_profiles/boundary_layer.json) and [all planned grid checks](grid_checks.json). Complete physical arrays remain under `results/sectional_c_blunt*`.

## Mesh and remaining qualification

![Main block (blue), base (orange), wake (green); sparse displayed lines](mesh.png)

On the finest source, normal-plane cell angles remain above 50.35 degrees and all cell areas are positive. Interface node mismatch is below 5e-16. However, the largest interface width ratio is about 63; the normal-only fine case reaches about 126 because base and wake blocks stay fixed. The C topology does not remove the need to study these transitions.

Only the medium finite-taper case has been solved here. The full C-grid global/normal/outer sensitivity campaign remains incomplete; [its partial report](mesh/summary.md) explicitly lists the missing cases. No C-grid convergence order or mesh-independence claim follows from these checks. Resolution of the blunt-base layer, wake thickness and wake length also require separate assessment. Historical O-grid convergence results are not transferred to this baseline.

## Reproduction

```bash
JAX_PLATFORMS=cuda python -m validation.sectional_qualification --study limits --physics sa --out results/sectional_c_blunt_limits_sa
JAX_PLATFORMS=cuda python -m validation.sectional_qualification --study mesh --only mesh_sa_l1 --out results/sectional_c_blunt
MPLCONFIGDIR=/tmp/jaxrans-mpl JAX_PLATFORMS=cpu python -m validation.sectional_boundary_layer results/sectional_c_blunt/mesh_sa_l1 --out reports/sectional_c_blunt/finite_taper_profiles
```

The selected mesh run exits nonzero because the remaining planned solves are missing; the saved medium solve itself is converged. CUDA/float64 was used, with `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` on this WSL2 installation. For existing matching outputs add `--resume`. For the full protocol see [the guide](../../docs/sectional-qualification.md).
