# Sectional qualification: converged limits and mesh sensitivity

Baseline: naca0012_blunt_c_grid_v1. Numerical verification; no finite-wing validation.

## Converged limits

Each pair starts independently from freestream. Stopping uses the residual, not force settling.

| model / sweep | result | max coefficient difference | Cp Linf | Cf Linf | field relative Linf |
|---|---|---|---|---|---|

Tolerances (absolute for coefficients and wall distributions):

```json
{
  "coefficient_abs": 1e-06,
  "cp_linf": 1e-05,
  "cf_linf": 1e-06,
  "field_relative": 1e-05
}
```

The field comparison includes density, three velocities, pressure and the SA variable where present.

## Global study

All requested solves converged: **False**.

| case | section cells | CL | CD | friction CD | CM | y+ p95 | outer distance | orders |
|---|---|---|---|---|---|---|---|---|
| mesh_sa_l2 | missing | — | — | — | — | — | — | — |
| mesh_sa_l1 | 240 × 96; 4 × 12; 32 × 4 | 0.2613628126 | 0.0071969917 | 0.0085748250 | 0.0016518347 | 0.296 | 90.34 | 10.03 |
| mesh_sa_l0 | missing | — | — | — | — | — | — | — |

| change | delta CL | delta CD | relative CL | relative CD | Cp L2 | Cf L2 |
|---|---|---|---|---|---|---|

## Normal study

All requested solves converged: **False**.

| case | section cells | CL | CD | friction CD | CM | y+ p95 | outer distance | orders |
|---|---|---|---|---|---|---|---|---|
| normal_sa_s4 | missing | — | — | — | — | — | — | — |
| mesh_sa_l1 | 240 × 96; 4 × 12; 32 × 4 | 0.2613628126 | 0.0071969917 | 0.0085748250 | 0.0016518347 | 0.296 | 90.34 | 10.03 |
| normal_sa_s1 | missing | — | — | — | — | — | — | — |

| change | delta CL | delta CD | relative CL | relative CD | Cp L2 | Cf L2 |
|---|---|---|---|---|---|---|

## Outer study

All requested solves converged: **False**.

| case | section cells | CL | CD | friction CD | CM | y+ p95 | outer distance | orders |
|---|---|---|---|---|---|---|---|---|
| mesh_sa_l1 | 240 × 96; 4 × 12; 32 × 4 | 0.2613628126 | 0.0071969917 | 0.0085748250 | 0.0016518347 | 0.296 | 90.34 | 10.03 |
| outer_sa_2 | missing | — | — | — | — | — | — | — |
| outer_sa_4 | missing | — | — | — | — | — | — | — |

| change | delta CL | delta CD | relative CL | relative CD | Cp L2 | Cf L2 |
|---|---|---|---|---|---|---|

## Incomplete cases

- mesh_sa_l2
- mesh_sa_l0
- normal_sa_s4
- normal_sa_s1
- outer_sa_2
- outer_sa_4
