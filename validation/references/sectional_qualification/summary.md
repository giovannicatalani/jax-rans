> **Historical O-grid results — superseded baseline.** These measurements
> predate the switch to the blunt-TE NACA 0012 three-block C-grid. They do not
> qualify the current examples or establish C-grid convergence. Numerical
> records below are preserved; commands/paths describe the earlier campaign.

# Sectional qualification: converged limits and mesh sensitivity

Numerical verification on NACA 0012; no finite-wing aerodynamic validation.

## Converged limits

Each pair starts independently from freestream. Stopping uses the residual, not force settling.

| model / sweep | result | max coefficient difference | Cp Linf | Cf Linf | field relative Linf |
|---|---|---|---|---|---|
| euler_0 | PASS | 3.004e-10 | 2.084e-09 | 0.000e+00 | 1.425e-09 |
| euler_30 | PASS | 1.835e-09 | 1.455e-08 | 0.000e+00 | 1.066e-08 |
| laminar_0 | PASS | 7.396e-11 | 8.859e-11 | 1.510e-11 | 2.407e-09 |
| laminar_30 | PASS | 8.196e-11 | 9.705e-11 | 1.670e-11 | 2.948e-09 |
| sa_0 | PASS | 5.211e-10 | 3.234e-09 | 8.908e-11 | 1.209e-07 |
| sa_30 | PASS | 2.033e-10 | 1.340e-09 | 3.275e-11 | 5.569e-08 |
| qcr_0 | PASS | 5.125e-10 | 3.382e-09 | 9.274e-11 | 1.263e-07 |
| qcr_30 | PASS | 2.046e-10 | 1.452e-09 | 3.543e-11 | 6.028e-08 |

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

All requested solves converged: **True**.

| case | section cells | CL | CD | friction CD | CM | y+ p95 | outer distance | orders |
|---|---|---|---|---|---|---|---|---|
| mesh_sa_l3 | 32 x 24 | 0.2856413207 | 0.0129985703 | 0.0085107922 | -0.0047600308 | 1.668 | 45.14 | 10.88 |
| mesh_sa_l2 | 64 x 48 | 0.2768467241 | 0.0083203173 | 0.0086180806 | -0.0015799900 | 0.762 | 45.14 | 10.93 |
| mesh_sa_l1 | 128 x 96 | 0.2667399913 | 0.0073820786 | 0.0085997292 | 0.0005806099 | 0.358 | 45.14 | 10.37 |
| mesh_sa_l0 | 256 x 192 | 0.2601250766 | 0.0071188388 | 0.0085793907 | 0.0018732220 | 0.173 | 45.14 | 10.22 |

| change | delta CL | delta CD | relative CL | relative CD | Cp L2 | Cf L2 |
|---|---|---|---|---|---|---|
| mesh_sa_l3 → mesh_sa_l2 | -8.795e-03 | -4.678e-03 | 3.1767% | 56.2269% | 2.650e-02 | 3.317e-04 |
| mesh_sa_l2 → mesh_sa_l1 | -1.011e-02 | -9.382e-04 | 3.7890% | 12.7097% | 8.766e-03 | 1.581e-04 |
| mesh_sa_l1 → mesh_sa_l0 | -6.615e-03 | -2.632e-04 | 2.5430% | 3.6978% | 4.445e-03 | 8.993e-05 |

Richardson fits from the three finest levels (only monotone, shrinking differences are extrapolated):

| coefficient | observed order | extrapolated value |
|---|---|---|
| cl | 0.612 | 0.2475937545 |
| cd | 1.834 | 0.0070161791 |
| cm | 0.741 | 0.0037981868 |
| cd_friction | unavailable | no extrapolation |

A fit alone does not establish an asymptotic range or certify mesh independence.

## Wall study

All requested solves converged: **True**.

| case | section cells | CL | CD | friction CD | CM | y+ p95 | outer distance | orders |
|---|---|---|---|---|---|---|---|---|
| mesh_sa_l2 | 64 x 48 | 0.2768467241 | 0.0083203173 | 0.0086180806 | -0.0015799900 | 0.762 | 45.14 | 10.93 |
| wall_sa_0.5 | 64 x 48 | 0.2766475062 | 0.0083303569 | 0.0086111358 | -0.0015712141 | 0.384 | 45.14 | 10.49 |
| wall_sa_0.25 | 64 x 48 | 0.2764455176 | 0.0083484517 | 0.0086085719 | -0.0015608606 | 0.193 | 45.14 | 10.63 |

| change | delta CL | delta CD | relative CL | relative CD | Cp L2 | Cf L2 |
|---|---|---|---|---|---|---|
| mesh_sa_l2 → wall_sa_0.5 | -1.992e-04 | +1.004e-05 | 0.0720% | 0.1205% | 1.605e-04 | 5.120e-06 |
| wall_sa_0.5 → wall_sa_0.25 | -2.020e-04 | +1.809e-05 | 0.0731% | 0.2167% | 1.643e-04 | 3.079e-06 |

## Outer study

All requested solves converged: **True**.

| case | section cells | CL | CD | friction CD | CM | y+ p95 | outer distance | orders |
|---|---|---|---|---|---|---|---|---|
| mesh_sa_l2 | 64 x 48 | 0.2768467241 | 0.0083203173 | 0.0086180806 | -0.0015799900 | 0.762 | 45.14 | 10.93 |
| outer_sa_2 | 64 x 52 | 0.2780933280 | 0.0082605767 | 0.0086166745 | -0.0015840678 | 0.762 | 102.93 | 10.17 |
| outer_sa_4 | 64 x 56 | 0.2786413746 | 0.0082342098 | 0.0086160447 | -0.0015860309 | 0.762 | 225.39 | 10.09 |

| change | delta CL | delta CD | relative CL | relative CD | Cp L2 | Cf L2 |
|---|---|---|---|---|---|---|
| mesh_sa_l2 → outer_sa_2 | +1.247e-03 | -5.974e-05 | 0.4483% | 0.7232% | 9.536e-04 | 4.696e-06 |
| outer_sa_2 → outer_sa_4 | +5.480e-04 | -2.637e-05 | 0.1967% | 0.3202% | 4.193e-04 | 2.064e-06 |
