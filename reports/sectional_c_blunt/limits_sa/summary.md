# Sectional qualification: converged limits and mesh sensitivity

Baseline: naca0012_blunt_c_grid_v1. Numerical verification; no finite-wing validation.

## Converged limits

Each pair starts independently from freestream. Stopping uses the residual, not force settling.

| model / sweep | result | max coefficient difference | Cp Linf | Cf Linf | field relative Linf |
|---|---|---|---|---|---|
| sa_0 | PASS | 7.332e-13 | 1.184e-12 | 1.954e-13 | 4.329e-10 |
| sa_30 | PASS | 1.094e-12 | 1.323e-12 | 2.786e-13 | 5.243e-10 |

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
