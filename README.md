# JAX-RANS

**Reynolds Averaged Navier Stokes solver in Jax.** Euler, laminar Navier-Stokes and
Spalart-Allmaras RANS on structured multi-block grids, 2D and 3D, in double
precision on GPUs (one or several), with exact derivatives of the converged forces.

[![tests](https://github.com/giovannicatalani/jax-rans/actions/workflows/tests.yml/badge.svg)](https://github.com/giovannicatalani/jax-rans/actions/workflows/tests.yml)
[![docs](https://github.com/giovannicatalani/jax-rans/actions/workflows/docs.yml/badge.svg)](https://giovannicatalani.github.io/jax-rans/)
[![PyPI](https://img.shields.io/pypi/v/jax-rans)](https://pypi.org/project/jax-rans/)
[![licence: GPL v3](https://img.shields.io/badge/licence-GPL%20v3-blue)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/giovannicatalani/jax-rans/blob/main/notebooks/quickstart.ipynb)

<p align="center">
  <img src="docs/figures/readme_m6.png" alt="ONERA M6 wing, wall pressure and tip vortex" width="500">
  <br>
  <em>ONERA M6 wing, SA RANS, M 0.84, 3.06 deg, Re 14.6 million (NASA TMR): 1.08 million cells, 57 minutes on one NVIDIA A100.</em>
</p>


## Quickstart

```bash
pip install "jax-rans[plot] @ git+https://github.com/giovannicatalani/jax-rans"
jaxrans examples/naca0012_euler            # solve; results/naca0012_euler/
python -m jaxrans.plot results/naca0012_euler
```

For a GPU, install JAX with CUDA first
([instructions](https://docs.jax.dev/en/latest/installation.html);
[if it crashes](docs/install-troubleshooting.md)). A case is
two JSON files, the physics and the numerics; from Python:

```python
from jaxrans import load_case, run, sensitivity

result = run(load_case("examples/rae2822_sensitivity"))
print(result.forces.summary())
print(sensitivity(result, outputs=("cl", "cd"), parameters=("alpha", "mach")).summary())
```

Or try it in the browser, a transonic airfoil and its derivatives on a free
GPU: [quickstart notebook](notebooks/quickstart.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/giovannicatalani/jax-rans/blob/main/notebooks/quickstart.ipynb).


## Examples and Validation cases

Example cases are in [examples/](examples/README.md).

Validation cases can be found in the validation folder.

How to rerun them, with their sources: [validation/](validation/README.md).

## What it does

- **Physics**: compressible Euler, laminar Navier-Stokes, Spalart-Allmaras
  (SA-noft2, SA-neg, QCR2000); cell-centred finite volumes, HLLC or Roe
  fluxes, MUSCL reconstruction (minmod, van Albada, smooth van Albada or none).
- **Grids**: generated O- and C-grids round any airfoil from its coordinate
  file; generated slat-main-flap sections; any point-matched multi-block
  Plot3D or CGNS grid, 2D or 3D, with interfaces and patches found from the
  geometry.
- **Solvers**: pseudo-transient Newton-Krylov with line-implicit
  preconditioning, or explicit Runge-Kutta; grid sequencing; a polar's angles
  batched on the GPU; [batched profiles with alpha continuation](docs/guide.md#several-profiles-with-continuation);
  one large case split over several GPUs.
- **Derivatives**: dCL, dCD and dCM with respect to the angle of attack and
  the Mach number, exact for the discrete solution, one linear solve each
  ([how](docs/gradients.md)).Still quite experimental for general inputs.


## Derivatives

```python
from jaxrans import sensitivity

s = sensitivity(result, outputs=("cl", "cd"), parameters=("alpha", "mach"))
s.derivatives["cl"]["alpha"]               # per degree
```
([docs/gradients.md](docs/gradients.md)).

## Existing CFD solvers and differences with them

[JAX-CFD](https://github.com/google/jax-cfd) solves incompressible flow on
Cartesian grids, for machine learning research;
[JAX-Fluids](https://github.com/tumaer/JAXFLUIDS) compressible flow on
Cartesian grids with immersed boundaries, for shocks and two-phase flow.
JAX-RANS is for aerodynamics on body-fitted structured grids: wall-resolved
RANS round airfoils and wings, the grids an aerodynamicist already has.
[SU2](https://su2code.github.io) is the mature general-purpose code (C++,
unstructured, a discrete adjoint for shape optimization); JAX-RANS is
smaller, Python throughout, and differentiable through JAX.

## Documentation

[giovannicatalani.github.io/jax-rans](https://giovannicatalani.github.io/jax-rans/):
running a case, meshing and simulating your own airfoil, simulating multi-element sections, 3D wings,
derivatives, including the equations and method.


## Citing, licence, contributing

If you use JAX-RANS, please cite it ([CITATION.cff](CITATION.cff)). Licensed
under the GNU GPL v3. Issues and pull requests are welcome on
[GitHub](https://github.com/giovannicatalani/jax-rans/issues); the
[developer guide](docs/developer.md) explains the layout and the tests.
