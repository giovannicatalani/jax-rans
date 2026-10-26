# JAX-RANS

Differentiable compressible RANS in JAX: Euler, laminar Navier-Stokes and
Spalart-Allmaras RANS on structured grids, in 2D and 3D, on CPU or GPU (one
or several), with exact derivatives of the converged forces. Grids are
generated round a single airfoil or a slat, main element and flap from their
coordinates, or read from Plot3D and CGNS files with any number of blocks. A
case is two JSON files.

Try a transonic airfoil and its derivatives in the browser:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/giovannicatalani/jax-rans/blob/main/notebooks/quickstart.ipynb)

```bash
pip install jax-rans[plot,cgns]
jaxrans my_case --check     # build and draw the grid
jaxrans my_case             # solve
python -m jaxrans.polar my_case --alpha -2 10 1
```

- [Running a case](guide.md)
- [Your own airfoil](own-airfoil.md)
- [A multi-element airfoil](multi-element.md)
- [A 3D wing](wing-3d.md)
- [Derivatives](gradients.md): dCL/dalpha and friends from a converged run
- [Case reference](case_reference.md)
- [Equations and method](theory.md)
- [Developer guide](developer.md) and [design notes](dev/notes.md)

Licensed under the GNU GPL v3.
