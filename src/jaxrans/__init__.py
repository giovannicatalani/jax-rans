"""Differentiable compressible RANS in JAX.

Cell-centred finite-volume solver for steady flow on structured single- and
multi-block grids, 2D and 3D: compressible Euler, laminar Navier-Stokes and
Spalart-Allmaras RANS, with derivatives of the converged forces.

    from jaxrans import load_case, run, sensitivity
    result = run(load_case("examples/naca0012_laminar"))
    result.write()
    sensitivity(result, outputs=("cl", "cd"), parameters=("alpha",))

or, from the shell, ``python -m jaxrans examples/naca0012_euler``.

Float64 support stays enabled for geometry and verification. Each case selects
its field dtype independently; this global flag does not force FP64 arrays.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

from .case import Case, CaseError, load_case  # noqa: E402
from .simulation import Result, run  # noqa: E402
from .derivatives import Sensitivity, sensitivity  # noqa: E402

__all__ = ["Case", "CaseError", "Result", "Sensitivity", "load_case", "run", "sensitivity"]
__version__ = "0.2.0"
