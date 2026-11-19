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

import os as _os
from pathlib import Path as _Path

import jax as _jax

_jax.config.update("jax_enable_x64", True)

# Persistent XLA compilation cache: a resumed or repeated campaign reuses the
# executables of an earlier run instead of compiling them again (tens of
# seconds each). ``JAXRANS_COMPILATION_CACHE`` sets the directory, ``off``
# disables it; an explicit ``jax_compilation_cache_dir`` is left alone.
_cache = _os.environ.get("JAXRANS_COMPILATION_CACHE",
                         str(_Path.home() / ".cache" / "jaxrans" / "xla"))
if _cache.strip().lower() not in ("", "0", "off", "false", "none") and not _jax.config.jax_compilation_cache_dir:
    _jax.config.update("jax_compilation_cache_dir", _cache)
    _jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

from .case import Case, CaseError, load_case  # noqa: E402
from .simulation import Result, run  # noqa: E402
from .derivatives import Sensitivity, sensitivity  # noqa: E402

__all__ = ["Case", "CaseError", "Result", "Sensitivity", "load_case", "run", "sensitivity"]
__version__ = "0.2.0"
