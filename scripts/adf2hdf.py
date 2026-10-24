"""Convert a CGNS file from the old ADF storage to HDF5, which the solver reads.

    python scripts/adf2hdf.py in_adf.cgns out_hdf5.cgns

The CGNS tools do this (``adf2hdf in out``, or ``cgnsconvert -h``, conda-forge
``cgns``); use them where you can. Where they cannot be installed, this
script calls the CGNS library that VTK bundles (``pip install vtk``):
``cg_open`` the file, ``cg_save_as`` HDF5. Tooling only, not a dependency of
the solver.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

CG_MODE_READ, CG_FILE_HDF5 = 0, 2


def main(source: str, target: str) -> None:
    import vtkmodules

    lib = ctypes.CDLL(str(Path(vtkmodules.__file__).parent / "libvtkcgns.so"))
    handle = ctypes.c_int()
    if lib.vtkcgns_cg_open(source.encode(), CG_MODE_READ, ctypes.byref(handle)):
        raise SystemExit(f"cannot open {source} as CGNS")
    status = lib.vtkcgns_cg_save_as(handle, target.encode(), CG_FILE_HDF5, 0)
    lib.vtkcgns_cg_close(handle)
    if status:
        raise SystemExit(f"cg_save_as failed ({status})")
    print(f"{source} -> {target} (HDF5)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
