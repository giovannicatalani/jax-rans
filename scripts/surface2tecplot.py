#!/usr/bin/env python3
"""Convertit un fichier *_surface.dat de jax-rans en fichier Tecplot ASCII structuré."""
import sys
from pathlib import Path
import numpy as np

src = Path(sys.argv[1])
dst = src.with_name(src.stem + "_tecplot.dat")
data = np.loadtxt(src, comments="#", ndmin=2)

with dst.open("w") as f:
    f.write(f'TITLE = "{src.stem}"\n')
    f.write('VARIABLES = "X" "Y" "Z" "Cp" "Cf_x" "Cf_y" "Cf_z" "Cf_mag"\n')
    for b in np.unique(data[:, 0]).astype(int):
        sub = data[data[:, 0] == b]
        ui, uk = np.unique(sub[:, 1]), np.unique(sub[:, 2])
        ni, nj = len(ui), len(uk)
        # Trie par (i, k) puis transpose (nj, ni) pour que I varie le plus vite dans Tecplot
        order = np.lexsort((sub[:, 2], sub[:, 1]))
        vals = sub[order, 3:10]
        cf_mag = np.linalg.norm(vals[:, 4:7], axis=1, keepdims=True)
        vals = np.hstack([vals, cf_mag])
        if len(vals) == ni * nj:
            vals = vals.reshape(ni, nj, 8).transpose(1, 0, 2).reshape(-1, 8)
            f.write(f'ZONE T="wall_block{b}", I={ni}, J={nj}, K=1, ZONETYPE=ORDERED, DATAPACKING=POINT\n')
        else:
            f.write(f'ZONE T="wall_block{b}", I={len(vals)}, J=1, K=1, ZONETYPE=ORDERED, DATAPACKING=POINT\n')
        np.savetxt(f, vals, fmt="%.12e")

print(f"Écrit : {dst}")
