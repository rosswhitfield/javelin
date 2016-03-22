"""Theses functions read the legacy DISCUS stru file format in ASE Atoms."""

from __future__ import absolute_import
import numpy as np
from ase.atoms import Atoms


def read_stru(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()

    a = b = c = alpha = beta = gamma = 0

    reading_atom_list = False

    symbols = []
    positions = []

    for l in lines:
        line = l.replace(',', ' ').split()
        if not reading_atom_list:  # Wait for 'atoms' line before reading atoms
            if line[0] == 'cell':
                a, b, c, alpha, beta, gamma = [float(x) for x in line[1:7]]
                if alpha == 90 and beta == 90 and gamma == 90:  # orthorhombic
                    cell = [a, b, c]
                else:
                    deg2rad = np.pi / 180
                    alpha *= deg2rad
                    beta *= deg2rad
                    gamma *= deg2rad
                    # Convert to unit cell vectors
                    a_vec = [a, 0, 0]
                    b_vec = [b * np.cos(gamma), b * np.sin(gamma), 0]
                    cy_scale = np.cos(alpha) * (1 - np.cos(beta)) / np.sin(gamma)
                    c_vec = [c * np.cos(beta),
                             c * cy_scale,
                             c * np.sqrt(np.sin(beta)**2 - cy_scale**2)]
                    cell = np.round([a_vec, b_vec, c_vec], 14)
            if line[0] == 'atoms':
                if a == 0:
                    print("Cell not found, using a=b=c=1, alpha=beta=gamma=90")
                    cell = [1, 1, 1]
                reading_atom_list = True
        else:
            symbol, x, y, z = line[:4]
            symbol = symbol.capitalize()
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])

    # Return ASE Atoms object
    return Atoms(symbols=symbols, scaled_positions=positions, cell=cell)
