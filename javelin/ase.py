"""Theses functions read the legacy DISCUS stru file format in ASE Atoms."""

from __future__ import absolute_import
from ase.atoms import Atoms
from javelin.utils import unit_cell_to_vectors


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
                cell = unit_cell_to_vectors(a, b, c, alpha, beta, gamma)
            if line[0] == 'atoms':
                if a == 0:
                    print("Cell not found")
                    cell = None
                reading_atom_list = True
        else:
            symbol, x, y, z = line[:4]
            symbol = symbol.capitalize()
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])

    # Return ASE Atoms object
    return Atoms(symbols=symbols, scaled_positions=positions, cell=cell)
