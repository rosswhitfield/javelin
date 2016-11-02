"""
ase
===

Theses functions read the legacy DISCUS stru file format in ASE Atoms.
"""

from __future__ import absolute_import
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell


def read_stru(filename):
    with open(filename) as f:
        lines = f.readlines()

    a = b = c = alpha = beta = gamma = 0

    reading_atom_list = False

    symbols = []
    positions = []

    for l in lines:
        line = l.replace(',', ' ').split()
        if not reading_atom_list:  # Wait for 'atoms' line before reading atoms
            if line[0] == 'cell':
                a, b, c, alpha, beta, gamma = [float(x) for x in line[1:7]]
                cell = cellpar_to_cell([a, b, c, alpha, beta, gamma])
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
