"""Theses functions read and write the legacy DISCUS stru file format."""

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
        if not reading_atom_list:
            if line[0] == 'cell':
                a, b, c, alpha, beta, gamma = line[1:7]
            if line[0] == 'atoms':
                if a == 0:
                    print("Cell not found, using a=b=c=1, alpha=beta=gamma=90")
                    a = b = c = 1
                    alpha = beta = gamma = 90
                reading_atom_list = True
        else:
            symbol, x, y, z = line[:4]
            symbol = symbol.lower().capitalize()
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
    return Atoms(symbols=symbols,
                 scaled_positions=positions,
                 cell=[a, b, c])  # TODO fix for non-cubic cells
