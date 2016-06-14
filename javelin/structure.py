import numpy as np
from pandas import DataFrame
from javelin.unitcell import UnitCell


class Structure(object):
    def __init__(self):
        self.unitcell = UnitCell()
        self.atoms = DataFrame(columns=['i', 'j', 'k', 'site',
                                        'Z', 'symbol',
                                        'rel_x', 'rel_y', 'rel_z',
                                        'x', 'y', 'z']).set_index(['i', 'j', 'k', 'site'])
        self.molecules = {}

    @property
    def number_of_atoms(self):
        return len(self.atoms)

    def get_atom_symbols(self):
        return self.atoms.symbol.unique()

    def get_atom_Zs(self):
        return self.atoms.Z.unique()

    def get_atom_count(self):
        return self.atoms.symbol.value_counts()

    def get_atomic_numbers(self):
        return self.atoms.Z.values

    def get_chemical_symbols(self):
        return self.atoms.symbol.values

    def get_scaled_positions(self):
        return np.array([self.atoms.x.values,
                         self.atoms.y.values,
                         self.atoms.z.values]).T

    def get_positions(self):
        return np.array([self.atoms.x.values * self.unitcell.a,
                         self.atoms.y.values * self.unitcell.b,
                         self.atoms.z.values * self.unitcell.c]).T

    def add_atom(self, i=0, j=0, k=0, site=0, Z=None, symbol='', position=None):
        Z, symbol = get_atomic_number_symbol(Z, symbol)
        if position is None:
            raise ValueError("position not provided")
        position_x = position[0] + i
        position_y = position[1] + j
        position_z = position[2] + k
        self.atoms.loc[i, j, k, site] = [Z, symbol,
                                         position[0], position[1], position[2],
                                         position_x, position_y, position_z]


def get_atomic_number_symbol(Z=None, symbol=''):
    import periodictable

    if symbol is '':
        if Z is None:
            raise ValueError("symbol and/or Z number not given")
        else:
            symbol = periodictable.elements[Z].symbol
    else:
        symbol = symbol.capitalize()
        z = periodictable.elements.symbol(symbol).number
        if Z is None:
            Z = z
        elif Z is not z:
            raise ValueError("symbol and Z don't match")
    return (Z, symbol)
