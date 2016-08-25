import numpy as np
from pandas import DataFrame, MultiIndex
from javelin.unitcell import UnitCell


class Structure(object):
    def __init__(self, symbols=None, numbers=None, unitcell=None, ncells=None,
                 positions=None, cartn_positions=None,
                 rotations=False, translations=False):

        if isinstance(unitcell, UnitCell):
            self.unitcell = unitcell
        elif unitcell is None:
            self.unitcell = UnitCell()
        else:
            self.unitcell = UnitCell(unitcell)

        if ncells is not None and positions is not None and ncells.prod() != len(positions):
            raise ValueError("Product of ncells values equal length of positions")

        if ncells is None:
            if positions is None:
                miindex = MultiIndex(levels=[[], [], [], []],
                                     labels=[[], [], [], []],
                                     names=['i', 'j', 'k', 'site'])
            else:
                miindex = MultiIndex.from_product([0, 0, 0, range(len(positions))],
                                                  names=['i', 'j', 'k', 'site'])
        else:
            miindex = MultiIndex.from_product([range(ncells[0]),
                                               range(ncells[1]),
                                               range(ncells[2]),
                                               range(ncells[3])],
                                              names=['i', 'j', 'k', 'site'])

        self.atoms = DataFrame(index=miindex,
                               columns=['Z', 'symbol',
                                        'x', 'y', 'z',
                                        'cartn_x', 'cartn_y', 'cartn_z'])

        if rotations:
            self.rotations = DataFrame(index=miindex.droplevel(3),
                                       columns=['w', 'x', 'y', 'z'])
        else:
            self.rotations = None

        if translations:
            self.translations = DataFrame(index=miindex.droplevel(3),
                                          columns=['x', 'y', 'z'])
        else:
            self.translations = None

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

        cartn = self.unitcell.cartesian(position)

        self.atoms.loc[i, j, k, site] = [Z, symbol,
                                         position[0], position[1], position[2],
                                         cartn[0], cartn[1], cartn[2]]

        if self.rotations is not None:
            self.rotations[i, j, k] = [1, 0, 0, 0]

        if self.translations is not None:
            self.translations[i, j, k] = [0, 0, 0]

    def _recalculate_cartn(self):
        self.atoms[['cartn_x', 'cartn_y', 'cartn_z']] = self.unitcell.cartesian(
            self.atoms[['x', 'y', 'z']].values)


def get_atomic_number_symbol(Z=None, symbol=None):
    import periodictable

    if symbol is None:
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


def get_rotation_matrix(l, m, n, theta, unit='degrees'):

    norm = np.linalg.norm([l, m, n])

    if norm == 0:
        raise ValueError("Rotation vector must have non-zero length")

    l /= norm
    m /= norm
    n /= norm

    if unit == 'degrees':
        theta = np.deg2rad(theta)

    ct = np.cos(theta)
    st = np.sin(theta)
    return np.matrix([[l*l*(1-ct)+ct,   m*l*(1-ct)-n*st, n*l*(1-ct)+m*st],
                      [l*m*(1-ct)+n*st, m*m*(1-ct)+ct,   n*m*(1-ct)-l*st],
                      [l*n*(1-ct)-m*st, m*n*(1-ct)+l*st, n*n*(1-ct)+ct]]).T
