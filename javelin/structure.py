"""
=========
structure
=========
"""


import numpy as np
from pandas import DataFrame
from javelin.unitcell import UnitCell
from javelin.utils import (get_atomic_number_symbol, is_structure, get_unitcell, get_positions,
                           get_atomic_numbers)


class Structure(object):
    """
    Structure class

    :param symbols: atoms symbols to initialize structure
    :type symbols: list
    :param numbers: atomic numbers to initialize structure
    :type numbers: list
    """
    def __init__(self, symbols=None, numbers=None, unitcell=1, ncells=None,
                 positions=None, rotations=False, translations=False, magnetic_moments=False):

        # Check if initialising from another structure
        if symbols is not None and is_structure(symbols):
            unitcell = get_unitcell(symbols)
            positions = get_positions(symbols)
            numbers = get_atomic_numbers(symbols)
            symbols = None

        if positions is not None:
            numberOfAtoms = len(positions)
        else:
            numberOfAtoms = 0

        if ncells is not None:
            ncells = np.asarray(ncells)

        if ncells is not None and positions is not None and ncells.prod() != numberOfAtoms:
            raise ValueError("Product of ncells values doesn't equal length of positions")

        if isinstance(unitcell, UnitCell):
            self.unitcell = unitcell
        else:
            self.unitcell = UnitCell(unitcell)

        miindex = get_miindex(numberOfAtoms, ncells)

        self.atoms = DataFrame(index=miindex,
                               columns=['Z', 'symbol',
                                        'x', 'y', 'z'])

        if numbers is not None or symbols is not None:
            self.atoms.Z, self.atoms.symbol = get_atomic_number_symbol(Z=numbers, symbol=symbols)

        positions = np.asarray(positions)
        self.atoms[['x', 'y', 'z']] = positions

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

        if magnetic_moments:
            self.magmons = DataFrame(index=miindex,
                                     columns=['spinx', 'spiny', 'spinz'])
        else:
            self.magmons = None

    @property
    def number_of_atoms(self):
        return len(self.atoms)

    @property
    def element(self):
        return self.atoms.symbol.values

    @property
    def xyz(self):
        return self.atoms[['x', 'y', 'z']].values

    @property
    def x(self):
        return self.atoms.x.values

    @property
    def y(self):
        return self.atoms.y.values

    @property
    def z(self):
        return self.atoms.z.values

    @property
    def xyz_cartn(self):
        return self.unitcell.cartesian(self.atoms[['x', 'y', 'z']].values +
                                       np.asarray([self.atoms.index.get_level_values(0).values,
                                                   self.atoms.index.get_level_values(1).values,
                                                   self.atoms.index.get_level_values(2).values]).T)

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
        return (self.atoms[['x', 'y', 'z']].values +
                np.asarray([self.atoms.index.get_level_values(0).values,
                            self.atoms.index.get_level_values(1).values,
                            self.atoms.index.get_level_values(2).values]).T)

    def get_positions(self):
        return self.xyz_cartn

    def get_magnetic_moments(self):
        return self.magmons.values

    def add_atom(self, i=0, j=0, k=0, site=0, Z=None, symbol=None, position=None):
        Z, symbol = get_atomic_number_symbol([Z], [symbol])
        if position is None:
            raise ValueError("position not provided")

        self.atoms.loc[i, j, k, site] = [Z[0], symbol[0],
                                         position[0], position[1], position[2]]

        if self.rotations is not None:
            self.rotations[i, j, k] = [1, 0, 0, 0]

        if self.translations is not None:
            self.translations[i, j, k] = [0, 0, 0]

    def rattle(self, scale=0.001, seed=None):
        """Randomly move all atoms by a normal distbution with a standard
        deviation given by scale.

        :param scale: standard deviation
        :type scale: float
        :param scale: seed for random number generator
        :type scale: int
        """
        rs = np.random.RandomState(seed)
        self.atoms[['x', 'y', 'z']] += rs.normal(scale=scale, size=self.xyz.shape)

    def repeat(self, rep):
        """Repeat the cells a number of time along each dimension

        *rep* argument should be either three value like *(1,2,3)* or
        a single value *r* equivalent to *(r,r,r)*."""

        if isinstance(rep, int):
            rep = np.array((rep, rep, rep, 1))
        else:
            rep = np.append(rep, 1)

        ncells = np.array(self.atoms.index.max()) + 1

        x = np.tile(np.reshape(self.x, ncells), rep).flatten()
        y = np.tile(np.reshape(self.y, ncells), rep).flatten()
        z = np.tile(np.reshape(self.z, ncells), rep).flatten()
        Z = np.tile(np.reshape(self.get_atomic_numbers(), ncells), rep).flatten()

        miindex = get_miindex(0, ncells * rep)

        self.atoms = DataFrame(index=miindex,
                               columns=['Z', 'symbol',
                                        'x', 'y', 'z'])

        self.atoms.Z, self.atoms.symbol = get_atomic_number_symbol(Z=Z)

        self.atoms.x = x
        self.atoms.y = y
        self.atoms.z = z

    def reindex(self, ncells):
        self.atoms.set_index(get_miindex(ncells=ncells), inplace=True)


def axisAngle2Versor(x, y, z, angle, unit='degrees'):
    norm = np.linalg.norm([x, y, z])

    if norm == 0:
        raise ValueError("Vector must have non-zero length")

    x /= norm
    y /= norm
    z /= norm

    if unit == 'degrees':
        angle = np.deg2rad(angle)

    sw = np.sin(angle/2)
    return [np.cos(angle/2), x*sw, y*sw, z*sw]


def get_rotation_matrix(l, m, n, theta, unit='degrees'):
    w, x, y, z = axisAngle2Versor(l, m, n, theta, unit=unit)
    return get_rotation_matrix_from_versor(w, x, y, z)


def get_rotation_matrix_from_versor(w, x, y, z):
    return np.matrix([[1-2*y**2-2*z**2, 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*x**2-2*z**2, 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*x**2-2*y**2]]).T.A


def get_miindex(l=0, ncells=None):
    from pandas import MultiIndex

    if ncells is None:
        if l == 0:
            miindex = MultiIndex(levels=[[], [], [], []],
                                 labels=[[], [], [], []],
                                 names=['i', 'j', 'k', 'site'])
        else:
            miindex = MultiIndex.from_product([[0], [0], [0], range(l)],
                                              names=['i', 'j', 'k', 'site'])
    else:
        miindex = MultiIndex.from_product([range(ncells[0]),
                                           range(ncells[1]),
                                           range(ncells[2]),
                                           range(ncells[3])],
                                          names=['i', 'j', 'k', 'site'])

    return miindex
