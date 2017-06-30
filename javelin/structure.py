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
    """The structure class is made up of a **unitcell** and a list of **atoms**

    Structure can be initialize using either another
    :class:`javelin.structure.Structure`, :class:`ase.Atoms` or
    :class:`diffpy.Structure.structure.Structure`. It is recommended
    you use :func:`javelin.structure.Structure.reindex` after
    initializing from a foreign type in order to get the correct
    unitcell structure type.

    :param symbols: atoms symbols to initialize structure
    :type symbols: list
    :param numbers: atomic numbers to initialize structure
    :type numbers: list
    :param unitcell: unitcell of structure, can be :class:`javelin.unitcell.UnitCell`
         or values used to initialize the UnitCell
    :type unitcell: :class:`javelin.unitcell.UnitCell`
    :param ncells: **ncells** has four components, (**i**, **j**, **k**, **n**)
        where **i**, **j**, **k** are the number of unitcell in each
        direction and **n** is the number of site positions in each
        unitcell. The product of **ncells** must equal the total number of
        atoms in the structure.
    :type ncells: list
    :param positions: array of atom coordinates
    :type positions: 3 x n array-like

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
            """Attribute containing the unitcell of the structure. Must be of type
            :class:`javelin.unitcell.UnitCell`

            """

        miindex = get_miindex(numberOfAtoms, ncells)

        self.atoms = DataFrame(index=miindex,
                               columns=['Z', 'symbol',
                                        'x', 'y', 'z'])
        """Attribute storing list of atom type and positions as a :class:`pandas.DataFrame`

        :example:

        >>> stru = Structure(symbols=['Na','Cl','Na'],positions=[[0,0,0],[0.5,0.5,0.5],[0,1,0]])
        >>> stru.atoms
                     Z symbol    x    y    z
        i j k site
        0 0 0 0     11     Na  0.0  0.0  0.0
              1     17     Cl  0.5  0.5  0.5
              2     11     Na  0.0  1.0  0.0

        """

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

    def __str__(self):
        return "{}({}, {})".format(self.__class__.__name__,
                                   self.get_chemical_formula(),
                                   self.unitcell)

    @property
    def number_of_atoms(self):
        """The total number of atoms in the structure

        :return: number of atoms in structure
        :rtype: int

        :example:

        >>> stru = Structure(symbols=['Na','Cl','Na'],positions=[[0,0,0],[0.5,0.5,0.5],[0,1,0]])
        >>> stru.number_of_atoms
        3
        """
        return len(self.atoms)

    @property
    def element(self):
        """Array of all elements in the stucture

        :return: array of element symbols
        :rtype: :class:`numpy.ndarray`

        :example:

        >>> stru = Structure(symbols=['Na','Cl','Na'],positions=[[0,0,0],[0.5,0.5,0.5],[0,1,0]])
        >>> stru.element
        array(['Na', 'Cl', 'Na'], dtype=object)
        """
        return self.atoms.symbol.values

    @property
    def xyz(self):
        """Array of all xyz positions in fractional lattice units of the atoms
        within the unitcell of the structure

        :return: 3 x n array of atom positions
        :rtype: :class:`numpy.ndarray`

        :example:

        >>> stru = Structure(symbols=['Na', 'Cl'], positions=[[0,0,0],[0.5,0.5,0.5]], unitcell=5.64)
        >>> stru.xyz
        array([[ 0. ,  0. ,  0. ],
               [ 0.5,  0.5,  0.5]])

        """
        return self.atoms[['x', 'y', 'z']].values

    @property
    def x(self):
        """Array of all x positions of in fractional lattice units of the atoms
        within the unitcell of the structure

        :return: array of atom x positions
        :rtype: :class:`numpy.ndarray`

        :example:

        >>> stru = Structure(symbols=['Na', 'Cl'], positions=[[0,0,0],[0.5,0.5,0.5]], unitcell=5.64)
        >>> stru.x
        array([ 0. ,  0.5])
        """
        return self.atoms.x.values

    @property
    def y(self):
        """Array of all y positions of in fractional lattice units of the atoms
        within the unitcell of the structure

        :return: array of atom y positions
        :rtype: :class:`numpy.ndarray`

        :example:

        >>> stru = Structure(symbols=['Na', 'Cl'], positions=[[0,0,0],[0.5,0.5,0.5]], unitcell=5.64)
        >>> stru.y
        array([ 0. ,  0.5])
        """
        return self.atoms.y.values

    @property
    def z(self):
        """Array of all z positions of in fractional lattice units of the atoms
        within the unitcell of the structure

        :return: array of atom z positions
        :rtype: :class:`numpy.ndarray`

        :example:

        >>> stru = Structure(symbols=['Na', 'Cl'], positions=[[0,0,0],[0.5,0.5,0.5]], unitcell=5.64)
        >>> stru.z
        array([ 0. ,  0.5])
        """
        return self.atoms.z.values

    @property
    def xyz_cartn(self):
        """Array of all xyz positions in cartesian coordinates of the atoms in
        the structure

        :return: 3 x n array of atom positions
        :rtype: :class:`numpy.ndarray`

        :example:

        >>> stru = Structure(symbols=['Na', 'Cl'], positions=[[0,0,0],[0.5,0.5,0.5]], unitcell=5.64)
        >>> stru.xyz_cartn
        array([[ 0.  ,  0.  ,  0.  ],
               [ 2.82,  2.82,  2.82]])

        """
        return self.unitcell.cartesian(self.get_scaled_positions())

    def get_atom_symbols(self):
        """Get a list of unique atom symbols in structure

        :return: array of atom symbols
        :rtype: :class:`numpy.ndarray`

        :example:

        >>> stru = Structure(symbols=['Na','Cl','Na'],positions=[[0,0,0],[0.5,0.5,0.5],[0,1,0]])
        >>> stru.get_atom_symbols()
        array(['Na', 'Cl'], dtype=object)
        """
        return self.atoms.symbol.unique()

    def get_atom_Zs(self):
        """Get a list of unique atomic number in structure

        :return: array of Zs
        :rtype: :class:`numpy.ndarray`

        :example:

        >>> stru = Structure(symbols=['Na','Cl','Na'],positions=[[0,0,0],[0.5,0.5,0.5],[0,1,0]])
        >>> stru.get_atom_Zs()
        array([11, 17])
        """
        return self.atoms.Z.unique()

    def get_atom_count(self):
        """Returns a count of each different type of atom in the structure

        :return: series of atom count
        :rtype: :class:`pandas.Series`

        :example:

        >>> stru = Structure(symbols=['Na','Na'],positions=[[0,0,0],[0.5,0.5,0.5]])
        >>> stru.get_atom_count()
        Na    2
        Name: symbol, dtype: int64
        """
        return self.atoms.symbol.value_counts()

    def get_atomic_numbers(self):
        """Array of all atomic numbers in the stucture

        :return: array of atomic numbers
        :rtype: :class:`numpy.ndarray`

        :example:

        >>> stru = Structure(symbols=['Na','Cl','Na'],positions=[[0,0,0],[0.5,0.5,0.5],[0,1,0]])
        >>> stru.get_atomic_numbers()
        array([11, 17, 11])
        """
        return self.atoms.Z.values

    def get_chemical_symbols(self):
        """Same as :obj:`javelin.structure.Structure.element`
        """
        return self.element

    def get_chemical_formula(self):
        """Returns the chemical formula of the structure

        :return: chemical formula
        :rtype: str

        :example:

        >>> stru = Structure(symbols=['Na','Cl','Na'],positions=[[0,0,0],[0.5,0.5,0.5],[0,1,0]])
        >>> stru.get_chemical_formula()
        'Na2Cl1'
        """
        return (self.get_atom_count().index.values+self.get_atom_count().values.astype('str')).sum()

    def get_scaled_positions(self):
        """Array of all xyz positions in fractional lattice units of the atoms
        in the structure

        :return: 3 x n array of atom positions
        :rtype: :class:`numpy.ndarray`

        :example:

        >>> stru = Structure(symbols=['Na', 'Cl'], positions=[[0,0,0],[0.5,0.5,0.5]], unitcell=5.64)
        >>> stru.get_scaled_positions()
        array([[ 0. ,  0. ,  0. ],
               [ 0.5,  0.5,  0.5]])

        """
        return (self.atoms[['x', 'y', 'z']].values +
                np.asarray([self.atoms.index.get_level_values(0).values,
                            self.atoms.index.get_level_values(1).values,
                            self.atoms.index.get_level_values(2).values]).T)

    def get_positions(self):
        """Same as :obj:`javelin.structure.Structure.xyz_cartn`
        """
        return self.xyz_cartn

    def get_magnetic_moments(self):
        return self.magmons.values

    def add_atom(self, i=0, j=0, k=0, site=0, Z=None, symbol=None, position=None):
        """Adds a single atom to the structure. It the atom exist as provided
        **i**, **j**, **k** and **site** it will be replaced.

        :param i: unitcell index
        :type i: int
        :param j: unitcell index
        :type j: int
        :param k: unitcell index
        :type k: int
        :param site: site index
        :type site: int
        :param Z: atomic number
        :type Z: int
        :param symbol: chemical symbol
        :type symbol: int
        :param position: position within the unitcell
        :type position: vector


        >>> stru=Structure(unitcell=5.64)
        >>> stru.atoms
        Empty DataFrame
        Columns: [Z, symbol, x, y, z]
        Index: []
        >>> stru.add_atom(Z=12, position=[0,0,0])
        >>> stru.atoms # doctest: +NORMALIZE_WHITESPACE
                     Z symbol  x  y  z
        i j k site
        0 0 0 0     12     Mg  0  0  0
        >>> stru.add_atom(Z=13, position=[0.5,0,0], i=1)
        >>> stru.atoms # doctest: +NORMALIZE_WHITESPACE
                     Z symbol    x  y  z
        i j k site
        0 0 0 0     12     Mg    0  0  0
        1 0 0 0     13     Al  0.5  0  0

        """
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
        :param seed: seed for random number generator
        :type seed: int

        :example:

        >>> stru = Structure(symbols=['Na', 'Cl'], positions=[[0,0,0],[0.5,0.5,0.5]], unitcell=5.64)
        >>> print(stru.xyz)
        [[ 0.   0.   0. ]
         [ 0.5  0.5  0.5]]
        >>> stru.rattle(seed=42)
        >>> print(stru.xyz)
        [[  4.96714153e-04  -1.38264301e-04   6.47688538e-04]
         [  5.01523030e-01   4.99765847e-01   4.99765863e-01]]
        """
        rs = np.random.RandomState(seed)
        self.atoms[['x', 'y', 'z']] += rs.normal(scale=scale, size=self.xyz.shape)

    def repeat(self, rep):
        """Repeat the cells a number of time along each dimension

        *rep* argument should be either three value like *(1,2,3)* or
        a single value *r* equivalent to *(r,r,r)*.

        :param rep: repeating rate
        :type rep: 1 or 3 ints

        :examples:

        >>> stru = Structure(symbols=['Na', 'Cl'], positions=[[0,0,0],[0.5,0.5,0.5]], unitcell=5.64)
        >>> print(stru.element)
        ['Na' 'Cl']
        >>> print(stru.xyz_cartn)
        [[ 0.    0.    0.  ]
         [ 2.82  2.82  2.82]]
        >>> stru.repeat((2,1,1))
        >>> print([str(e) for e in stru.element])
        ['Na', 'Cl', 'Na', 'Cl']
        >>> print(stru.xyz_cartn)
        [[  0.00000000e+00   0.00000000e+00   0.00000000e+00]
         [  2.82000000e+00   2.82000000e+00   2.82000000e+00]
         [  5.64000000e+00   9.06981174e-16   9.06981174e-16]
         [  8.46000000e+00   2.82000000e+00   2.82000000e+00]]

        >>> stru = Structure(symbols=['Na'], positions=[[0,0,0]], unitcell=5.64)
        >>> print(stru.element)
        ['Na']
        >>> print(stru.xyz_cartn)
        [[ 0.  0.  0.]]
        >>> stru.repeat(2)
        >>> print([str(e) for e in stru.element])
        ['Na', 'Na', 'Na', 'Na', 'Na', 'Na', 'Na', 'Na']
        >>> print(stru.xyz_cartn)
        [[  0.00000000e+00   0.00000000e+00   0.00000000e+00]
         [  0.00000000e+00   0.00000000e+00   5.64000000e+00]
         [  0.00000000e+00   5.64000000e+00   3.45350397e-16]
         [  0.00000000e+00   5.64000000e+00   5.64000000e+00]
         [  5.64000000e+00   9.06981174e-16   9.06981174e-16]
         [  5.64000000e+00   9.06981174e-16   5.64000000e+00]
         [  5.64000000e+00   5.64000000e+00   1.25233157e-15]
         [  5.64000000e+00   5.64000000e+00   5.64000000e+00]]
        """

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
        """This will reindex the list of atoms into the unitcell framework of this structure

        **ncells** has four components, (**i**, **j**, **k**, **n**)
        where **i**, **j**, **k** are the number of unitcell in each
        direction and **n** is the number of site positions in each
        unitcell. The product of **ncells** must equal the total number of
        atoms in the structure.

        :example:

        >>> stru = Structure(symbols=['Na', 'Cl'], positions=[[0,0,0],[0.5,0.5,0.5]], unitcell=5.64)
        >>> stru.atoms # doctest: +NORMALIZE_WHITESPACE
                     Z symbol    x    y    z
        i j k site
        0 0 0 0     11     Na  0.0  0.0  0.0
              1     17     Cl  0.5  0.5  0.5
        >>> stru.reindex([2,1,1,1])
        >>> stru.atoms # doctest: +NORMALIZE_WHITESPACE
                     Z symbol    x    y    z
        i j k site
        0 0 0 0     11     Na  0.0  0.0  0.0
        1 0 0 0     17     Cl  0.5  0.5  0.5

        """
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
