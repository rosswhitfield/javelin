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


class Structure:
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

        positions = np.asarray(positions, dtype=np.float)
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

    def __len__(self):
        return self.number_of_atoms

    def __getitem__(self, key):
        from pandas import IndexSlice as idx
        output = Structure(unitcell=self.unitcell)
        output.atoms = self.atoms.loc[idx[key], slice(None)]
        return output

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
        """Array of all elements in the structure

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

    @property
    def info(self):
        """Dictionary of key-value pairs with additional information about the system.

        Not implemented, only for ASE compatibility.
        """
        return {}

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
        >>> print(stru.get_atom_Zs())
        [11 17]
        """
        return self.atoms.Z.unique()

    def update_atom_symbols(self):
        """This will update the atom symbol list from the Z numbers, this
        should be run if the Z numbers are modified directly
        """
        _, self.atoms.symbol = get_atomic_number_symbol(self.get_atomic_numbers())

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
        return self.atoms.symbol.value_counts().sort_index()

    def get_atomic_numbers(self):
        """Array of all atomic numbers in the structure

        :return: array of atomic numbers
        :rtype: :class:`numpy.ndarray`

        :example:

        >>> stru = Structure(symbols=['Na','Cl','Na'],positions=[[0,0,0],[0.5,0.5,0.5],[0,1,0]])
        >>> print(stru.get_atomic_numbers())
        [11 17 11]
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
        'Cl1Na2'
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

    def get_cell(self):
        return self.unitcell.Binv

    def get_celldisp(self):
        return np.zeros((3, 1))

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


        >>> stru=Structure(numbers=[12],positions=[[0.,0.,0.]],unitcell=5.64)
        >>> stru.atoms # doctest: +NORMALIZE_WHITESPACE
                     Z symbol    x    y    z
        i j k site
        0 0 0 0     12     Mg  0  0  0
        >>> stru.add_atom(Z=13, position=[0.,0.5,0.])
        >>> stru.atoms # doctest: +NORMALIZE_WHITESPACE
                     Z symbol    x    y    z
        i j k site
        0 0 0 0     13     Al  0...  0.5  0...
        >>> stru.add_atom(Z=13, position=[0.5,0.,0.], i=1)
        >>> stru.atoms # doctest: +NORMALIZE_WHITESPACE
                     Z symbol    x    y    z
        i j k site
        0 0 0 0     13     Al  0.0  0.5  0...
        1 0 0 0     13     Al  0.5  0.0  0...

        """
        Z, symbol = get_atomic_number_symbol(Z, symbol)
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

    def replace_atom(self, to_replace: int, value: int) -> None:
        """Replace all atoms in the structure that has Z=`to_replace` with
        Z=`value`. This uses :meth:`pandas.DataFrame.replace` to
        replace the atom Z values

        :param to_replace: Z value to replace
        :type to_replace: int
        :param value: what it is going to be replaced with
        :type value: int

        :example:

        >>> stru = Structure(symbols=['Na', 'Cl'], positions=[[0,0,0],[0.5,0.5,0.5]], unitcell=5.64)
        >>> print(stru.get_atom_count())
        Cl    1
        Na    1
        Name: symbol, dtype: int64
        >>> stru.replace_atom(11, 111)
        >>> print(stru.get_atom_count())
        Cl    1
        Rg    1
        Name: symbol, dtype: int64

        """
        self.atoms.Z.replace(to_replace=to_replace, value=value, inplace=True, method=None)
        self.update_atom_symbols()

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
        >>> print(stru.element)  # doctest: +ALLOW_UNICODE
        ['Na' 'Cl' 'Na' 'Cl']
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
        >>> print(stru.element)  # doctest: +ALLOW_UNICODE
        ['Na' 'Na' 'Na' 'Na' 'Na' 'Na' 'Na' 'Na']
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

    def to_ase(self):
        from ase import Atoms
        return Atoms(symbols=self.get_chemical_symbols(),
                     scaled_positions=self.get_scaled_positions(),
                     cell=self.unitcell.cell)

    def get_neighbors(self, site=0, target_site=None, minD=0.01, maxD=1.1):
        """


        Return a :class:`javelin.neighborlist.NeighborList` for the given sites and distances
        """
        from javelin.neighborlist import NeighborList
        from math import ceil, floor
        nl = NeighborList()
        site_aver = self.get_average_site(site, separate_site=False)
        for other_site in self.atoms.index.get_level_values(3).unique():
            if target_site is not None and other_site != target_site:
                continue
            other_site_aver = self.get_average_site(other_site, separate_site=False)
            for i in range(-ceil(maxD+site_aver['x']), floor(maxD+site_aver['x']+1)):
                di = site_aver['x'] - (other_site_aver['x'] + i)
                for j in range(-ceil(maxD+site_aver['y']), floor(maxD+site_aver['y']+1)):
                    dj = site_aver['y'] - (other_site_aver['y'] + j)
                    for k in range(-ceil(maxD+site_aver['z']), floor(maxD+site_aver['z']+1)):
                        dk = site_aver['z'] - (other_site_aver['z'] + k)
                        if minD**2 <= di**2 + dj**2 + dk**2 <= maxD**2:
                            nl.append([site, other_site, i, j, k])
        return nl

    def get_occupational_correlation(self, vectors, atom):
        """
        :param vectors: neighbor vectors
        :type vectors: :class:`javelin.neighborlist.NeighborList` or `n x 5` array of
            neighbor vectors
        :param atom: atom type for which to calculate correlation
        :type atom: int
        :return: occupational correlation
        :rtype: float
        """
        from pandas import MultiIndex
        vectors = np.asarray(vectors)

        count = 0
        total = 0
        match_count = 0
        for site1, site2, i, j, k in vectors:
            Z1 = self.atoms.xs(site1, level='site').Z
            Z2 = self.atoms.xs(site2, level='site').Z
            Z2 = Z2.reindex(MultiIndex.from_product(
                [np.roll(Z2.index.get_level_values(0).drop_duplicates(), i),
                 np.roll(Z2.index.get_level_values(1).drop_duplicates(), j),
                 np.roll(Z2.index.get_level_values(2).drop_duplicates(), k)],
                names=['i', 'j', 'k']))
            count += Z1.value_counts()[atom]
            total += Z1.size
            match_count += np.logical_and(Z1.values == atom, Z2.values == atom).sum()
        theta = count/total
        return (match_count/total - theta*theta)/(theta*(1-theta))

    def get_displacement_correlation(self, vectors, direction=(1, 1, 1), direction2=None):
        """
        :param vectors: neighbor vectors
        :type vectors: :class:`javelin.neighborlist.NeighborList` or `n x 5` array of
            neighbor vectors

        :return: displacement correlation
        :rtype: float
        """
        from pandas import MultiIndex
        dir_dict = {'x': (1, 0, 0),
                    'y': (0, 1, 0),
                    'z': (0, 0, 1)}

        aver_stru = self.get_average_structure(separate_sites=False)

        vectors = np.asarray(vectors)

        if direction2 is None:
            direction2 = direction

        if direction in dir_dict:
            direction = dir_dict[direction]

        if direction2 in dir_dict:
            direction2 = dir_dict[direction2]

        sum0 = 0
        sum1 = 0
        sum2 = 0
        for site1, site2, i, j, k in vectors:
            atoms1 = self.atoms.xs(site1, level='site')
            atoms2 = self.atoms.xs(site2, level='site')
            atoms2 = atoms2.reindex(MultiIndex.from_product(
                [np.roll(atoms2.index.get_level_values(0).drop_duplicates(), i),
                 np.roll(atoms2.index.get_level_values(1).drop_duplicates(), j),
                 np.roll(atoms2.index.get_level_values(2).drop_duplicates(), k)],
                names=['i', 'j', 'k']))
            temp1 = ((atoms1.x.values - aver_stru[site1]['x'])*direction[0] +
                     (atoms1.y.values - aver_stru[site1]['y'])*direction[1] +
                     (atoms1.z.values - aver_stru[site1]['z'])*direction[2])
            temp2 = ((atoms2.x.values - aver_stru[site2]['x'])*direction2[0] +
                     (atoms2.y.values - aver_stru[site2]['y'])*direction2[1] +
                     (atoms2.z.values - aver_stru[site2]['z'])*direction2[2])
            sum0 += (temp1*temp2).sum()
            sum1 += np.square(temp1).sum()
            sum2 += np.square(temp2).sum()

        return sum0/np.sqrt(sum1*sum2)

    def get_average_structure(self, separate_sites=True):
        output = {}
        for site in self.atoms.index.get_level_values(3).unique():
            output[site] = self.get_average_site(site=site, separate_site=separate_sites)
        return output

    def get_average_site(self, site=0, separate_site=True):
        atoms_site = self.atoms.xs(site, level='site')
        if separate_site:
            output = {}
            for atom in atoms_site.symbol.unique():
                atoms_site_atom = atoms_site[atoms_site.symbol == atom]
                output[atom] = {'occ': atoms_site_atom.size/atoms_site.size,
                                'x': atoms_site.x.mean(),
                                'y': atoms_site.y.mean(),
                                'z': atoms_site.z.mean()}
            return output
        else:
            return {'x': atoms_site.x.mean(),
                    'y': atoms_site.y.mean(),
                    'z': atoms_site.z.mean()}


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
    return get_rotation_matrix_from_versor(*axisAngle2Versor(l, m, n, theta, unit=unit))


def get_rotation_matrix_from_versor(w, x, y, z):
    return np.array([[1-2*y**2-2*z**2, 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*x**2-2*z**2, 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*x**2-2*y**2]]).T


def get_miindex(length=0, ncells=None):
    from pandas import MultiIndex

    if ncells is None:
        if length == 0:
            miindex = MultiIndex(levels=[[], [], [], []],
                                 labels=[[], [], [], []],
                                 names=['i', 'j', 'k', 'site'])
        else:
            miindex = MultiIndex.from_product([[0], [0], [0], range(length)],
                                              names=['i', 'j', 'k', 'site'])
    else:
        miindex = MultiIndex.from_product([range(ncells[0]),
                                           range(ncells[1]),
                                           range(ncells[2]),
                                           range(ncells[3])],
                                          names=['i', 'j', 'k', 'site'])

    return miindex
