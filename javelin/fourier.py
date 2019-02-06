"""
=======
fourier
=======

This module define the Fourier class and other functions related to
the fourier transformation.
"""
import numpy as np
import pandas as pd
from javelin.grid import Grid
from javelin.utils import get_unitcell, get_positions, get_atomic_numbers
from javelin.fourier_cython import calculate_cython, approx_calculate_cython


class Fourier:
    """The Fourier class contains everything required to calculate the
    diffuse scattering. The only required thing to be set is
    :obj:`javelin.fourier.Fourier.structure`. There are defaults for
    all other options including **grid**, **radiation**, **average**
    structure subtraction and **lots** options.

    :examples:

    >>> from javelin.structure import Structure
    >>> fourier = Fourier()
    >>> print(fourier)
    Radiation         : neutron
    Fourier volume    : complete crystal
    Aver. subtraction : False
    <BLANKLINE>
    Reciprocal layer  :
    lower left  corner :     [ 0.  0.  0.]
    lower right corner :     [ 1.  0.  0.]
    upper left  corner :     [ 0.  1.  0.]
    top   left  corner :     [ 0.  0.  1.]
    <BLANKLINE>
    hor. increment     :     [ 0.01  0.    0.  ]
    vert. increment    :     [ 0.    0.01  0.  ]
    top   increment    :     [ 0.  0.  1.]
    <BLANKLINE>
    # of points        :     101 x 101 x 1
    >>> results = fourier.calc(Structure())
    >>> print(results) # doctest: +SKIP
    <xarray.DataArray 'Intensity' ([ 1.  0.  0.]: 101, [ 0.  1.  0.]: 101, [ 0.  0.  1.]: 1)>
    array([[[ 0.],
            [ 0.],
            ...,
            [ 0.],
            [ 0.]],
    <BLANKLINE>
           [[ 0.],
            [ 0.],
            ...,
            [ 0.],
            [ 0.]],
    <BLANKLINE>
           ...,
           [[ 0.],
            [ 0.],
            ...,
            [ 0.],
            [ 0.]],
    <BLANKLINE>
           [[ 0.],
            [ 0.],
            ...,
            [ 0.],
            [ 0.]]])
    Coordinates:
      * [ 1.  0.  0.]  ([ 1.  0.  0.]) float64 0.0 0.01 0.02 0.03 0.04 0.05 0.06 ...
      * [ 0.  1.  0.]  ([ 0.  1.  0.]) float64 0.0 0.01 0.02 0.03 0.04 0.05 0.06 ...
      * [ 0.  0.  1.]  ([ 0.  0.  1.]) float64 0.0
    Attributes:
        units:    r.l.u

    """
    def __init__(self):
        self._radiation = 'neutron'
        self._lots = None
        self._number_of_lots = None
        self._average = False
        self._magnetic = False
        self._cython = True
        self._approximate = True
        self._fast = True

        #: The **grid** attribute defines the reciprocal volume from
        #: which the scattering will be calculated. Must of type
        #: :class:`javelin.grid.Grid` And check
        #: :class:`javelin.grid.Grid` for details on how to change the
        #: grid.
        self.grid = Grid()

    def __str__(self):
        return """Radiation         : {}
Fourier volume    : {}
Aver. subtraction : {}

Reciprocal layer  :
{}""".format(self.radiation,
             "complete crystal" if self.lots is None else "{} lots of {} x {} x {} unit cells"
             .format(self.number_of_lots, *self.lots),
             self.average,
             self.grid)

    @property
    def radiation(self):
        """The radiation used

        :getter: Returns the radiation selected
        :setter: Sets the radiation
        :type: str ('xray' or 'neutron')
        """
        return self._radiation

    @radiation.setter
    def radiation(self, rad):
        if rad not in ('neutron', 'xray'):
            raise ValueError("radiation must be one of 'neutron' or 'xray'")
        self._radiation = rad

    @property
    def lots(self):
        """The size of lots

        :getter: Returns the lots size
        :setter: Sets the lots size
        :type: list of 3 integers or None
        """
        return self._lots

    @lots.setter
    def lots(self, lots):
        if lots is None:
            self._lots = None
        else:
            lots = np.asarray(lots)
            if len(lots) == 3:
                self._lots = lots
            else:
                raise ValueError("Must provied 3 values for lots")

    @property
    def number_of_lots(self):
        """The number of lots to use

        :getter: Returns the number of lots
        :setter: Sets the number of lots
        :type: int
        """
        return self._number_of_lots

    @number_of_lots.setter
    def number_of_lots(self, value):
        self._number_of_lots = value

    @property
    def average(self):
        """This sets the options of calculating average structure and
        subtracted it from the simulated scattering

        :getter: Returns bool of average structure subtraction option
        :setter: Sets whether average structure is subtracted
        :type: bool
        """
        return self._average

    @average.setter
    def average(self, value):
        if isinstance(value, bool):
            self._average = value
        else:
            raise TypeError("Expected a bool, True or False")

    @property
    def magnetic(self):
        """This sets the options of calculating the magnetic scattering
        instead of nuclear. This assume neutrons are being used.

        :getter: Returns bool of magnetic scattering option
        :setter: Sets whether magnetic sacttering is calculated
        :type: bool

        """
        return self._magnetic

    @magnetic.setter
    def magnetic(self, value):
        if isinstance(value, bool):
            self._magnetic = value
        else:
            raise TypeError("Expected a bool, True or False")

    @property
    def approximate(self):
        """This sets the options of calculating the approximate scattering
        instead of exact. This is much quicker and is likely good enough for
        most cases.

        :getter: Returns bool of approximate scattering option
        :setter: Sets whether approximate sacttering is calculated
        :type: bool

        """
        return self._approximate

    @approximate.setter
    def approximate(self, value):
        if isinstance(value, bool):
            self._approximate = value
        else:
            raise TypeError("Expected a bool, True or False")

    def __get_q(self, unitcell):
        qx, qy, qz = self.grid.get_q_meshgrid()
        q = np.linalg.norm(np.array([qx.ravel(),
                                     qy.ravel(),
                                     qz.ravel()]).T @ unitcell.B, axis=1)
        q.shape = qx.shape
        return q*2*np.pi

    def calc(self, structure):
        """Calculates the fourier transform

        :param structure: The structure from which fourier transform
        is calculated. The calculation work with any of the following
        types of structures :class:`javelin.structure.Structure`,
        :class:`ase.Atoms` or
        :class:`diffpy.Structure.structure.Structure` but if you are
        using average structure subtraction or the lots option it
        needs to be :class:`javelin.structure.Structure` type.

        :return: DataArray containing calculated diffuse scattering
        :rtype: :class:`xarray.DataArray`

        """

        if structure is None:
            raise ValueError("You have not set a structure for this calculation")

        if self.average:
            aver = self._calculate_average(structure)

        unitcell = get_unitcell(structure)

        if self.lots is None:
            atomic_numbers = get_atomic_numbers(structure)
            positions = get_positions(structure)
            if self.magnetic:
                magmons = structure.get_magnetic_moments()
                return create_xarray_dataarray(self._calculate_magnetic(atomic_numbers,
                                                                        positions,
                                                                        unitcell,
                                                                        magmons), self.grid)
            else:
                results = self._calculate(atomic_numbers, positions, unitcell)
                if self.average:
                    results -= aver

                return create_xarray_dataarray(np.real(results*np.conj(results)), self.grid)

        else:  # needs to be Javelin structure, lots by unit cell
            total = np.zeros(self.grid.bins)
            levels = structure.atoms.index.levels
            for lot in range(self.number_of_lots):
                print(lot+1, 'out of', self.number_of_lots)
                starti = np.random.randint(len(levels[0]))
                startj = np.random.randint(len(levels[1]))
                startk = np.random.randint(len(levels[2]))
                ri = np.roll(levels[0], -starti)[:self.lots[0]]
                rj = np.roll(levels[1], -startj)[:self.lots[1]]
                rk = np.roll(levels[2], -startk)[:self.lots[2]]
                atoms = structure.atoms.loc[ri, rj, rk, :]
                atomic_numbers = atoms.Z.values
                positions = (atoms[['x', 'y', 'z']].values +
                             np.asarray([np.mod(atoms.index.get_level_values(0).values-starti,
                                                len(levels[0])),
                                         np.mod(atoms.index.get_level_values(1).values-startj,
                                                len(levels[1])),
                                         np.mod(atoms.index.get_level_values(2).values-startk,
                                                len(levels[2]))]).T)
                if self.magnetic:
                    magmons = structure.magmons.loc[ri, rj, rk, :].values
                    total += self._calculate_magnetic(atomic_numbers, positions, unitcell, magmons)
                else:
                    results = self._calculate(atomic_numbers, positions, unitcell)
                    if self.average:
                        results -= aver
                    total += np.real(results*np.conj(results))

            scale = (structure.atoms.index.droplevel(3).drop_duplicates().size /
                     (self.number_of_lots*self.lots.prod()))

            return create_xarray_dataarray(total*scale, self.grid)

    def calc_average(self, structure):
        """Calculates the scattering from the avarage structure

        :param structure: The structure from which fourier transform
        is calculated. The calculation work with any of the following
        types of structures :class:`javelin.structure.Structure`,
        :class:`ase.Atoms` or
        :class:`diffpy.Structure.structure.Structure` but if you are
        using average structure subtraction or the lots option it
        needs to be :class:`javelin.structure.Structure` type.

        :return: DataArray containing calculated average scattering
        :rtype: :class:`xarray.DataArray`
        """

        if structure is None:
            raise ValueError("You have not set a structure for this calculation")

        aver = self._calculate_average(structure)
        return create_xarray_dataarray(np.real(aver*np.conj(aver)), self.grid)

    def _calculate_average(self, structure):
        aver = self._calculate(get_atomic_numbers(structure),
                               structure.xyz, get_unitcell(structure))

        aver /= structure.atoms.index.droplevel(3).drop_duplicates().size

        # compute the interference function of the lot shape

        if self.lots is None:
            index = structure.atoms.index.droplevel(3).drop_duplicates()
        else:
            index = structure.atoms.loc[pd.RangeIndex(self.lots[0]),
                                        pd.RangeIndex(self.lots[1]),
                                        pd.RangeIndex(self.lots[2]),
                                        :].index.droplevel(3).drop_duplicates()

        aver *= self._calculate(np.zeros(len(index), dtype=np.int),
                                np.asarray([index.get_level_values(0).astype('double').values,
                                            index.get_level_values(1).astype('double').values,
                                            index.get_level_values(2).astype('double').values]).T,
                                use_ff=False)

        return aver

    def _calculate(self, atomic_numbers, positions, unitcell=None, use_ff=True):
        if self._fast and not self._cython:
            qx, qy, qz = self.grid.get_squashed_q_meshgrid()
        else:
            qx, qy, qz = self.grid.get_q_meshgrid()
        qx *= (2*np.pi)
        qy *= (2*np.pi)
        qz *= (2*np.pi)

        # Get unique list of atomic numbers
        unique_atomic_numbers = np.unique(atomic_numbers)

        results = np.zeros(self.grid.bins, dtype=np.complex)
        # Loop of atom types
        for atomic_number in unique_atomic_numbers:
            try:
                ff = get_ff(atomic_number, self.radiation, self.__get_q(unitcell)) if use_ff else 1
            except KeyError as e:
                print("Skipping fourier calculation for atom " + str(e) +
                      ", unable to get scattering factors.")
                continue

            atom_positions = positions[np.where(atomic_numbers == atomic_number)]
            temp_array = np.zeros(self.grid.bins, dtype=np.complex)
            print("Working on atom number", atomic_number, "Total atoms:", len(atom_positions))

            # Loop over atom positions of type atomic_number
            if self._cython:
                if self.approximate:
                    cex = np.exp(np.linspace(0, 2j*np.pi*(1-2**-16), 2**16))
                    approx_calculate_cython(self.grid.ll, self.grid.v1_delta,
                                            self.grid.v2_delta, self.grid.v3_delta,
                                            atom_positions, temp_array.real,
                                            temp_array.imag, cex.real, cex.imag)
                else:
                    calculate_cython(qx, qy, qz, atom_positions, temp_array.real, temp_array.imag)
            else:
                if self._fast:
                    for atom in atom_positions:
                        dotx = np.exp(qx*atom[0]*1j)
                        doty = np.exp(qy*atom[1]*1j)
                        dotz = np.exp(qz*atom[2]*1j)
                        temp_array += dotx * doty * dotz
                else:
                    for atom in atom_positions:
                        dot = qx*atom[0] + qy*atom[1] + qz*atom[2]
                        temp_array += np.exp(dot*1j)

            results += temp_array * ff  # scale by form factor

        return results

    def _calculate_magnetic(self, atomic_numbers, positions, unitcell, magmons):
        if self._fast:
            qx, qy, qz = self.grid.get_squashed_q_meshgrid()
        else:
            qx, qy, qz = self.grid.get_q_meshgrid()
        qx *= (2*np.pi)
        qy *= (2*np.pi)
        qz *= (2*np.pi)
        q2 = self.__get_q(unitcell)**2

        # Get unique list of atomic numbers
        unique_atomic_numbers = np.unique(atomic_numbers)

        # Loop of atom types
        spinx = np.zeros(self.grid.bins, dtype=np.complex)
        spiny = np.zeros(self.grid.bins, dtype=np.complex)
        spinz = np.zeros(self.grid.bins, dtype=np.complex)
        for atomic_number in unique_atomic_numbers:
            try:
                ff = get_mag_ff(atomic_number, self.__get_q(unitcell), ion=3)
            except (AttributeError, KeyError) as e:
                print("Skipping fourier calculation for atom " + str(e) +
                      ", unable to get magnetic scattering factors.")
                continue

            atom_positions = positions[np.where(atomic_numbers == atomic_number)]
            temp_spinx = np.zeros(self.grid.bins, dtype=np.complex)
            temp_spiny = np.zeros(self.grid.bins, dtype=np.complex)
            temp_spinz = np.zeros(self.grid.bins, dtype=np.complex)
            print("Working on atom number", atomic_number, "Total atoms:", len(atom_positions))
            # Loop over atom positions of type atomic_number
            if self._fast:
                for atom, spin in zip(atom_positions, magmons):
                    dotx = np.exp(qx*atom[0]*1j)
                    doty = np.exp(qy*atom[1]*1j)
                    dotz = np.exp(qz*atom[2]*1j)
                    exp_temp = dotx * doty * dotz
                    temp_spinx += exp_temp*spin[0]
                    temp_spiny += exp_temp*spin[1]
                    temp_spinz += exp_temp*spin[2]
            else:
                for atom, spin in zip(atom_positions, magmons):
                    dot = qx*atom[0] + qy*atom[1] + qz*atom[2]
                    exp_temp = np.exp(dot*1j)
                    temp_spinx += exp_temp*spin[0]
                    temp_spiny += exp_temp*spin[1]
                    temp_spinz += exp_temp*spin[2]
            spinx += temp_spinx * ff
            spiny += temp_spiny * ff
            spinz += temp_spinz * ff
        # Calculate vector rejection of spin onto q
        # M - M.Q/|Q|^2 Q
        scale = (spinx*qx + spiny*qy + spinz*qz)/q2
        spinx = spinx - scale * qx
        spiny = spiny - scale * qy
        spinz = spinz - scale * qz
        return np.real(spinx*np.conj(spinx) + spiny*np.conj(spiny) + spinz*np.conj(spinz))


def create_xarray_dataarray(values, grid):
    """Create a xarry DataArray from the input numpy array and grid
    object.

    :param values: Input array containing the scattering intensities
    :type values: :class:`numpy.ndarray`
    :param numbers: Grid object describing the array properties
    :type numbers: :class:`javelin.grid.Grid`
    :return: DataArray produced from the values and grid object
    :rtype: :class:`xarray.DataArray`
    """
    import xarray as xr
    return xr.DataArray(data=values,
                        name="Intensity",
                        dims=(grid.get_axes_names()),
                        coords=(grid.r1, grid.r2, grid.r3),
                        attrs=(("units", grid.units),))


def get_ff(atomic_number, radiation, q=None):
    """Returns the form factor for a given atomic number, radiation and q
    values

    :param atomic_number: atomic number
    :type atomic_number: int
    :param radiation: type of radiation ('xray' or 'neutron')
    :type radiation: str
    :param q: value or values of q for which to get form factors
    :type q: float, list, :class:`numpy.ndarray`
    :return: form factors for given q
    :rtype: float, :class:`numpy.ndarray`

    :Examples:

    >>> get_ff(8, 'neutron')
    5.805

    >>> get_ff(8, 'xray', q=2.0)
    6.31826029176493

    >>> get_ff(8, 'xray', q=[0.0, 3.5, 7.0])
    array([ 7.999706  ,  4.38417867,  2.08928068])
    """
    import periodictable

    if atomic_number < 1:
        raise KeyError(atomic_number)

    if radiation == 'neutron':
        return periodictable.elements[atomic_number].neutron.b_c
    elif radiation == 'xray':
        return periodictable.elements[atomic_number].xray.f0(q)
    else:
        raise ValueError("Unknown radition: " + radiation)


def get_mag_ff(atomic_number, q, ion=0, j=0):
    """Returns the j0 magnetic form factor for a given atomic number,
    radiation and q values

    :param atomic_number: atomic number
    :type atomic_number: int
    :param q: value or values of q for which to get form factors
    :type q: float, list, :class:`numpy.ndarray`
    :param ion: charge of selected atom
    :type ion: int
    :param j: order of spherical Bessel function (0, 2, 4 or 6)
    :type j: int
    :return: magnetic form factor for given q
    :rtype: float, :class:`numpy.ndarray`

    :Examples:

    >>> get_mag_ff(8, q=2, ion=1)
    0.58510426376585045

    >>> get_mag_ff(26, q=[0.0, 3.5, 7.0], ion=2)
    array([ 1.        ,  0.49729671,  0.09979243])

    >>> get_mag_ff(26, q=[0.0, 3.5, 7.0], ion=4)
    array([ 0.9997    ,  0.58273549,  0.13948496])

    >>> get_mag_ff(26, q=[0.0, 3.5, 7.0], ion=4, j=4)
    array([ 0.       ,  0.0149604,  0.0759222])
    """
    import periodictable
    return getattr(periodictable.elements[atomic_number].magnetic_ff[ion], 'j'+str(j)+'_Q')(q)
