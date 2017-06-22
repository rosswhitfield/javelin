"""
=======
fourier
=======

This module define the Fourier object and other functions related to
the fourier transformation.
"""
from __future__ import absolute_import, division
import numpy as np
from javelin.grid import Grid
from javelin.utils import get_unitcell, get_positions, get_atomic_numbers
from javelin.fourier_cython import calculate_cython


class Fourier(object):
    """The Fourier class
    """
    def __init__(self):
        self._structure = None
        self._radiation = 'neutron'
        self._lots = None
        self._number_of_lots = None
        self._average = False
        self.grid = Grid()

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
    def structure(self):
        """The structure from which fourier transform is calculated

        :getter: Returns the structure
        :setter: Sets the structure
        :type: :class:`javelin.structure.Structure`, :class:`ase.Atoms`,
           :class:`diffpy.Structure.structure.Structure`
        """
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure

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

    def __get_q(self):
        qx, qy, qz = self.grid.get_q_meshgrid()
        q = np.linalg.norm(np.array([qx.ravel(),
                                     qy.ravel(),
                                     qz.ravel()]).T * get_unitcell(self.structure).B, axis=1)
        q.shape = qx.shape
        return q*2*np.pi

    def calc(self, mag=False, fast=True, cython=False):
        """Calculates the fourier transform

        :param mag: select if calculating magnetic scattering
        :type mag: bool
        :param fast: fast option
        :type fast: bool
        :param cython: use cython fourier code
        :type cython: bool
        :return: DataArray containing calculated diffuse scattering
        :rtype: :class:`xarray.DataArray`
        """

        if self.structure is None:
            raise ValueError("You have not set a structure for this calculation")

        if self._average:
            aver = self._calculate_average(fast, cython)

        if self.lots is None:
            atomic_numbers = get_atomic_numbers(self.structure)
            positions = get_positions(self.structure)
            if mag:
                magmons = self.structure.get_magnetic_moments()
                return create_xarray_dataarray(self._calculate_magnetic(atomic_numbers,
                                                                        positions,
                                                                        magmons,
                                                                        fast=fast), self.grid)
            else:
                results = self._calculate(atomic_numbers,
                                          positions,
                                          fast=fast,
                                          cython=cython)
                if self._average:
                    results -= aver

                return create_xarray_dataarray(np.real(results*np.conj(results)), self.grid)

        else:  # needs to be Javelin structure, lots by unit cell
            total = np.zeros(self.grid.bins)
            levels = self.structure.atoms.index.levels
            for lot in range(self.number_of_lots):
                print(lot+1, 'out of', self.number_of_lots)
                starti = np.random.randint(len(levels[0]))
                startj = np.random.randint(len(levels[1]))
                startk = np.random.randint(len(levels[2]))
                ri = np.roll(levels[0], -starti)[:self.lots[0]]
                rj = np.roll(levels[1], -startj)[:self.lots[1]]
                rk = np.roll(levels[2], -startk)[:self.lots[2]]
                atoms = self.structure.atoms.loc[ri, rj, rk, :]
                atomic_numbers = atoms.Z.values
                positions = (atoms[['x', 'y', 'z']].values +
                             np.asarray([np.mod(atoms.index.get_level_values(0).values-starti,
                                                len(levels[0])),
                                         np.mod(atoms.index.get_level_values(1).values-startj,
                                                len(levels[1])),
                                         np.mod(atoms.index.get_level_values(2).values-startk,
                                                len(levels[2]))]).T)
                print(starti, startj, startk, ri, rj, rk)
                print(len(atomic_numbers))
                print(positions.shape)
                if mag:
                    magmons = self.structure.magmons.loc[ri, rj, rk, :].values
                    total += self._calculate_magnetic(atomic_numbers, positions, magmons, fast=fast)
                else:
                    results = self._calculate(atomic_numbers, positions, fast=fast, cython=cython)
                    if self._average:
                        results -= aver
                    total += np.real(results*np.conj(results))

            scale = (self.structure.atoms.index.droplevel(3).drop_duplicates().size /
                     (self.number_of_lots*self.lots.prod()))

            return create_xarray_dataarray(total*scale, self.grid)

    def calc_average(self, fast=True, cython=False):
        aver = self._calculate_average(fast, cython)
        return create_xarray_dataarray(np.real(aver*np.conj(aver)), self.grid)

    def _calculate_average(self, fast, cython):
        aver = self._calculate(self.structure.get_atomic_numbers(),
                               self.structure.xyz,
                               fast, cython=cython)

        aver /= self.structure.atoms.index.droplevel(3).drop_duplicates().size

        # compute the interference function of the lot shape

        if self.lots is None:
            index = self.structure.atoms.index.droplevel(3).drop_duplicates()
        else:
            index = self.structure.atoms.loc[range(self.lots[0]),
                                             range(self.lots[1]),
                                             range(self.lots[2]),
                                             :].index.droplevel(3).drop_duplicates()

        aver *= self._calculate(np.zeros(len(index), dtype=np.int),
                                np.asarray([index.get_level_values(0).astype('double').values,
                                            index.get_level_values(1).astype('double').values,
                                            index.get_level_values(2).astype('double').values]).T,
                                fast,
                                use_ff=False,
                                cython=cython)

        return aver

    def _calculate(self, atomic_numbers, positions, fast, use_ff=True, cython=False):
        if fast and not cython:
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
                ff = get_ff(atomic_number, self.radiation, self.__get_q()) if use_ff else 1
            except KeyError as e:
                print("Skipping fourier calculation for atom " + str(e) +
                      ", unable to get scattering factors.")
                continue

            atom_positions = positions[np.where(atomic_numbers == atomic_number)]
            temp_array = np.zeros(self.grid.bins, dtype=np.complex)
            print("Working on atom number", atomic_number, "Total atoms:", len(atom_positions))

            # Loop over atom positions of type atomic_number
            if cython:
                calculate_cython(qx, qy, qz, atom_positions, temp_array.real, temp_array.imag)
            else:
                if fast:
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

    def _calculate_magnetic(self, atomic_numbers, positions, magmons, fast):
        if fast:
            qx, qy, qz = self.grid.get_squashed_q_meshgrid()
        else:
            qx, qy, qz = self.grid.get_q_meshgrid()
        qx *= (2*np.pi)
        qy *= (2*np.pi)
        qz *= (2*np.pi)
        q2 = self.__get_q()**2

        # Get unique list of atomic numbers
        unique_atomic_numbers = np.unique(atomic_numbers)

        # Loop of atom types
        spinx = np.zeros(self.grid.bins, dtype=np.complex)
        spiny = np.zeros(self.grid.bins, dtype=np.complex)
        spinz = np.zeros(self.grid.bins, dtype=np.complex)
        for atomic_number in unique_atomic_numbers:
            try:
                ff = get_mag_ff(atomic_number, self.__get_q(), ion=3)
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
            if fast:
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
