"""
=======
fourier
=======

This module define the Structure object
"""
import numpy as np
import periodictable
from javelin.grid import Grid


class Fourier(object):

    def __init__(self):
        self._structure = None
        self._radiation = 'neutrons'
        self._lots = None
        self._average = 0.0
        self.grid = Grid()

    @property
    def radiation(self):
        return self._radiation

    @radiation.setter
    def radiation(self, rad):
        self._radiation = rad

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, stru):
        self._structure = stru

    def __get_unitcell(self):
        """Wrapper to get the unit cell from different structure classes"""
        from javelin.unitcell import UnitCell
        try:  # javelin structure
            return self._structure.unitcell
        except AttributeError:
            try:  # diffpy structure
                return UnitCell(self._structure.lattice.abcABG())
            except AttributeError:
                try:  # ASE structure
                    from ase.geometry import cell_to_cellpar
                    return UnitCell(cell_to_cellpar(self._structure.cell))
                except (ImportError, AttributeError):
                    raise ValueError("Unable to get unit cell from structure")

    def __get_q(self):
            qx, qy, qz = self.grid.get_q_meshgrid()
            q = np.linalg.norm(np.array([qx.ravel(),
                                         qy.ravel(),
                                         qz.ravel()]).T * self.__get_unitcell().B, axis=1)
            q.shape = qx.shape
            return q*2*np.pi

    def __get_ff(self, atomic_number):
        if self._radiation == 'neutrons':
            return periodictable.elements[atomic_number].neutron.b_c
        elif self._radiation == 'xray':
            return periodictable.elements[atomic_number].xray.f0(self.__get_q())
        else:
            raise ValueError("Unknown radition: " + self._radiation)

    def __get_mag_ff(self, atomic_number, ion=0):
        return periodictable.elements[atomic_number].magnetic_ff[ion].j0_Q(self.__get_q())

    def __get_positions(self):
        """Wrapper to get the positions from different structure classes"""
        try:  # ASE structure
            return self._structure.get_scaled_positions()
        except AttributeError:
            try:  # diffpy structure
                return self._structure.xyz
            except AttributeError:
                raise ValueError("Unable to get positions from structure")

    def __get_atomic_numbers(self):
        """Wrapper to get the atomic numbers from different structure classes"""
        from javelin.utils import get_atomic_number_symbol
        try:  # ASE structure
            return self._structure.get_atomic_numbers()
        except AttributeError:
            try:  # diffpy structure
                atomic_numbers, _ = get_atomic_number_symbol(symbol=self._structure.element)
                return atomic_numbers
            except AttributeError:
                raise ValueError("Unable to get elements from structure")

    def calculate(self):
        """Returns a Data object"""
        output_array = np.zeros(self.grid.bins, dtype=np.complex)
        qx, qy, qz = self.grid.get_q_meshgrid()
        qx *= (2*np.pi)
        qy *= (2*np.pi)
        qz *= (2*np.pi)
        # Get unique list of atomic numbers
        atomic_numbers = self.__get_atomic_numbers()
        unique_atomic_numbers = np.unique(atomic_numbers)
        # Get atom positions
        positions = self.__get_positions()
        # Loop of atom types
        for atomic_number in unique_atomic_numbers:
            try:
                ff = self.__get_ff(atomic_number)
            except KeyError as e:
                print("Skipping fourier calculation for atom " + str(e) +
                      ", unable to get scattering factors.")
                continue
            atom_positions = positions[np.where(atomic_numbers == atomic_number)]
            temp_array = np.zeros(self.grid.bins, dtype=np.complex)
            print("Working on atom number", atomic_number, "Total atoms:", len(atom_positions))
            # Loop over atom positions of type atomic_number
            for atom in atom_positions:
                dot = qx*atom[0] + qy*atom[1] + qz*atom[2]
                temp_array += np.exp(dot*1j)
            output_array += temp_array * ff  # scale by form factor
        results = np.real(output_array*np.conj(output_array))
        return self.create_xarray_dataarray(results)

    def calculate_fast(self):
        """Returns a Data object"""
        output_array = np.zeros(self.grid.bins, dtype=np.complex)
        qx, qy, qz = self.grid.get_squashed_q_meshgrid()
        qx *= (2*np.pi)
        qy *= (2*np.pi)
        qz *= (2*np.pi)
        # Get unique list of atomic numbers
        atomic_numbers = self.__get_atomic_numbers()
        unique_atomic_numbers = np.unique(atomic_numbers)
        # Get atom positions
        positions = self.__get_positions()
        # Loop of atom types
        for atomic_number in unique_atomic_numbers:
            try:
                ff = self.__get_ff(atomic_number)
            except KeyError as e:
                print("Skipping fourier calculation for atom " + str(e) +
                      ", unable to get scattering factors.")
                continue
            atom_positions = positions[np.where(atomic_numbers == atomic_number)]
            temp_array = np.zeros(self.grid.bins, dtype=np.complex)
            print("Working on atom number", atomic_number, "Total atoms:", len(atom_positions))
            # Loop over atom positions of type atomic_number
            for atom in atom_positions:
                dotx = np.exp(qx*atom[0]*1j)
                doty = np.exp(qy*atom[1]*1j)
                dotz = np.exp(qz*atom[2]*1j)
                temp_array += dotx * doty * dotz
            output_array += temp_array * ff  # scale by form factor
        results = np.real(output_array*np.conj(output_array))
        return self.create_xarray_dataarray(results)

    def calculate_magnetic(self):
        """Returns a Data object"""
        qx, qy, qz = self.grid.get_q_meshgrid()
        qx *= (2*np.pi)
        qy *= (2*np.pi)
        qz *= (2*np.pi)
        q2 = self.__get_q()**2
        # Get unique list of atomic numbers
        atomic_numbers = self.__get_atomic_numbers()
        unique_atomic_numbers = np.unique(atomic_numbers)
        # Get atom positions
        positions = self.__get_positions()
        magmons = self._structure.get_magnetic_moments()
        # Loop of atom types
        spinx = np.zeros(self.grid.bins, dtype=np.complex)
        spiny = np.zeros(self.grid.bins, dtype=np.complex)
        spinz = np.zeros(self.grid.bins, dtype=np.complex)
        for atomic_number in unique_atomic_numbers:
            try:
                ff = self.__get_mag_ff(atomic_number, ion=3)
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
            for atom, spin in zip(atom_positions, magmons):
                dot = qx*atom[0] + qy*atom[1] + qz*atom[2]
                exp_temp = np.exp(dot*1j)
                temp_spinx += exp_temp*spin[0]
                temp_spiny += exp_temp*spin[1]
                temp_spinz += exp_temp*spin[2]
            spinx += temp_spinx * ff
            spiny += temp_spiny * ff
            spinz += temp_spinz * ff
        # Caluculate vector rejection of spin onto q
        # M - M.Q/|Q|^2 Q
        scale = (spinx*qx + spiny*qy + spinz*qz)/q2
        newx = spinx - scale * qx
        newy = spiny - scale * qy
        newz = spinz - scale * qz
        results = np.real(newx*np.conj(newx) + newy*np.conj(newy) + newz*np.conj(newz))
        return self.create_xarray_dataarray(results)

    def calculate_fast_magnetic(self):
        """Returns a Data object"""
        qx, qy, qz = self.grid.get_squashed_q_meshgrid()
        qx *= (2*np.pi)
        qy *= (2*np.pi)
        qz *= (2*np.pi)
        q2 = self.__get_q()**2
        # Get unique list of atomic numbers
        atomic_numbers = self.__get_atomic_numbers()
        unique_atomic_numbers = np.unique(atomic_numbers)
        # Get atom positions
        positions = self.__get_positions()
        magmons = self._structure.get_magnetic_moments()
        # Loop of atom types
        spinx = np.zeros(self.grid.bins, dtype=np.complex)
        spiny = np.zeros(self.grid.bins, dtype=np.complex)
        spinz = np.zeros(self.grid.bins, dtype=np.complex)
        for atomic_number in unique_atomic_numbers:
            try:
                ff = self.__get_mag_ff(atomic_number, ion=3)
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
            for atom, spin in zip(atom_positions, magmons):
                dotx = np.exp(qx*atom[0]*1j)
                doty = np.exp(qy*atom[1]*1j)
                dotz = np.exp(qz*atom[2]*1j)
                exp_temp = dotx * doty * dotz
                temp_spinx += exp_temp*spin[0]
                temp_spiny += exp_temp*spin[1]
                temp_spinz += exp_temp*spin[2]
            spinx += temp_spinx * ff
            spiny += temp_spiny * ff
            spinz += temp_spinz * ff
        # Caluculate vector rejection of spin onto q
        # M - M.Q/|Q|^2 Q
        scale = (spinx*qx + spiny*qy + spinz*qz)/q2
        newx = spinx - scale * qx
        newy = spiny - scale * qy
        newz = spinz - scale * qz
        results = np.real(newx*np.conj(newx) + newy*np.conj(newy) + newz*np.conj(newz))
        return self.create_xarray_dataarray(results)

    def create_xarray_dataarray(self, values):
        import xarray as xr
        if self.grid.twoD:
            return xr.DataArray(data=values,
                                name="Intensity",
                                dims=("Q1", "Q2"),
                                coords=(self.grid.r1, self.grid.r2),
                                attrs=(("radiation", self._radiation),
                                       ("units", self.grid.units)))
        else:
            return xr.DataArray(data=values,
                                name="Intensity",
                                dims=("Q1", "Q2", "Q3"),
                                coords=(self.grid.r1, self.grid.r2, self.grid.r3),
                                attrs=(("radiation", self._radiation),
                                       ("units", self.grid.units)))
