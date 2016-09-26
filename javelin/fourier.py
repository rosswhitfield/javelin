"""This module define the Structure object"""
import numpy as np
import periodictable
from javelin.grid import Grid


class Fourier(object):

    def __init__(self):
        self._structure = None
        self._radiation = 'neutrons'
        self._wavelenght = 1.54
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

    def calculate(self):
        """Returns a Data object"""
        output_array = np.zeros(self.grid.bins, dtype=np.complex)
        kx, ky, kz = self.grid.get_k_meshgrid()
        kx *= (2*np.pi)
        ky *= (2*np.pi)
        kz *= (2*np.pi)
        # Get unique list of atomic numbers
        atomic_numbers = self.structure.get_atomic_numbers()
        unique_atomic_numbers = np.unique(atomic_numbers)
        # Get atom positions
        positions = self.structure.get_scaled_positions()
        # Loop of atom types
        for atomic_number in unique_atomic_numbers:
            if atomic_number == 0:
                continue
            atom_positions = positions[np.where(atomic_numbers == atomic_number)]
            temp_array = np.zeros(self.grid.bins, dtype=np.complex)
            f = periodictable.elements[atomic_number].neutron.b_c
            print("Working on atom number", atomic_number, "Total atoms:", len(atom_positions))
            # Loop over atom positions of type atomic_number
            for atom in atom_positions:
                dot = kx*atom[0] + ky*atom[1] + kz*atom[2]
                temp_array += np.exp(dot*1j)
            output_array += temp_array * f  # scale by form factor
        results = np.real(output_array*np.conj(output_array))
        return self.create_xarray_dataarray(results)

    def calculate_fast(self):
        """Returns a Data object"""
        output_array = np.zeros(self.grid.bins, dtype=np.complex)
        kx, ky, kz = self.grid.get_squashed_k_meshgrid()
        kx *= (2*np.pi)
        ky *= (2*np.pi)
        kz *= (2*np.pi)
        # Get unique list of atomic numbers
        atomic_numbers = self.structure.get_atomic_numbers()
        unique_atomic_numbers = np.unique(atomic_numbers)
        # Get atom positions
        positions = self.structure.get_scaled_positions()
        # Loop of atom types
        for atomic_number in unique_atomic_numbers:
            if atomic_number == 0:
                continue
            atom_positions = positions[np.where(atomic_numbers == atomic_number)]
            temp_array = np.zeros(self.grid.bins, dtype=np.complex)
            f = periodictable.elements[atomic_number].neutron.b_c
            print("Working on atom number", atomic_number, "Total atoms:", len(atom_positions))
            # Loop over atom positions of type atomic_number
            for atom in atom_positions:
                dotx = np.exp(kx*atom[0]*1j)
                doty = np.exp(ky*atom[1]*1j)
                dotz = np.exp(kz*atom[2]*1j)
                temp_array += dotx * doty * dotz
            output_array += temp_array * f  # scale by form factor
        results = np.real(output_array*np.conj(output_array))
        return self.create_xarray_dataarray(results)

    def create_xarray_dataarray(self, values):
        import xarray as xr
        if self.grid._2D:
            return xr.DataArray(data=values,
                                name="Intensity",
                                dims=("Q1", "Q2"),
                                coords=(self.grid._r1, self.grid._r2),
                                attrs=(("radiation", self._radiation),
                                       ("units", self.grid.units)))
        else:
            return xr.DataArray(data=values,
                                name="Intensity",
                                dims=("Q1", "Q2", "Q3"),
                                coords=(self.grid._r1, self.grid._r2, self.grid._r3),
                                attrs=(("radiation", self._radiation),
                                       ("units", self.grid.units)))
