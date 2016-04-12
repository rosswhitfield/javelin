"""This module define the Structure object"""
import numpy as np
import periodictable


class Fourier(object):

    def __init__(self):
        self._structure = None
        self._radiation = 'neutrons'
        self._wavelenght = 1.54
        self._lots = None
        self._average = 0.0
        self._na = 101
        self._no = 101
        self._ll = np.array([0.0, 0.0, 0.0])
        self._lr = np.array([2.0, 0.0, 0.0])
        self._ul = np.array([0.0, 2.0, 0.0])

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

    @property
    def bins(self):
        return self._na, self._no

    @bins.setter
    def bins(self, dims):
        if len(dims) < 2:
            return
        self._na = dims[0]
        self._no = dims[1]

    @property
    def ll(self):
        return self._ll

    @ll.setter
    def ll(self, ll):
        if len(ll) != 3:
            return
        self._ll = np.asarray(ll)

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, lr):
        if len(lr) != 3:
            return
        self._lr = np.asarray(lr)

    @property
    def ul(self):
        return self._ul

    @ul.setter
    def ul(self, ul):
        if len(ul) != 3:
            return
        self._ul = np.asarray(ul)

    def calculate(self):
        """Returns a Data object"""
        output_array = np.zeros([self._na, self._no], dtype=np.complex)
        vector1_step = (self._lr - self._ll)/(self._na-1)
        vector2_step = (self.ul - self._ll)/(self._no-1)
        kx = np.zeros([self._na, self._no])
        ky = np.zeros([self._na, self._no])
        kz = np.zeros([self._na, self._no])
        for x in range(self._na):
            for y in range(self._no):
                v = self._ll + x*vector1_step + y*vector2_step
                kx[x, y] = v[0]
                ky[x, y] = v[1]
                kz[x, y] = v[2]
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
            atom_positions = positions[np.where(atomic_numbers == atomic_number)]
            temp_array = np.zeros([self._na, self._no], dtype=np.complex)
            f = periodictable.elements[atomic_number].neutron.b_c
            print("Working on atom number", atomic_number, "Total atoms:", len(atom_positions))
            # Loop over atom positions of type atomic_number
            for atom in atom_positions:
                dot = kx*atom[0] + ky*atom[1] + kz*atom[2]
                temp_array += np.exp(dot*1j)
            output_array += temp_array * f  # scale by form factor
        results = np.real(output_array*np.conj(output_array))
        return results

    def calculate_fast(self):
        """Returns a Data object"""
        output_array = np.zeros([self._na, self._no], dtype=np.complex)
        kx, ky, kz = calc_k_grid(self._ll, self._lr, self.ul, self._na, self._no)
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
            atom_positions = positions[np.where(atomic_numbers == atomic_number)]
            temp_array = np.zeros([self._na, self._no], dtype=np.complex)
            f = periodictable.elements[atomic_number].neutron.b_c
            print("Working on atom number", atomic_number, "Total atoms:", len(atom_positions))
            # Loop over atom positions of type atomic_number
            for atom in atom_positions:
                dotx = np.exp(kx*atom[0]*1j)
                doty = np.exp(ky*atom[1]*1j)
                dotz = np.exp(kz*atom[2]*1j)
                sumexp = dotx * doty * dotz
                temp_array += sumexp
            output_array += temp_array * f  # scale by form factor
        results = np.real(output_array*np.conj(output_array))
        return results


def calc_k_grid(ll, lr, ul, na, no):
    va = lr - ll
    vo = ul - ll
    vector1_step = (lr - ll)/(na-1)
    vector2_step = (ul - ll)/(no-1)
    kx_bina, kx_bino = get_bin_number(va, vo, na, no, 0)
    ky_bina, ky_bino = get_bin_number(va, vo, na, no, 1)
    kz_bina, kz_bino = get_bin_number(va, vo, na, no, 2)
    kx = np.zeros([kx_bina, kx_bino])
    ky = np.zeros([ky_bina, ky_bino])
    kz = np.zeros([kz_bina, kz_bino])
    for x in range(kx_bina):
        for y in range(kx_bino):
            v = ll + x*vector1_step + y*vector2_step
            kx[x, y] = v[0]
    for x in range(ky_bina):
        for y in range(ky_bino):
            v = ll + x*vector1_step + y*vector2_step
            ky[x, y] = v[1]
    for x in range(kz_bina):
        for y in range(kz_bino):
            v = ll + x*vector1_step + y*vector2_step
            kz[x, y] = v[2]
    return kx, ky, kz


def get_bin_number(va, vo, na, no, index):
    if va[index] == 0:
        binx = 1
    else:
        binx = na
    if vo[index] == 0:
        biny = 1
    else:
        biny = no
    return binx, biny
