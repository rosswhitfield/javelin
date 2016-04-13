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
        self._nabs = 101  # abscissa  (lr - ll)
        self._nord = 101  # ordinate  (ul - ll)
        self._napp = 1    # applicate (tl - ll)
        self._dims = 2
        self._2D = True
        self._ll = np.array([0.0, 0.0, 0.0])  # lower left
        self._lr = np.array([1.0, 0.0, 0.0])  # lower right
        self._ul = np.array([0.0, 1.0, 0.0])  # upper left
        self._tl = np.array([0.0, 0.0, 0.0])  # top left

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
        if self._2D:
            return self._nabs, self._nord
        else:
            return self._nabs, self._nord, self._napp

    @bins.setter
    def bins(self, dims):
        if len(dims) == 2:
            self._dims = 2
            self._2D = True
            self._nabs = dims[0]
            self._nord = dims[1]
            self._napp = 1
        elif len(dims) == 3:
            self._dims = 3
            self._2D = False
            self._nabs = dims[0]
            self._nord = dims[1]
            self._napp = dims[2]
        else:
            return  # TODO warning

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

    @property
    def tl(self):
        return self._tl

    @tl.setter
    def tl(self, tl):
        if len(tl) != 3:
            return
        self._tl = np.asarray(tl)

    def calculate(self):
        """Returns a Data object"""
        vector1_step = (self._lr - self._ll)/(self._nabs-1)
        vector2_step = (self.ul - self._ll)/(self._nord-1)
        output_array = np.zeros(self.bins, dtype=np.complex)
        kx = np.zeros(self.bins)
        ky = np.zeros(self.bins)
        kz = np.zeros(self.bins)
        if self._2D:
            for x in range(self._nabs):
                for y in range(self._nord):
                    v = self._ll + x*vector1_step + y*vector2_step
                    kx[x, y] = v[0]
                    ky[x, y] = v[1]
                    kz[x, y] = v[2]
        else:  # assume _dims == 3
            vector3_step = (self.tl - self._ll)/(self._napp-1)
            for x in range(self._nabs):
                for y in range(self._nord):
                    for z in range(self._napp):
                        v = self._ll + x*vector1_step + y*vector2_step + z*vector3_step
                        kx[x, y, z] = v[0]
                        ky[x, y, z] = v[1]
                        kz[x, y, z] = v[2]
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
            temp_array = np.zeros(self.bins, dtype=np.complex)
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
        output_array = np.zeros(self.bins, dtype=np.complex)
        kx, ky, kz = calc_k_grid(self._ll, self._lr, self.ul, self._tl, self.bins)
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
            temp_array = np.zeros(self.bins, dtype=np.complex)
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


def calc_k_grid(ll, lr, ul, tl, bins):
    vabs = lr - ll
    vord = ul - ll
    vapp = tl - ll
    vector1_step = (lr - ll)/(bins[0]-1)
    vector2_step = (ul - ll)/(bins[1]-1)
    kx_bins = get_bin_number(vabs, vord, vapp, bins, 0)
    ky_bins = get_bin_number(vabs, vord, vapp, bins, 1)
    kz_bins = get_bin_number(vabs, vord, vapp, bins, 2)
    kx = np.zeros(kx_bins)
    ky = np.zeros(ky_bins)
    kz = np.zeros(kz_bins)
    if len(bins) == 2:
        for x in range(kx_bins[0]):
            for y in range(kx_bins[1]):
                v = ll + x*vector1_step + y*vector2_step
                kx[x, y] = v[0]
        for x in range(ky_bins[0]):
            for y in range(ky_bins[1]):
                v = ll + x*vector1_step + y*vector2_step
                ky[x, y] = v[1]
        for x in range(kz_bins[0]):
            for y in range(kz_bins[1]):
                v = ll + x*vector1_step + y*vector2_step
                kz[x, y] = v[2]
    else:
        vector3_step = (tl - ll)/(bins[2]-1)
        for x in range(kx_bins[0]):
            for y in range(kx_bins[1]):
                for z in range(kx_bins[2]):
                    v = ll + x*vector1_step + y*vector2_step + z*vector3_step
                    kx[x, y, z] = v[0]
        for x in range(ky_bins[0]):
            for y in range(ky_bins[1]):
                for z in range(ky_bins[2]):
                    v = ll + x*vector1_step + y*vector2_step + z*vector3_step
                    ky[x, y, z] = v[1]
        for x in range(kz_bins[0]):
            for y in range(kz_bins[1]):
                for z in range(kz_bins[2]):
                    v = ll + x*vector1_step + y*vector2_step + z*vector3_step
                    kz[x, y, z] = v[2]
    return kx, ky, kz


def get_bin_number(vabs, vord, vapp, bins, index):
    if vabs[index] == 0:
        binx = 1
    else:
        binx = bins[0]
    if vord[index] == 0:
        biny = 1
    else:
        biny = bins[1]
    if len(bins) == 2:
        return binx, biny
    else:
        if vapp[index] == 0:
            binz = 1
        else:
            binz = bins[2]
        return binx, biny, binz
