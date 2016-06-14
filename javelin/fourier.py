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
        self._vertices = {'ll': np.array([0.0, 0.0, 0.0]),  # lower left
                          'lr': np.array([1.0, 0.0, 0.0]),  # lower right
                          'ul': np.array([0.0, 1.0, 0.0]),  # upper left
                          'tl': np.array([0.0, 0.0, 0.0])}  # top left

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
        dims = np.asarray(dims)
        if (dims < 2).any():
            raise ValueError("Must have more than 1 bin in each direction")
        if len(dims) == 2:
            self._dims = 2
            self._2D = True
            self._nabs = dims[0]
            self._nord = dims[1]
            self._napp = 0
        elif len(dims) == 3:
            self._dims = 3
            self._2D = False
            self._nabs = dims[0]
            self._nord = dims[1]
            self._napp = dims[2]
        else:
            raise ValueError("Must provide 2 or 3 dimensions")

    @property
    def ll(self):
        return self._vertices['ll']

    @ll.setter
    def ll(self, ll):
        if len(ll) != 3:
            raise ValueError("Must have length 3")
        self._vertices['ll'] = np.asarray(ll)

    @property
    def lr(self):
        return self._vertices['lr']

    @lr.setter
    def lr(self, lr):
        if len(lr) != 3:
            raise ValueError("Must have length 3")
        self._vertices['lr'] = np.asarray(lr)

    @property
    def ul(self):
        return self._vertices['ul']

    @ul.setter
    def ul(self, ul):
        if len(ul) != 3:
            raise ValueError("Must have length 3")
        self._vertices['ul'] = np.asarray(ul)

    @property
    def tl(self):
        return self._vertices['tl']

    @tl.setter
    def tl(self, tl):
        if len(tl) != 3:
            raise ValueError("Must have length 3")
        self._vertices['tl'] = np.asarray(tl)

    def validate_vectors(self):
        vertices = ['lr', 'ul'] if self._2D else ['lr', 'ul', 'tl']
        vector_dict = {}
        for vertex in vertices:
            # Create vector from ll to 'vertex'
            vector = self._vertices[vertex] - self._vertices['ll']
            # Check length of vector
            if length(vector) == 0:
                raise ValueError("Distance between ll and " + vertex + " is 0")
            # Compare vector with previous to check it parallel
            for item in vector_dict:
                if check_parallel(vector, vector_dict[item]):
                    raise ValueError("Vector from ll to " + vertex +
                                     " is parallel with the vector from ll to "+item)
            vector_dict[vertex] = vector  # Store to allow comparison with other vectors

    def calculate(self):
        """Returns a Data object"""
        self.validate_vectors()
        dx = (self.lr - self.ll)/(self._nabs-1)
        dy = (self.ul - self.ll)/(self._nord-1)
        output_array = np.zeros(self.bins, dtype=np.complex)
        x = np.arange(self._nabs).reshape((self._nabs, 1))
        y = np.arange(self._nord).reshape((1, self._nord))
        if self._2D:
            kx = self.ll[0] + x*dx[0] + y*dy[0]
            ky = self.ll[1] + x*dx[1] + y*dy[1]
            kz = self.ll[2] + x*dx[2] + y*dy[2]
        else:  # assume _dims == 3
            x.shape = (self._nabs, 1, 1)
            y.shape = (1, self._nord, 1)
            z = np.arange(self._napp).reshape((1, 1, self._napp))
            dz = (self.tl - self.ll)/(self._napp-1)
            kx = self.ll[0] + x*dx[0] + y*dy[0] + z*dz[0]
            ky = self.ll[1] + x*dx[1] + y*dy[1] + z*dz[1]
            kz = self.ll[2] + x*dx[2] + y*dy[2] + z*dz[2]
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
            temp_array = np.zeros(self.bins, dtype=np.complex)
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
        self.validate_vectors()
        output_array = np.zeros(self.bins, dtype=np.complex)
        kx, ky, kz = calc_k_grid(self.ll, self.lr, self.ul, self.tl, self.bins)
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
            temp_array = np.zeros(self.bins, dtype=np.complex)
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
        x = np.linspace(length(self.ll), length(self.lr), self.bins[0])
        y = np.linspace(length(self.ll), length(self.ul), self.bins[1])
        if self._2D:
            return xr.DataArray(data=values,
                                name="Intensity",
                                dims=("Q1", "Q2"),
                                coords=(x, y),
                                attrs=(("radiation", self._radiation),
                                       ("units", "q")))
        else:
            z = np.linspace(length(self.ll), length(self.tl), self.bins[2])
            return xr.DataArray(data=values,
                                name="Intensity",
                                dims=("Q1", "Q2", "Q3"),
                                coords=(x, y, z),
                                attrs=(("radiation", self._radiation),
                                       ("units", "q")))


def calc_k_grid(ll, lr, ul, tl, bins):
    vabs = lr - ll
    vord = ul - ll
    vapp = tl - ll
    dx = (lr - ll)/(bins[0]-1)
    dy = (ul - ll)/(bins[1]-1)
    kx_bins = get_bin_number(vabs, vord, vapp, bins, 0)
    ky_bins = get_bin_number(vabs, vord, vapp, bins, 1)
    kz_bins = get_bin_number(vabs, vord, vapp, bins, 2)
    kx = np.zeros(kx_bins)
    ky = np.zeros(ky_bins)
    kz = np.zeros(kz_bins)
    if len(bins) == 2:
        x = np.arange(kx_bins[0]).reshape((kx_bins[0], 1))
        y = np.arange(kx_bins[1]).reshape((1, kx_bins[1]))
        kx = ll[0] + x*dx[0] + y*dy[0]
        x = np.arange(ky_bins[0]).reshape((ky_bins[0], 1))
        y = np.arange(ky_bins[1]).reshape((1, ky_bins[1]))
        ky = ll[1] + x*dx[1] + y*dy[1]
        x = np.arange(kz_bins[0]).reshape((kz_bins[0], 1))
        y = np.arange(kz_bins[1]).reshape((1, kz_bins[1]))
        kz = ll[2] + x*dx[2] + y*dy[2]
    else:
        dz = (tl - ll)/(bins[2]-1)
        x = np.arange(kx_bins[0]).reshape((kx_bins[0], 1, 1))
        y = np.arange(kx_bins[1]).reshape((1, kx_bins[1], 1))
        z = np.arange(kx_bins[2]).reshape((1, 1, kx_bins[2]))
        kx = ll[0] + x*dx[0] + y*dy[0] + z*dz[0]
        x = np.arange(ky_bins[0]).reshape((ky_bins[0], 1, 1))
        y = np.arange(ky_bins[1]).reshape((1, ky_bins[1], 1))
        z = np.arange(ky_bins[2]).reshape((1, 1, ky_bins[2]))
        ky = ll[1] + x*dx[1] + y*dy[1] + z*dz[1]
        x = np.arange(kz_bins[0]).reshape((kz_bins[0], 1, 1))
        y = np.arange(kz_bins[1]).reshape((1, kz_bins[1], 1))
        z = np.arange(kz_bins[2]).reshape((1, 1, kz_bins[2]))
        kz = ll[2] + x*dx[2] + y*dy[2] + z*dz[2]
    return kx, ky, kz


def get_bin_number(vabs, vord, vapp, bins, index):
    binx = 1 if vabs[index] == 0 else bins[0]
    biny = 1 if vord[index] == 0 else bins[1]
    if len(bins) == 2:
        return binx, biny
    else:
        binz = 1 if vapp[index] == 0 else bins[2]
        return binx, biny, binz


def length(v):
    return np.sqrt((np.dot(v, v)))


def check_parallel(v1, v2):
    return (np.cross(v1, v2) == 0).all()


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (length(v1) * length(v2)))
