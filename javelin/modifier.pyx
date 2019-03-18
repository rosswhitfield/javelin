"""
========
modifier
========

The Modifier object is the method by which :obj:`javelin.mc.MC`
changes a :obj:`javelin.structure.Structure`.

All modifiers inherit from :obj:`javelin.modifier.BaseModifier`.

"""

import numpy as np
from .random cimport random_int, random_range, random_normal
cimport cython

cdef class BaseModifier:
    """This class does not actually change the structure but is the base
    of all modifiers. The `number_of_cells` is number of random
    location that the modifier will change, `for example` swap type
    modifiers require 2 sites while shift requires only one. The
    methods ``self.initialize_cells(int number_of_cells)`` and
    ``self.initialize_sites(sites)`` must be called to set the number
    of cells and which sites to use."""
    def __init__(self, int number_of_cells = 1, object sites = 0):
        self.initialize_cells(number_of_cells)
        self.initialize_sites(sites)
    def __str__(self):
        return  "{}(number_of_cells={},sites={})".format(self.__class__.__name__,self.number_of_cells,np.array(self.sites))
    cdef void initialize_cells(self, int number_of_cells):
        """Initialize the cells"""
        self.number_of_cells = number_of_cells
        cdef cnp.ndarray[Py_ssize_t, ndim=2] cells = np.zeros((number_of_cells, 4), dtype=np.intp)
        self.cells = cells
    cdef void initialize_sites(self, object sites):
        """Initialize the sites correctly as numpy Py_ssize_t (np.intp) array"""
        self.sites = np.atleast_1d(np.asarray(sites, dtype=np.intp))
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef Py_ssize_t[:,:] get_random_cells(self, Py_ssize_t size_x, Py_ssize_t size_y, Py_ssize_t size_z) except *:
        """Sets internally and returns randomly selected cells, shape
        (number_of_cells, 3), based on `size_x`, `size_y` and
        `size_z`. This needs to be executed before ``self.run``."""
        cdef Py_ssize_t i
        for i in range(self.number_of_cells):
            self.cells[i][0] = random_int(size_x)
            self.cells[i][1] = random_int(size_y)
            self.cells[i][2] = random_int(size_z)
            self.cells[i][3] = self.sites[random_int(self.sites.shape[0])]
        return self.cells
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        """Modifies the provided arrays (a, x, y, z) for cells selected by
        ``self.get_random_cells``."""
        return
    cpdef void undo_last_run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        """Undoes the last modification done by ``self.run``.

        By default it just executes ``self.run`` again assuming the
        process is reversible, such as swapping."""
        self.run(a, x, y, z)

cdef class SwapOccupancy(BaseModifier):
    """Swap the atoms occupancy at swap_site between two cells."""
    def __init__(self, object sites):
        self.initialize_cells(2)
        self.initialize_sites(sites)
    def __str__(self):
        return  "{}(swap_sites={})".format(self.__class__.__name__,np.array(self.sites))
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        cdef cnp.int64_t tmp_a = a[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        a[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = a[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]]
        a[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]] = tmp_a

cdef class SwapDisplacement(BaseModifier):
    """Swap the atom displacement at swap_site between two cells."""
    def __init__(self, object sites):
        self.initialize_cells(2)
        self.initialize_sites(sites)
    def __str__(self):
        return  "{}(swap_sites={})".format(self.__class__.__name__,np.array(self.sites))
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        cdef double tmp_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        cdef double tmp_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        cdef double tmp_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = x[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]]
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = y[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]]
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = z[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]]
        x[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]] = tmp_x
        y[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]] = tmp_y
        z[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]] = tmp_z

cdef class Swap(BaseModifier):
    """Swap the atom occupancy and displacement at swap_site between two cells."""
    def __init__(self, object sites):
        self.initialize_cells(2)
        self.initialize_sites(sites)
    def __str__(self):
        return  "{}(swap_sites={})".format(self.__class__.__name__,np.array(self.sites))
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        cdef cnp.int64_t tmp_a = a[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        cdef double tmp_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        cdef double tmp_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        cdef double tmp_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        a[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = a[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = x[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]]
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = y[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]]
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = z[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]]
        a[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]] = tmp_a
        x[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]] = tmp_x
        y[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]] = tmp_y
        z[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.cells[1,3]] = tmp_z

cdef class ShiftDisplacementRange(BaseModifier):
    """Shifts the atoms displacement in all directions by a random amount
    in the given range."""
    cdef double minimum, maximum
    cdef double old_x, old_y, old_z
    def __init__(self, object sites, double minimum, double maximum):
        self.initialize_cells(1)
        self.initialize_sites(sites)
        self.minimum = minimum
        self.maximum = maximum
    def __str__(self):
        return  "{}(sites={},minimum={},maximum={})".format(self.__class__.__name__,np.array(self.sites),self.minimum,self.maximum)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_range(self.minimum, self.maximum)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_range(self.minimum, self.maximum)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_range(self.minimum, self.maximum)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_z

cdef class ShiftDisplacementNormal(BaseModifier):
    """Shifts the atoms displacement in all directions by a random amount
    in the normal distribution given by ``mu`` and ``sigma``.
    """
    cdef double mu, sigma
    cdef double old_x, old_y, old_z
    def __init__(self, object sites, double mu, double sigma):
        self.initialize_cells(1)
        self.initialize_sites(sites)
        self.mu = mu
        self.sigma = sigma
    def __str__(self):
        return  "{}(sites={},mu={},sigma={})".format(self.__class__.__name__,np.array(self.sites),self.mu,self.sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_normal(self.mu, self.sigma)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_normal(self.mu, self.sigma)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_normal(self.mu, self.sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_z

cdef class SetDisplacementRange(BaseModifier):
    """Sets the atoms displacement in all directions to a random point in
    the given range."""
    cdef double minimum, maximum
    cdef double old_x, old_y, old_z
    def __init__(self, object sites, double minimum, double maximum):
        self.initialize_cells(1)
        self.initialize_sites(sites)
        self.minimum = minimum
        self.maximum = maximum
    def __str__(self):
        return  "{}(sites={},minimum={},maximum={})".format(self.__class__.__name__,np.array(self.sites),self.minimum,self.maximum)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_range(self.minimum, self.maximum)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_range(self.minimum, self.maximum)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_range(self.minimum, self.maximum)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_z

cdef class SetDisplacementNormal(BaseModifier):
    """Sets the atoms displacement in all directions to a random point in
    the normal distribution given by ``mu`` and ``sigma``."""
    cdef double mu, sigma
    cdef double old_x, old_y, old_z
    def __init__(self, object sites, double mu, double sigma):
        self.initialize_cells(1)
        self.initialize_sites(sites)
        self.mu = mu
        self.sigma = sigma
    def __str__(self):
        return  "{}(sites={},mu={},sigma={})".format(self.__class__.__name__,np.array(self.sites),self.mu,self.sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_normal(self.mu, self.sigma)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_normal(self.mu, self.sigma)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_normal(self.mu, self.sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_z

cdef class ShiftDisplacementRangeXYZ(BaseModifier):
    """Shifts the atoms displacement in all directions by a random amount
    in the given range for each direction"""
    cdef double x_min, x_max, y_min, y_max, z_min, z_max
    cdef double old_x, old_y, old_z
    def __init__(self, object sites, double x_min, double x_max, double y_min, double y_max, double z_min, double z_max):
        self.initialize_cells(1)
        self.initialize_sites(sites)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
    def __str__(self):
        return  "{}(sites={},min_x={},max_x={},min_y={},max_y={},min_z={},max_z={})".format(self.__class__.__name__,
                                                                                            np.array(self.sites),
                                                                                            self.x_min, self.x_max,
                                                                                            self.y_min, self.y_max,
                                                                                            self.z_min, self.z_max)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_range(self.x_min, self.x_max)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_range(self.y_min, self.y_max)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_range(self.z_min, self.z_max)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_z

cdef class ShiftDisplacementNormalXYZ(BaseModifier):
    """Shifts the atoms displacement in all directions by a random amount
    in the normal distribution given by ``mu`` and ``sigma`` for each direction.
    """
    cdef double x_mu, x_sigma, y_mu, y_sigma, z_mu, z_sigma
    cdef double old_x, old_y, old_z
    def __init__(self, object sites, double x_mu, double x_sigma, double y_mu, double y_sigma, double z_mu, double z_sigma):
        self.initialize_cells(1)
        self.initialize_sites(sites)
        self.x_mu = x_mu
        self.x_sigma = x_sigma
        self.y_mu = y_mu
        self.y_sigma = y_sigma
        self.z_mu = z_mu
        self.z_sigma = z_sigma
    def __str__(self):
        return  "{}(sites={},mu_x={},sigma_x={},mu_y={},sigma_y={},mu_z={},sigma_z={})".format(self.__class__.__name__,
                                                                                               np.array(self.sites),
                                                                                               self.x_mu, self.x_sigma,
                                                                                               self.y_mu, self.y_sigma,
                                                                                               self.z_mu, self.z_sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_normal(self.x_mu, self.x_sigma)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_normal(self.y_mu, self.y_sigma)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] += random_normal(self.z_mu, self.z_sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_z

cdef class SetDisplacementRangeXYZ(BaseModifier):
    """Sets the atoms displacement in all directions to a random point in
    the given range for each direction"""
    cdef double x_min, x_max, y_min, y_max, z_min, z_max
    cdef double old_x, old_y, old_z
    def __init__(self, object sites, double x_min, double x_max, double y_min, double y_max, double z_min, double z_max):
        self.initialize_cells(1)
        self.initialize_sites(sites)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
    def __str__(self):
        return  "{}(sites={},min_x={},max_x={},min_y={},max_y={},min_z={},max_z={})".format(self.__class__.__name__,
                                                                                            np.array(self.sites),
                                                                                            self.x_min, self.x_max,
                                                                                            self.y_min, self.y_max,
                                                                                            self.z_min, self.z_max)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_range(self.x_min, self.x_max)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_range(self.y_min, self.y_max)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_range(self.z_min, self.z_max)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_z

cdef class SetDisplacementNormalXYZ(BaseModifier):
    """Sets the atoms displacement in all directions to a random point in
    the normal distribution given by ``mu`` and ``sigma`` for each direction."""
    cdef double x_mu, x_sigma, y_mu, y_sigma, z_mu, z_sigma
    cdef double old_x, old_y, old_z
    def __init__(self, object sites, double x_mu, double x_sigma, double y_mu, double y_sigma, double z_mu, double z_sigma):
        self.initialize_cells(1)
        self.initialize_sites(sites)
        self.x_mu = x_mu
        self.x_sigma = x_sigma
        self.y_mu = y_mu
        self.y_sigma = y_sigma
        self.z_mu = z_mu
        self.z_sigma = z_sigma
    def __str__(self):
        return  "{}(sites={},mu_x={},sigma_x={},mu_y={},sigma_y={},mu_z={},sigma_z={})".format(self.__class__.__name__,
                                                                                               np.array(self.sites),
                                                                                               self.x_mu, self.x_sigma,
                                                                                               self.y_mu, self.y_sigma,
                                                                                               self.z_mu, self.z_sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_normal(self.x_mu, self.x_sigma)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_normal(self.y_mu, self.y_sigma)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = random_normal(self.z_mu, self.z_sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, cnp.int64_t[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.cells[0,3]] = self.old_z
