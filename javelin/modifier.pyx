"""
========
Modifier
========
"""

import numpy as np
cimport numpy as np
from .random cimport random_int, random_range, random_normal
cimport cython

cdef class BaseModifier:
    def __init__(self, int number_of_cells):
        self.initialize_cells(number_of_cells)
    def __str__(self):
        return  "{}(number_of_cells={})".format(self.__class__.__name__,self.number_of_cells)
    cdef void initialize_cells(self, int number_of_cells):
        self.number_of_cells = number_of_cells
        cdef np.ndarray[Py_ssize_t, ndim=2] cells = np.zeros((number_of_cells, 3), dtype=np.intp)
        self.cells = cells
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef Py_ssize_t[:,:] get_random_cells(self, Py_ssize_t size_x, Py_ssize_t size_y, Py_ssize_t size_z) except *:
        cdef Py_ssize_t i
        for i in range(self.number_of_cells):
            self.cells[i][0] = random_int(size_x)
            self.cells[i][1] = random_int(size_y)
            self.cells[i][2] = random_int(size_z)
        return self.cells
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        return
    cpdef void undo_last_run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.run(a, x, y, z)

cdef class SwapOccupancy(BaseModifier):
    cdef Py_ssize_t swap_site
    def __init__(self, Py_ssize_t swap_site):
        self.initialize_cells(2)
        self.swap_site = swap_site
    def __str__(self):
        return  "{}(swap_site={})".format(self.__class__.__name__,self.swap_site)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        cdef long tmp_a = a[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site]
        a[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site] = a[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site]
        a[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site] = tmp_a

cdef class SwapDisplacement(BaseModifier):
    cdef Py_ssize_t swap_site
    def __init__(self, Py_ssize_t swap_site):
        self.initialize_cells(2)
        self.swap_site = swap_site
    def __str__(self):
        return  "{}(swap_site={})".format(self.__class__.__name__,self.swap_site)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        cdef double tmp_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site]
        cdef double tmp_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site]
        cdef double tmp_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site] = x[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site]
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site] = y[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site]
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site] = z[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site]
        x[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site] = tmp_x
        y[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site] = tmp_y
        z[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site] = tmp_z

cdef class Swap(BaseModifier):
    cdef Py_ssize_t swap_site
    def __init__(self, Py_ssize_t swap_site):
        self.initialize_cells(2)
        self.swap_site = swap_site
    def __str__(self):
        return  "{}(swap_site={})".format(self.__class__.__name__,self.swap_site)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        cdef long tmp_a = a[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site]
        cdef double tmp_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site]
        cdef double tmp_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site]
        cdef double tmp_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site]
        a[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site] = a[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site] = x[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site]
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site] = y[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site]
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.swap_site] = z[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site]
        a[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site] = tmp_a
        x[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site] = tmp_x
        y[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site] = tmp_y
        z[self.cells[1,0], self.cells[1,1], self.cells[1,2], self.swap_site] = tmp_z

cdef class ShiftDisplacementRange(BaseModifier):
    cdef Py_ssize_t site
    cdef double minimum, maximum
    cdef double old_x, old_y, old_z
    def __init__(self, Py_ssize_t site, double minimum, double maximum):
        self.initialize_cells(1)
        self.site = site
        self.minimum = minimum
        self.maximum = maximum
    def __str__(self):
        return  "{}(site={},minimum={},maximum={})".format(self.__class__.__name__,self.site,self.minimum,self.maximum)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_range(self.minimum, self.maximum)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_range(self.minimum, self.maximum)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_range(self.minimum, self.maximum)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_z

cdef class ShiftDisplacementNormal(BaseModifier):
    cdef Py_ssize_t site
    cdef double mu, sigma
    cdef double old_x, old_y, old_z
    def __init__(self, Py_ssize_t site, double mu, double sigma):
        self.initialize_cells(1)
        self.site = site
        self.mu = mu
        self.sigma = sigma
    def __str__(self):
        return  "{}(site={},mu={},sigma={})".format(self.__class__.__name__,self.site,self.mu,self.sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_normal(self.mu, self.sigma)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_normal(self.mu, self.sigma)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_normal(self.mu, self.sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_z

cdef class SetDisplacementRange(BaseModifier):
    cdef Py_ssize_t site
    cdef double minimum, maximum
    cdef double old_x, old_y, old_z
    def __init__(self, Py_ssize_t site, double minimum, double maximum):
        self.initialize_cells(1)
        self.site = site
        self.minimum = minimum
        self.maximum = maximum
    def __str__(self):
        return  "{}(site={},minimum={},maximum={})".format(self.__class__.__name__,self.site,self.minimum,self.maximum)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_range(self.minimum, self.maximum)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_range(self.minimum, self.maximum)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_range(self.minimum, self.maximum)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_z

cdef class SetDisplacementNormal(BaseModifier):
    cdef Py_ssize_t site
    cdef double mu, sigma
    cdef double old_x, old_y, old_z
    def __init__(self, Py_ssize_t site, double mu, double sigma):
        self.initialize_cells(1)
        self.site = site
        self.mu = mu
        self.sigma = sigma
    def __str__(self):
        return  "{}(site={},mu={},sigma={})".format(self.__class__.__name__,self.site,self.mu,self.sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_normal(self.mu, self.sigma)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_normal(self.mu, self.sigma)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_normal(self.mu, self.sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_z

cdef class ShiftDisplacementRangeXYZ(BaseModifier):
    cdef Py_ssize_t site
    cdef double x_min, x_max, y_min, y_max, z_min, z_max
    cdef double old_x, old_y, old_z
    def __init__(self, Py_ssize_t site, double x_min, double x_max, double y_min, double y_max, double z_min, double z_max):
        self.initialize_cells(1)
        self.site = site
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
    def __str__(self):
        return  "{}(site={},min_x={},max_x={},min_y={},max_y={},min_z={},max_z={})".format(self.__class__.__name__, self.site,
                                                                                           self.x_min, self.x_max,
                                                                                           self.y_min, self.y_max,
                                                                                           self.z_min, self.z_max)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_range(self.x_min, self.x_max)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_range(self.y_min, self.y_max)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_range(self.z_min, self.z_max)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_z

cdef class ShiftDisplacementNormalXYZ(BaseModifier):
    cdef Py_ssize_t site
    cdef double x_mu, x_sigma, y_mu, y_sigma, z_mu, z_sigma
    cdef double old_x, old_y, old_z
    def __init__(self, Py_ssize_t site, double x_mu, double x_sigma, double y_mu, double y_sigma, double z_mu, double z_sigma):
        self.initialize_cells(1)
        self.site = site
        self.x_mu = x_mu
        self.x_sigma = x_sigma
        self.y_mu = y_mu
        self.y_sigma = y_sigma
        self.z_mu = z_mu
        self.z_sigma = z_sigma
    def __str__(self):
        return  "{}(site={},mu_x={},sigma_x={},mu_y={},sigma_y={},mu_z={},sigma_z={})".format(self.__class__.__name__, self.site,
                                                                                              self.x_mu, self.x_sigma,
                                                                                              self.y_mu, self.y_sigma,
                                                                                              self.z_mu, self.z_sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_normal(self.x_mu, self.x_sigma)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_normal(self.y_mu, self.y_sigma)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] += random_normal(self.z_mu, self.z_sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_z

cdef class SetDisplacementRangeXYZ(BaseModifier):
    cdef Py_ssize_t site
    cdef double x_min, x_max, y_min, y_max, z_min, z_max
    cdef double old_x, old_y, old_z
    def __init__(self, Py_ssize_t site, double x_min, double x_max, double y_min, double y_max, double z_min, double z_max):
        self.initialize_cells(1)
        self.site = site
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
    def __str__(self):
        return  "{}(site={},min_x={},max_x={},min_y={},max_y={},min_z={},max_z={})".format(self.__class__.__name__, self.site,
                                                                                           self.x_min, self.x_max,
                                                                                           self.y_min, self.y_max,
                                                                                           self.z_min, self.z_max)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_range(self.x_min, self.x_max)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_range(self.y_min, self.y_max)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_range(self.z_min, self.z_max)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_z

cdef class SetDisplacementNormalXYZ(BaseModifier):
    cdef Py_ssize_t site
    cdef double x_mu, x_sigma, y_mu, y_sigma, z_mu, z_sigma
    cdef double old_x, old_y, old_z
    def __init__(self, Py_ssize_t site, double x_mu, double x_sigma, double y_mu, double y_sigma, double z_mu, double z_sigma):
        self.initialize_cells(1)
        self.site = site
        self.x_mu = x_mu
        self.x_sigma = x_sigma
        self.y_mu = y_mu
        self.y_sigma = y_sigma
        self.z_mu = z_mu
        self.z_sigma = z_sigma
    def __str__(self):
        return  "{}(site={},mu_x={},sigma_x={},mu_y={},sigma_y={},mu_z={},sigma_z={})".format(self.__class__.__name__, self.site,
                                                                                              self.x_mu, self.x_sigma,
                                                                                              self.y_mu, self.y_sigma,
                                                                                              self.z_mu, self.z_sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        self.old_x = x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_y = y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        self.old_z = z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site]
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_normal(self.x_mu, self.x_sigma)
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_normal(self.y_mu, self.y_sigma)
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = random_normal(self.z_mu, self.z_sigma)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cpdef void undo_last_run(self, long[:,:,:,:] a, double[:,:,:,:] x, double[:,:,:,:] y, double[:,:,:,:] z) except *:
        x[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_x
        y[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_y
        z[self.cells[0,0], self.cells[0,1], self.cells[0,2], self.site] = self.old_z
