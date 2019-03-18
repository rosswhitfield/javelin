cimport numpy as cnp

cdef class BaseModifier:
    cdef readonly int number_of_cells
    cdef readonly Py_ssize_t[:,:] cells
    cdef readonly Py_ssize_t[:] sites
    cpdef Py_ssize_t[:,:] get_random_cells(self, Py_ssize_t, Py_ssize_t, Py_ssize_t) except *
    cdef void initialize_cells(self, int)
    cdef void initialize_sites(self, object)
    cpdef void run(self, cnp.int64_t[:,:,:,:], double[:,:,:,:], double[:,:,:,:], double[:,:,:,:]) except *
    cpdef void undo_last_run(self, cnp.int64_t[:,:,:,:], double[:,:,:,:], double[:,:,:,:], double[:,:,:,:]) except *
