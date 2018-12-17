cdef class BaseModifier:
    cdef readonly int number_of_cells
    cdef readonly Py_ssize_t[:,:] cells
    cpdef Py_ssize_t[:,:] get_random_cells(self, Py_ssize_t, Py_ssize_t, Py_ssize_t) except *
    cdef void initialize_cells(self, int)
    cpdef void run(self, long[:,:,:,:], double[:,:,:,:], double[:,:,:,:], double[:,:,:,:]) except *
    cpdef void undo_last_run(self, long[:,:,:,:], double[:,:,:,:], double[:,:,:,:], double[:,:,:,:]) except *
