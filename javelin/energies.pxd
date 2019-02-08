cdef class Energy:
    cdef readonly int correlation_type
    """This is used when feedback is applied, 0 = no_correlations, 1 =
    occupancy, 2 = displacement. The ``desired_correlation`` must also
    be set on the energy for this to work."""
    cpdef double run(self,
                     long[:,:,:,::1], double[:,:,:,::1], double[:,:,:,::1], double[:,:,:,::1],
                     Py_ssize_t[:],
                     Py_ssize_t[:,:],  Py_ssize_t,
                     Py_ssize_t, Py_ssize_t, Py_ssize_t) except *
    cpdef double evaluate(self,
                          int, double, double, double,
                          int, double, double, double,
                          Py_ssize_t, Py_ssize_t, Py_ssize_t) except *
