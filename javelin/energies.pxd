cdef class Energy:
    cdef readonly int correlation_type # for feedback, 1 = occupancy, 2 = displacement
    cpdef double evaluate(self,
                          int, double, double, double,
                          int, double, double, double,
                          Py_ssize_t, Py_ssize_t, Py_ssize_t) except *
