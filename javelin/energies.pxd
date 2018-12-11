cdef class Energy:
    cpdef double evaluate(self,
                          int, double, double, double,
                          int, double, double, double,
                          Py_ssize_t, Py_ssize_t, Py_ssize_t) except *
