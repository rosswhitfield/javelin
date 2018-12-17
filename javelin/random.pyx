from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport cos, log, sqrt, M_PI
cimport cython

@cython.cdivision(True)
cdef double random():
    return <double>rand()/<double>RAND_MAX

@cython.cdivision(True)
cdef Py_ssize_t random_int(Py_ssize_t max):
    return <Py_ssize_t>(<double>rand() / (<double>RAND_MAX+1) * max)


@cython.cdivision(True)
cdef double random_range(double min, double max):
    return min + <double>rand() / (<double>RAND_MAX) * (max - min)

@cython.cdivision(True)
cdef double random_normal(double mu=0, double sigma=1):
    """
    The Box–Muller method for generating values that are normally distributed
    """
    return sqrt(-2. * log(random())) * cos(2*M_PI * random()) * sigma + mu

cpdef void set_seed(int seed=42):
    """
    For testing, allows setting the seed consistent results
    """
    srand(seed)
