from cython.parallel import parallel, prange
import cython
from libc.math cimport sin, cos

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_cython(double[:,:,:] qx,
                       double[:,:,:] qy,
                       double[:,:,:] qz,
                       double[:,:] atoms,
                       double[:,:,:] results_real,
                       double[:,:,:] results_imag):

    cdef double dot
    cdef int A = atoms.shape[0], I = results_real.shape[0], J = results_real.shape[1], K = results_real.shape[2]
    cdef int a, i, j, k
    for i in prange(I, nogil=True):
        for a in range(A):
            for j in range(J):
                for k in range(K):
                    dot = qx[i,j,k]*atoms[a,0] + qy[i,j,k]*atoms[a,1] + qz[i,j,k]*atoms[a,2]
                    results_real[i,j,k] = results_real[i,j,k] + cos(dot)
                    results_imag[i,j,k] = results_imag[i,j,k] + sin(dot)
