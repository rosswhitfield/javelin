from cython.parallel import parallel, prange
import cython

cdef extern from 'complex.h' nogil:
    double complex cexp(double complex)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_cython(double[:,:,:] qx,
                       double[:,:,:] qy,
                       double[:,:,:] qz,
                       double[:,:] atoms,
                       double complex[:,:,:] results):
    cdef double complex dot
    cdef int A = atoms.shape[0], I = results.shape[0], J = results.shape[1], K = results.shape[2]
    cdef int a, i, j, k
    for i in prange(I, nogil=True):
        for a in range(A):
            for j in range(J):
                for k in range(K):
                    dot = qx[i,j,k]*atoms[a,0] + qy[i,j,k]*atoms[a,1] + qz[i,j,k]*atoms[a,2]
                    results[i,j,k] = results[i,j,k] + cexp(dot*1j)
