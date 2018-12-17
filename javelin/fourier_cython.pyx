from cython.parallel import parallel, prange
from libc.math cimport sin, cos, round
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_cython(double[:,:,:] qx,
                       double[:,:,:] qy,
                       double[:,:,:] qz,
                       double[:,:] atoms,
                       double[:,:,:] results_real,
                       double[:,:,:] results_imag):

    cdef double dot
    cdef Py_ssize_t A = atoms.shape[0], I = results_real.shape[0], J = results_real.shape[1], K = results_real.shape[2]
    cdef Py_ssize_t a, i, j, k
    with nogil:
        for i in prange(I):
            for a in range(A):
                for j in range(J):
                    for k in range(K):
                        dot = qx[i,j,k]*atoms[a,0] + qy[i,j,k]*atoms[a,1] + qz[i,j,k]*atoms[a,2]
                        results_real[i,j,k] = results_real[i,j,k] + cos(dot)
                        results_imag[i,j,k] = results_imag[i,j,k] + sin(dot)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef approx_calculate_cython(double[:] xm,
                              double[:] uin,
                              double[:] vin,
                              double[:] win,
                              double[:,:] xat,
                              double[:,:,:] results_real,
                              double[:,:,:] results_imag,
                              double[:] cex_real,
                              double[:] cex_imag):
    """
    This is adapted from the DIFFUSE program
    Butler, B. D. & Welberry T. R. (1992). 3. Appl. Cryst. 25, 391-399.
    """

    cdef int A = xat.shape[0], numu = results_real.shape[0], numv = results_real.shape[1], numw = results_real.shape[2]
    cdef int n, i, j, k

    cdef double i2pi64 = 64. * 65536.
    cdef double xarg0 = 0.;
    cdef double xincu = 0.;
    cdef double xincv = 0.;
    cdef double xincw = 0.;
    cdef int iarg0 = 0;
    cdef int iincu = 0;
    cdef int iincv = 0;
    cdef int iincw = 0;
    cdef int iarg = 0;
    cdef int iadd = 0;

    with nogil:
        for i in prange(numu):
            for n in range(A):
                # Get initial argument to the exponent and increments along
                # the two axies 'u' and 'v'
                xarg0=  xm[0]*xat[n,0] +  xm[1]*xat[n,1] +  xm[2]*xat[n,2]
                xincu= uin[0]*xat[n,0] + uin[1]*xat[n,1] + uin[2]*xat[n,2]
                xincv= vin[0]*xat[n,0] + vin[1]*xat[n,1] + vin[2]*xat[n,2]
                xincw= win[0]*xat[n,0] + win[1]*xat[n,1] + win[2]*xat[n,2]

                # Convert to high precision integers (64*i2pi=2^20) ...
                iarg0=int( i2pi64*( xarg0 - round(xarg0) ) )
                iincu=int( i2pi64*( xincu - round(xincu) ) )
                iincv=int( i2pi64*( xincv - round(xincv) ) )
                iincw=int( i2pi64*( xincw - round(xincw) ) )
                iarg=iarg0

                for k in range(numw):
                    iarg = iarg0 + i*iincu + k*iincw
                    for j in range(numv):
                        iadd = iarg >> 6
                        iadd = iadd % 65536
                        results_real[i,j,k] = results_real[i,j,k] + cex_real[iadd]
                        results_imag[i,j,k] = results_imag[i,j,k] + cex_imag[iadd]
                        iarg = iarg + iincv
