#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_get_max_threads() 1
#endif

static PyObject *fourier(PyObject *self, PyObject *args) {
  PyArrayObject *qxarg = NULL, *qyarg = NULL, *qzarg = NULL, *positionsarg = NULL, *outrealarg = NULL, *outimagarg = NULL;
  PyArrayObject *qx = NULL, *qy = NULL, *qz = NULL, *positions = NULL, *outreal = NULL, *outimag = NULL;

  double *qxd, *qyd, *qzd, *positionsd, *outreald, *outimagd;
  npy_intp *dims_q, dims_positions;
  int n, a, i, or, oi, natoms;
  double dot;

  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!",
                        &PyArray_Type, &qxarg,
                        &PyArray_Type, &qyarg,
                        &PyArray_Type, &qzarg,
                        &PyArray_Type, &positionsarg,
                        &PyArray_Type, &outrealarg,
                        &PyArray_Type, &outimagarg))
    return NULL;
  qx = PyArray_FROM_OTF(qxarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (qx == NULL)
    return NULL;
  qy = PyArray_FROM_OTF(qyarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (qy == NULL)
    return NULL;
  qz = PyArray_FROM_OTF(qzarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (qz == NULL)
    return NULL;
  positions = PyArray_FROM_OTF(positionsarg, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (positions == NULL)
    return NULL;

#if NPY_API_VERSION >= 0x0000000c
  outreal = PyArray_FROM_OTF(outrealarg, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
  outimag = PyArray_FROM_OTF(outimagarg, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
  outreal = PyArray_FROM_OTF(outrealarg, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
  outimag = PyArray_FROM_OTF(outimagarg, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif
  if (outreal == NULL || outimag == NULL)
    goto fail;
  /* code that makes use of arguments */
  /* You will probably need at least
       nd = PyArray_NDIM(<..>)    -- number of dimensions
       dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                     showing length in each dim.
       dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

       If an error occurs goto fail.
  */
  qxd = (double *)PyArray_DATA(qx);
  qyd = (double *)PyArray_DATA(qy);
  qzd = (double *)PyArray_DATA(qz);
  positionsd = (double *)PyArray_DATA(positions);
  outreald = (double *)PyArray_DATA(outreal);
  outimagd = (double *)PyArray_DATA(outimag);

  natoms = PyArray_DIMS(positions)[0];

#pragma omp parallel for private(i, a, n, dot)
  for(int i=0; i<PyArray_Size(qx); i++)
    {
      for(int a=0; a<natoms; a++)
        {
          n=a*3;
          dot = qxd[i]*positionsd[n] + qyd[i]*positionsd[n+1] + qzd[i]*positionsd[n+2];
          outreald[i] = outreald[i] + cos(dot);
          outimagd[i] = outimagd[i] + sin(dot);
        }
    }
      

  Py_DECREF(qx);
  Py_DECREF(qy);
  Py_DECREF(qz);
  Py_DECREF(positions);
#if NPY_API_VERSION >= 0x0000000c
  PyArray_ResolveWritebackIfCopy(outreal);
  PyArray_ResolveWritebackIfCopy(outimag);
#endif
  Py_DECREF(outreal);
  Py_DECREF(outimag);
  Py_INCREF(Py_None);
  return Py_None;

fail:
  Py_DECREF(qx);
  Py_DECREF(qy);
  Py_DECREF(qz);
  Py_DECREF(positions);
#if NPY_API_VERSION >= 0x0000000c
  PyArray_DiscardWritebackIfCopy(outreal);
  PyArray_DiscardWritebackIfCopy(outimag);
#endif
  Py_XDECREF(outreal);
  Py_XDECREF(outimag);
  return NULL;
}


static PyObject *fourier2(PyObject *self, PyObject *args) {
  PyArrayObject *py_xm = NULL, *py_uin = NULL, *py_vin = NULL, *py_win = NULL, *py_xat = NULL, *py_results_real = NULL, *py_results_imag = NULL, *py_cex_real = NULL, *py_cex_imag = NULL;
  PyArrayObject *xm = NULL, *uin = NULL, *vin = NULL, *win = NULL, *xat = NULL, *results_real = NULL, *results_imag = NULL, *cex_real = NULL, *cex_imag = NULL;

  double *xmd, *uind, *vind, *wind, *xatd, *results_reald, *results_imagd, *cex_reald, *cex_imagd;

  int n, i, j, k, A, numu, numv, numw;
  double i2pi64 = 64. * 65536.;
  double xarg0, xincu, xincv, xincw;
  int iarg0, iincu, iincv, iincw, iarg, iadd, address;

  printf("7\n");
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!",
                        &PyArray_Type, &py_xm,
                        &PyArray_Type, &py_uin,
                        &PyArray_Type, &py_vin,
                        &PyArray_Type, &py_win,
                        &PyArray_Type, &py_xat,
                        &PyArray_Type, &py_results_real,
                        &PyArray_Type, &py_results_imag,
                        &PyArray_Type, &py_cex_real,
                        &PyArray_Type, &py_cex_imag))
    return NULL;

  printf("8\n");

  xm = PyArray_FROM_OTF(py_xm, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (xm == NULL)
    return NULL;
  uin = PyArray_FROM_OTF(py_uin, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (uin == NULL)
    return NULL;
  vin = PyArray_FROM_OTF(py_vin, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (vin == NULL)
    return NULL;
  win = PyArray_FROM_OTF(py_win, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (win == NULL)
    return NULL;
  xat = PyArray_FROM_OTF(py_xat, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (xat == NULL)
    return NULL;
  cex_real = PyArray_FROM_OTF(py_cex_real, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (cex_real == NULL)
    return NULL;
  cex_imag = PyArray_FROM_OTF(py_cex_imag, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (cex_imag == NULL)
    return NULL;

#if NPY_API_VERSION >= 0x0000000c
  results_real = PyArray_FROM_OTF(py_results_real, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
  results_imag = PyArray_FROM_OTF(py_results_imag, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
  results_real = PyArray_FROM_OTF(py_results_real, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
  results_imag = PyArray_FROM_OTF(py_results_imag, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif
  if (results_real == NULL || results_imag == NULL)
    goto fail;
  /* code that makes use of arguments */
  /* You will probably need at least
       nd = PyArray_NDIM(<..>)    -- number of dimensions
       dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                     showing length in each dim.
       dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

       If an error occurs goto fail.
  */
  printf("9\n");
  xmd = (double *)PyArray_DATA(xm);
  uind = (double *)PyArray_DATA(uin);
  vind = (double *)PyArray_DATA(vin);
  wind = (double *)PyArray_DATA(win);
  xatd = (double *)PyArray_DATA(xat);
  results_reald = (double *)PyArray_DATA(results_real);
  results_imagd = (double *)PyArray_DATA(results_imag);
  cex_reald = (double *)PyArray_DATA(cex_real);
  cex_imagd = (double *)PyArray_DATA(cex_imag);
  printf("10\n");


/*
  natoms = PyArray_DIMS(positions)[0];

#pragma omp parallel for private(i, a, n, dot)
  for(int i=0; i<PyArray_Size(qx); i++)
    {
      for(int a=0; a<natoms; a++)
        {
          n=a*3;
          dot = qxd[i]*positionsd[n] + qyd[i]*positionsd[n+1] + qzd[i]*positionsd[n+2];
          outreald[i] = outreald[i] + cos(dot);
          outimagd[i] = outimagd[i] + sin(dot);
        }
    }
*/

  A = PyArray_SIZE(xat);
  numu = PyArray_SHAPE(results_real)[0];
  numv = PyArray_SHAPE(results_real)[1];
  numw = PyArray_SHAPE(results_real)[2];

  printf("11\n");
  for (n=0; n<A; n=n+3){
    address = 0;
    for (i=0; i<numu; i++){
      // Get initial argument to the exponent and increments along
      // the two axies 'u' and 'v'
      xarg0=  xmd[0]*xatd[n] +  xmd[1]*xatd[n+1] +  xmd[2]*xatd[n+2];
      xincu= uind[0]*xatd[n] + uind[1]*xatd[n+1] + uind[2]*xatd[n+2];
      xincv= vind[0]*xatd[n] + vind[1]*xatd[n+1] + vind[2]*xatd[n+2];
      xincw= wind[0]*xatd[n] + wind[1]*xatd[n+1] + wind[2]*xatd[n+2];

      // Convert to high precision integers (64*i2pi=2^20) ...
      iarg0=(int) i2pi64*( xarg0 - round(xarg0) );
      iincu=(int) i2pi64*( xincu - round(xincu) );
      iincv=(int) i2pi64*( xincv - round(xincv) );
      iincw=(int) i2pi64*( xincw - round(xincw) );
      iarg=iarg0;

      for (k=0; k<numw; k++){
        iarg = iarg0 + i*iincu + k*iincw;
        for (j=0; j<numv; j++){
          iadd = iarg >> 6;
          iadd = iadd % 65536;
          results_reald[address] = results_reald[address] + cex_reald[iadd];
          results_imagd[address] = results_imagd[address] + cex_imagd[iadd];
          iarg = iarg + iincv;
          address++;
        }
      }
    }
  }
  
  printf("12\n");

  Py_DECREF(xm);
  Py_DECREF(uin);
  Py_DECREF(vin);
  Py_DECREF(win);
  Py_DECREF(xat);
  Py_DECREF(cex_real);
  Py_DECREF(cex_imag);
#if NPY_API_VERSION >= 0x0000000c
  PyArray_ResolveWritebackIfCopy(results_real);
  PyArray_ResolveWritebackIfCopy(results_imag);
#endif
  Py_DECREF(results_real);
  Py_DECREF(results_imag);
  Py_INCREF(Py_None);
  return Py_None;

fail:
  Py_DECREF(xm);
  Py_DECREF(uin);
  Py_DECREF(vin);
  Py_DECREF(win);
  Py_DECREF(xat);
  Py_DECREF(cex_real);
  Py_DECREF(cex_imag);
#if NPY_API_VERSION >= 0x0000000c
  PyArray_ResolveWritebackIfCopy(results_real);
  PyArray_ResolveWritebackIfCopy(results_imag);
#endif
  Py_DECREF(results_real);
  Py_DECREF(results_imag);
  return NULL;
}


static PyMethodDef FourierMethods[] = {
    {"calc_fourier", fourier, METH_VARARGS, "Testing"},
    {"calc_fourier2", fourier2, METH_VARARGS, "Testing"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,
                                       "fouriertest",
                                       NULL,
                                       -1,
                                       FourierMethods,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL};

PyMODINIT_FUNC PyInit_fouriertest(void) {
  PyObject *m;
  m = PyModule_Create(&moduledef);
  if (!m) {
    return NULL;
  }
  import_array();
  return m;
}
