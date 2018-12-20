"""
========
energies
========

Custom energies can be created by inheriting from
:obj:`javelin.energies.Energy` and overriding the `evaluate` method. The
evaluate method must have the identical signature and this gives you
access to the origin and neighbor sites atom types and xyz's along
with the neighbor vector.

*For example*

.. code-block:: python

    class MyEnergy(Energy):
        def __init__(self, E=-1):
            self.E = E
        def evaluate(self,
                     a1, x1, y1, z1,
                     a2, x2, y2, z2,
                     neighbor_x, neighbor_y, neighbor_z):
        return self.E

This is slower than using compile classes by about a factor of 10. If
you are using IPython or Jupyter notebooks you can use Cython magic to
compile your own energies. You need load the Cython magic first
``%load_ext Cython``. Then *for example*

.. code-block:: cython

    %%cython
    from javelin.energies cimport Energy
    cdef class MyCythonEnergy(Energy):
        cdef double E
        def __init__(self, double E=-1):
            self.E = E
        cpdef double evaluate(self,
                              int a1, double x1, double y1, double z1,
                              int a2, double x2, double y2, double z2,
                              Py_ssize_t neighbor_x, Py_ssize_t neighbor_y, Py_ssize_t neighbor_z) except *:
        return self.E

"""

from libc.math cimport exp, sqrt, pow, INFINITY, NAN
cimport cython

cdef class Energy:
    """ This is the base energy class that all energies must inherit
    from. Inherited class should then override the evaluate method but
    keep the same function signature. This energy is always 0.

    >>> e = Energy()
    >>> e.evaluate(1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1)
    0.0
    """
    def __str__(self):
        return "{}()".format(self.__class__.__name__)
    cpdef double evaluate(self,
                          int a1, double x1, double y1, double z1,
                          int a2, double x2, double y2, double z2,
                          Py_ssize_t target_x, Py_ssize_t target_y, Py_ssize_t target_z) except *:
        """This just returns 0"""
        return 0

cdef class IsingEnergy(Energy):
    """
    You can either set the desired correlation which will
    automatically adjust the pair interation energy (J), or by setting
    the J directly.

    .. math::
        E_{occ} = \sum_{i} \sum_{n,n\\ne i} J_n \sigma_i \sigma_{i-n}

    The atom site occupancy is represented by Ising spin variables
    :math:`\sigma_i = \pm1`. :math:`\sigma = +1` is when a site is
    occupied by `atom1` and :math:`\sigma = +1` is for `atom2`.

    >>> e = IsingEnergy(13, 42, -1)  # J = -1 produces a positive correlation
    >>> e.atom1
    13
    >>> e.atom2
    42
    >>> e.J
    -1.0
    >>> e.evaluate(1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
    -0.0
    >>> e.evaluate(13, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0)
    -1.0
    >>> e.evaluate(42, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0)
    1.0
    >>> e.evaluate(42, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0)
    -1.0
    """
    cdef readonly int atom1, atom2
    cdef public double J
    cdef readonly double desired_correlation
    def __init__(self, int atom1, int atom2, double J=0, double desired_correlation = NAN):
        self.correlation_type = 1
        self.atom1 = atom1
        self.atom2 = atom2
        self.J = J
        self.desired_correlation = desired_correlation
    def __str__(self):
        return "{}(Atom1={},Atom2={},J={:.3})".format(self.__class__.__name__,self.atom1, self.atom2, self.J)
    cdef double sigma(self, int atom):
        if atom == self.atom1:
            return -1
        elif atom == self.atom2:
            return 1
        else:
            return 0
    cpdef double evaluate(self,
                          int a1, double x1, double y1, double z1,
                          int a2, double x2, double y2, double z2,
                          Py_ssize_t target_x, Py_ssize_t target_y, Py_ssize_t target_z) except *:
        return self.J * self.sigma(a1) * self.sigma(a2)

cdef class DisplacementCorrelationEnergy(Energy):
    """
    Displacement disorder

    .. math::
        E_{dis} = \sum_{i} \sum_{n} J_n x_i x_{i-n}
    """
    cdef public double J
    cdef readonly double desired_correlation
    def __init__(self, double J=0, double desired_correlation = NAN):
        self.correlation_type = 2
        self.J = J
        self.desired_correlation = desired_correlation
    cpdef double evaluate(self,
                          int a1, double x1, double y1, double z1,
                          int a2, double x2, double y2, double z2,
                          Py_ssize_t target_x, Py_ssize_t target_y, Py_ssize_t target_z) except *:
        cdef double norm = (x1*x1 + y1*y1 + z1*z1) * (x2*x2 + y2*y2 + z2*z2)
        if norm == 0:
            return 0
        else:
            return self.J * (x1*x2 + y1*y2 + z1*z2) / sqrt(norm)

cdef class SpringEnergy(Energy):
    """
    Hook's law (Spring)

    .. math::
        E_{spring} = \sum_{i} \sum_{n} k_n [d_{in} - \\tau_{in}]^2
    """
    cdef readonly double K, desired
    def __init__(self, double K, double desired):
        self.K = K
        self.desired = desired
        """
        desired separation distance
        """
    cpdef double evaluate(self,
                          int a1, double x1, double y1, double z1,
                          int a2, double x2, double y2, double z2,
                          Py_ssize_t target_x, Py_ssize_t target_y, Py_ssize_t target_z) except *:
        cdef double diff = distance(x1, y1, z1,
                                    x2+target_x, y2+target_y, z2+target_z) - self.desired
        return self.K * diff*diff

cdef class LennardJonesEnergy(Energy):
    """
    Lennard-Jones potential

    .. math::
        E_{lj} = \sum_{i} \sum_{n\\ne i} \left[\\frac{A}{d_{in}^M} - \\frac{B}{d_{in}^N}\\right]

        \\text{where } A = \\frac{DN}{N-M}\\tau_{in}^M \\text{ and } B \\frac{DM}{N-M}\\tau_{in}^N

        \\text{default: }M=12, N=6
    """
    cdef readonly double D, desired
    cdef double M, N
    def __init__(self, double D, double desired, double M=12, double N=6):
        self.M = M
        self.N = N
        self.D = D
        self.desired = desired
    @cython.cdivision(True)
    cpdef double evaluate(self,
                          int a1, double x1, double y1, double z1,
                          int a2, double x2, double y2, double z2,
                          Py_ssize_t target_x, Py_ssize_t target_y, Py_ssize_t target_z) except *:
        cdef double d = distance(x1, y1, z1,
                                 x2+target_x, y2+target_y, z2+target_z)
        return INFINITY if d == 0 else self.D * (pow(self.desired/d, self.M) - 2*pow(self.desired/d, self.N))

cdef double distance(double x1, double y1, double z1,
                     double x2, double y2, double z2):
    cdef double dX = x2 - x1
    cdef double dY = y2 - y1
    cdef double dZ = z2 - z1
    return sqrt( dX*dX + dY*dY + dZ*dZ )
