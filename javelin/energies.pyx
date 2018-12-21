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
    """This is the base energy class that all energies must inherit
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
        """This function always returns a double of the energy calcuated. This
        base Energy class always returns 0.

        The evaluate method get passed the atom type (Z) and positions
        in the unit cell (x, y and z) of the two atoms to be compared,
        (atom1 is ``a1``, ``x1``, ``y1``, ``z1``, atom2 is ``a2``,
        ``x2``, ``y2``, ``z2``) and the neighbor vector (``target_x``,
        ``target_y``, ``target_z``) which is the number of unit cells
        that separate the two atoms to be compared
        """
        return 0

cdef class IsingEnergy(Energy):
    """You can either set the ``desired_correlation`` which will
    automatically adjust the pair interaction energy (J), or set the J
    directly.

    .. math::
        E_{occ} = \sum_{i} \sum_{n,n\\ne i} J_n \sigma_i \sigma_{i-n}

    The atom site occupancy is represented by Ising spin variables
    :math:`\sigma_i = \pm1`. :math:`\sigma = -1` is when a site is
    occupied by `atom1` and :math:`\sigma = +1` is for `atom2`.

    >>> e = IsingEnergy(13, 42, -1)  # J = -1 produces a positive correlation
    >>> e.atom1
    13
    >>> e.atom2
    42
    >>> e.J
    -1.0
    >>> e.evaluate(1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)  # a1=1, a2=1
    -0.0
    >>> e.evaluate(13, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0)  # a1=13, a2=13
    -1.0
    >>> e.evaluate(42, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0)  # a1=42, a2=13
    1.0
    >>> e.evaluate(42, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0)  # a1=42, a2=42
    -1.0
    """
    cdef readonly int atom1, atom2
    cdef public double J
    """Interaction energy, positive J will creates negative correlations
while negative J creates positive correlations"""
    cdef readonly double desired_correlation
    """The desired occupancy correlation, this will automatically adjusted
    the interaction energy (J) during the :obj:`javelin.mc.MC`
    execution to achieve the desired correlation. The starting J can
    also be specified"""
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
    """You can either set the ``desired_correlation`` which will
    automatically adjust the pair interaction energy (J), or set the J
    directly.

    .. math::
        E_{dis} = \sum_{i} \sum_{n} J_n x_i x_{i-n}

    >>> e = DisplacementCorrelationEnergy(-1)  # J = -1 produces a positive correlation
    >>> e.evaluate(0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0)  # x1=1, y2=1
    -0.0
    >>> e.evaluate(0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0)  # x1=1, x2=1
    -1.0
    >>> e.evaluate(0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0)  # x1=1, x2=-1
    1.0
    >>> e.evaluate(0, 1, 1, 1, 0, 1, -1, -1, 0, 0, 0)  # x1=1, y1=1, z1=1, x2=1, y2=-1, z2=-1
    0.3333333333333333
    """
    cdef public double J
    """Interaction energy, positive J will creates negative correlations
while negative J creates positive correlations"""
    cdef readonly double desired_correlation
    """The desired displacement correlation, this will automatically adjust
    the interaction energy (J) during the :obj:`javelin.mc.MC`
    execution to achieve the desired correlation. The starting J can
    also be specified"""
    def __init__(self, double J=0, double desired_correlation = NAN):
        self.correlation_type = 2
        self.J = J
        self.desired_correlation = desired_correlation
    @cython.cdivision(True)
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
    """`Hooke's law <https://en.wikipedia.org/wiki/Hooke%27s_law>`_ for a
    simple spring can be used to describe atoms joined together by
    springs and are in a harmonic potential.

    The spring energy is described by :math:`E_{spring}`

    .. math::
        E_{spring} = \sum_{i} \sum_{n} k_n [d_{in} - \\tau_{in}]^2

    :math:`k` is the force constant, :math:`d` is the distance between
    the atoms and :math:`\\tau` is the desired distance (minimum
    energy occurs at :math:`d = \\tau`). Distances, :math:`d` and
    :math:`\\tau` are in fractional coordinates.

    .. plot::

        x=np.linspace(0,2,100)
        plt.plot(x, (x-1)**2)
        plt.axvline(x=1, linestyle='--', linewidth=4, color='gray')
        plt.xlabel("Distance d (fractional units)")
        plt.ylabel("Energy")
        plt.title(r"Spring Energy for $\\tau=1$ and $k=1$")

    >>> e = SpringEnergy(1, 1)  # K = 1, desired=1
    >>> e.evaluate(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)  # target_x=1 ≡ d=1
    0.0
    >>> e.evaluate(0, 0, 0, 0, 0, 0.5, 0, 0, 1, 0, 0)  # x2=0.5, target_x=1 ≡ d=1.5
    0.25
    >>> e.evaluate(0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)  # x2=1, target_x=1 ≡ d=2
    1.0
    >>> e.evaluate(0, 0, 0, 0, 0, -0.5, 0, 0, 1, 0, 0)  # x2=-0.5, target_x=1 ≡ d=0.5
    0.25
    >>> e.evaluate(0, 0, 0, 0, 0, 0, 0.5, 0, 1, 0, 0)  # y2=0.5, target_x=1 ≡ d=1.118
    0.013932

    Optionally you can define a particular atom combination that only
    this energy will apply to. You can do this by setting
    ``atom_type1`` and ``atom_type2`` (must set both otherwise this is
    ignored). If the atoms that are currently being evaluated don't
    match then the energy will be 0. It is suggested to include
    energies for all possible atom combinations in the simulation. For
    example

    >>> e = SpringEnergy(K=1, desired=1, atom_type1=11, atom_type2=17)  # Na - Cl
    >>> e.evaluate(99, 0, 0, 0, 99, 0.5, 0, 0, 1, 0, 0)  # a1 = 99, a2 = 99
    0.0
    >>> e.evaluate(11, 0, 0, 0, 11, 0.5, 0, 0, 1, 0, 0)  # a1 = 11, a2 = 11
    0.0
    >>> e.evaluate(11, 0, 0, 0, 17, 0.5, 0, 0, 1, 0, 0)  # a1 = 11, a2 = 17
    0.25
    >>> e.evaluate(17, 0, 0, 0, 11, 0.5, 0, 0, 1, 0, 0)  # a1 = 17, a2 = 11
    0.25
    >>>
    >>> e = SpringEnergy(K=1, desired=1, atom_type1=17, atom_type2=17)  # Cl - Cl
    >>> e.evaluate(99, 0, 0, 0, 99, 0.5, 0, 0, 1, 0, 0)  # a1 = 99, a2 = 99
    0.0
    >>> e.evaluate(11, 0, 0, 0, 11, 0.5, 0, 0, 1, 0, 0)  # a1 = 11, a2 = 11
    0.0
    >>> e.evaluate(11, 0, 0, 0, 17, 0.5, 0, 0, 1, 0, 0)  # a1 = 11, a2 = 17
    0.0
    >>> e.evaluate(17, 0, 0, 0, 17, 0.5, 0, 0, 1, 0, 0)  # a1 = 17, a2 = 17
    0.25
    """
    cdef readonly double K, desired
    cdef readonly int atom_type1, atom_type2
    cdef unsigned char check_atom_types
    def __init__(self, double K, double desired, int atom_type1=-1, int atom_type2=-1):
        self.K = K
        self.desired = desired
        self.atom_type1 = atom_type1
        self.atom_type2 = atom_type2
        self.check_atom_types = self.atom_type1 != -1 or self.atom_type2 != -1
        """
        desired separation distance
        """
    def __str__(self):
        return "{}(K={},desired={},atoms={})".format(self.__class__.__name__,
                                                     self.K,
                                                     self.desired,
                                                     '{}-{}'.format(self.atom_type1,self.atom_type2)
                                                     if self.check_atom_types else 'all')
    cdef unsigned char valid_atoms(self, int atom1, int atom2):
        if atom1 == self.atom_type1 and atom2 == self.atom_type2:
            return True
        elif atom1 == self.atom_type2 and atom2 == self.atom_type1:
            return True
        else:
            return False
    cpdef double evaluate(self,
                          int a1, double x1, double y1, double z1,
                          int a2, double x2, double y2, double z2,
                          Py_ssize_t target_x, Py_ssize_t target_y, Py_ssize_t target_z) except *:
        if self.check_atom_types:
            if not self.valid_atoms(a1, a2):
                return 0
        cdef double diff = distance(x1, y1, z1,
                                    x2+target_x, y2+target_y, z2+target_z) - self.desired
        return self.K * diff*diff

cdef class LennardJonesEnergy(Energy):
    """The `Lennard-Jones potential
    <https://en.wikipedia.org/wiki/Lennard-Jones_potential>`_ is a
    more realistic potential than :obj:`javelin.energies.SpringEnergy`
    that takes into account the strong repulsion between atoms as a
    close distance.

    The Lennard-Jones potential is described by :math:`E_{lj}`

    .. math::
        E_{lj} = \sum_{i} \sum_{n\\ne i} D \left[\left(\\frac{\\tau_{in}}{d_{in}}\\right)^{12} - 2 \left(\\frac{\\tau_{in}}{d_{in}}\\right)^6\\right]

    D is the depth of the potential well, :math:`d` is the distance
    between the atoms and :math:`\\tau` is the desired distance
    (minimum energy occurs at :math:`d = \\tau`). Distances, :math:`d`
    and :math:`\\tau` are in fractional coordinates.

    .. plot::

        x=np.linspace(0.75,3,100)
        plt.plot(x, (1/x)**12 - 2*(1/x)**6)
        plt.ylim(-1,1)
        plt.axvline(x=1, linestyle='--', linewidth=4, color='gray')
        plt.xlabel("Distance d (fractional units)")
        plt.ylabel("Energy")
        plt.title(r"Lennard-Jones Energy for $\\tau=1$ and $D=1$")

    >>> e = LennardJonesEnergy(1, 1)  # D = 1, desired=1
    >>> e.evaluate(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)  # target_x=1 ≡ d=1
    -1.0
    >>> e.evaluate(0, 0, 0, 0, 0, 0.5, 0, 0, 1, 0, 0)  # x2=0.5, target_x=1 ≡ d=1.5
    -0.167876
    >>> e.evaluate(0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)  # x2=1, target_x=1 ≡ d=2
    -0.031006
    >>> e.evaluate(0, 0, 0, 0, 0, -0.5, 0, 0, 1, 0, 0)  # x2=-0.5, target_x=1 ≡ d=0.5
    3968.0
    >>> e.evaluate(0, 0, 0, 0, 0, 0, 0.5, 0, 1, 0, 0)  # y2=0.5, target_x=1 ≡ d=1.118
    -0.761856

    Optionally you can define a particular atom combination that only
    this energy will apply to. You can do this by setting
    ``atom_type1`` and ``atom_type2`` (must set both otherwise this is
    ignored). If the atoms that are currently being evaluated don't
    match then the energy will be 0. It is suggested to include
    energies for all possible atom combinations in the simulation. For
    example

    >>> e = LennardJonesEnergy(D=1, desired=1, atom_type1=11, atom_type2=17)  # Na - Cl
    >>> e.evaluate(99, 0, 0, 0, 99, 0.5, 0, 0, 1, 0, 0)  # a1 = 99, a2 = 99
    0.0
    >>> e.evaluate(11, 0, 0, 0, 11, 0.5, 0, 0, 1, 0, 0)  # a1 = 11, a2 = 11
    0.0
    >>> e.evaluate(11, 0, 0, 0, 17, 0.5, 0, 0, 1, 0, 0)  # a1 = 11, a2 = 17
    -0.167876
    >>> e.evaluate(17, 0, 0, 0, 11, 0.5, 0, 0, 1, 0, 0)  # a1 = 17, a2 = 11
    -0.167876
    >>>
    >>> e = LennardJonesEnergy(D=1, desired=1, atom_type1=17, atom_type2=17)  # Cl - Cl
    >>> e.evaluate(99, 0, 0, 0, 99, 0.5, 0, 0, 1, 0, 0)  # a1 = 99, a2 = 99
    0.0
    >>> e.evaluate(11, 0, 0, 0, 11, 0.5, 0, 0, 1, 0, 0)  # a1 = 11, a2 = 11
    0.0
    >>> e.evaluate(11, 0, 0, 0, 17, 0.5, 0, 0, 1, 0, 0)  # a1 = 11, a2 = 17
    0.0
    >>> e.evaluate(17, 0, 0, 0, 17, 0.5, 0, 0, 1, 0, 0)  # a1 = 17, a2 = 17
    -0.167876
    """
    cdef readonly double D, desired
    cdef readonly int atom_type1, atom_type2
    cdef unsigned char check_atom_types
    def __init__(self, double D, double desired, int atom_type1=-1, int atom_type2=-1):
        self.D = D
        self.desired = desired
        self.atom_type1 = atom_type1
        self.atom_type2 = atom_type2
        self.check_atom_types = self.atom_type1 != -1 or self.atom_type2 != -1
    def __str__(self):
        return "{}(D={},desired={},atoms={})".format(self.__class__.__name__,
                                                     self.D,
                                                     self.desired,
                                                     '{}-{}'.format(self.atom_type1,self.atom_type2)
                                                     if self.check_atom_types else 'all')
    cdef unsigned char valid_atoms(self, int atom1, int atom2):
        if atom1 == self.atom_type1 and atom2 == self.atom_type2:
            return True
        elif atom1 == self.atom_type2 and atom2 == self.atom_type1:
            return True
        else:
            return False
    @cython.cdivision(True)
    cpdef double evaluate(self,
                          int a1, double x1, double y1, double z1,
                          int a2, double x2, double y2, double z2,
                          Py_ssize_t target_x, Py_ssize_t target_y, Py_ssize_t target_z) except *:
        if self.check_atom_types:
            if not self.valid_atoms(a1, a2):
                return 0
        cdef double d = distance(x1, y1, z1,
                                 x2+target_x, y2+target_y, z2+target_z)
        return INFINITY if d == 0 else self.D * (pow(self.desired/d, 12) - 2*pow(self.desired/d, 6))

cdef double distance(double x1, double y1, double z1,
                     double x2, double y2, double z2):
    cdef double dX = x2 - x1
    cdef double dY = y2 - y1
    cdef double dZ = z2 - z1
    return sqrt( dX*dX + dY*dY + dZ*dZ )
