"""
========
unitcell
========
"""

import numpy as np


class UnitCell:
    """The UnitCell object can be set with either 1, 3 or 6 parameters
    corresponding to cubic ``a`` parameters, ``(a, b, c)`` or ``(a, b,
    c, alpha, beta, gamma)``, where angles are in degrees.

    >>> cubic = UnitCell(5)
    >>> cubic.cell
    (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)

    >>> orthorhombic = UnitCell(5, 6, 7)
    >>> orthorhombic.cell
    (5.0, 6.0, 7.0, 90.0, 90.0, 90.0)

    >>> unitcell = UnitCell(4.0, 3.0, 6.0, 89.0, 90.0, 97.0)
    >>> unitcell.cell
    (4.0, 3.0, 6.0, 89.0, 90.0, 97.0)

    UnitCell objects can be set after being created simply by

    >>> unitcell = UnitCell()
    >>> unitcell.cell = 6
    >>> unitcell.cell
    (6.0, 6.0, 6.0, 90.0, 90.0, 90.0)
    >>> unitcell.cell = 3, 4, 5
    >>> unitcell.cell
    (3.0, 4.0, 5.0, 90.0, 90.0, 90.0)
    >>> unitcell.cell = 6, 7, 8, 91.0, 90, 89
    >>> unitcell.cell
    (6.0, 7.0, 8.0, 91.0, 90.0, 89.0)
    >>> # or using a list or tuple
    >>> unitcell.cell = [8, 7, 6, 89, 90, 90]
    >>> unitcell.cell
    (8.0, 7.0, 6.0, 89.0, 90.0, 90.0)

    """
    def __init__(self, *args):
        self.a = 1
        self.b = 1
        self.c = 1
        self.alpha = np.radians(90)
        self.beta = np.radians(90)
        self.gamma = np.radians(90)
        self.ra = 1  # a*
        self.rb = 1  # b*
        self.rc = 1  # c*
        self.ralpha = np.radians(90)  # alpha*
        self.rbeta = np.radians(90)   # beta*
        self.rgamma = np.radians(90)  # gamma*
        self.__G = np.eye(3)
        self.__Gstar = np.eye(3)
        self.__B = np.eye(3)
        if args:
            self.cell = args

    def __eq__(self, other):
        return self.cell == other.cell

    def __repr__(self):
        return "a={}, b={}, c={}, alpha={}, beta={}, gamma={}".format(*self.cell)

    def cartesian(self, u):
        """Return Cartesian coordinates of a lattice vector.

        >>> unitcell = UnitCell(3,4,5,90,90,120)
        >>> unitcell.cartesian([1,0,0])
        array([  2.59807621e+00,  -1.50000000e+00,   3.25954010e-16])

        A array of atoms position can also be passed

        >>> positions = [[1,0,0], [0,0,0.5]]
        >>> unitcell.cartesian(positions)
        array([[  2.59807621e+00,  -1.50000000e+00,   3.25954010e-16],
               [  0.00000000e+00,   0.00000000e+00,   2.50000000e+00]])
        """
        return np.dot(u, self.Binv)

    @property
    def cell(self):
        """Return the unit cell parameters (*a*, *b*, *c*, *alpha*, *beta*,
        *gamma*) in degrees.

        """
        return (self.a, self.b, self.c,
                np.degrees(self.alpha),
                np.degrees(self.beta),
                np.degrees(self.gamma))

    @cell.setter
    def cell(self, *args):
        """Sets the unit cell with either 1, 3 or 6 parameters corresponding
        to cubic ``a`` parameters, ``(a, b, c)`` or ``(a, b, c, alpha,
        beta, gamma)``, where angles are in degrees

        """
        args = np.asarray(args).flatten()
        if args.size == 1:  # cubic
            self.a = self.b = self.c = np.float(args)
            self.alpha = self.beta = self.gamma = np.radians(90)
        elif args.size == 3:  # orthorhombic
            a, b, c = args[0:3]
            self.a = 1 if a == 0 else np.float(a)
            self.b = 1 if b == 0 else np.float(b)
            self.c = 1 if c == 0 else np.float(c)
            self.alpha = self.beta = self.gamma = np.radians(90)
        elif args.size == 6:
            a, b, c = args[0:3]
            self.a = 1 if a == 0 else np.float(a)
            self.b = 1 if b == 0 else np.float(b)
            self.c = 1 if c == 0 else np.float(c)
            self.alpha = np.radians(args[3])
            self.beta = np.radians(args[4])
            self.gamma = np.radians(args[5])
        else:
            raise ValueError("Invalid number of variables, unit cell unchanged")
        self.__calculateG()
        self.__calculateReciprocalLattice()
        self.__calculateB()

    def fractional(self, u):
        """Return Cartesian coordinates of a lattice vector.

        >>> unitcell = UnitCell(3,4,5,90,90,120)
        >>> unitcell.fractional([0,4,0])
        array([  0.00000000e+00,   1.00000000e+00,  -4.89858720e-17])

        A array of atoms position can also be passed

        >>> positions = [[0,2,0], [0,0,5]]
        >>> unitcell.fractional(positions)
        array([[  0.00000000e+00,   5.00000000e-01,  -2.44929360e-17],
               [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
        """
        return (u @ self.B)

    @property
    def G(self):
        """Returns the metric tensor **G**"""
        return self.__G

    @property
    def Gstar(self):
        """Returns the reciprocal metric tensor **G***"""
        return np.linalg.inv(self.G)

    @property
    def B(self):
        """Returns the **B** matrix"""
        return self.__B

    @property
    def Binv(self):
        """Returns the inverse **B** matrix"""
        return np.linalg.inv(self.B)

    def dstar(self, h, k, l):
        """Returns d*=1/d for given h,k,l"""
        return np.linalg.norm(self.B @ np.array([[h],
                                                 [k],
                                                 [l]]))

    def d(self, h, k, l):
        """Returns d-spacing for given h,k,l"""
        return 1/self.dstar(h, k, l)

    def recAngle(self, h1, k1, l1, h2, k2, l2, degrees=False):
        """Calculates the angle between two reciprocal vectors"""
        q1 = np.array([[h1], [k1], [l1]])
        q2 = np.array([[h2], [k2], [l2]])
        q1 = self.Gstar @ q1
        E = (q1.T @ q2).sum()
        angle = np.arccos(E / (self.dstar(h1, k1, l1) * self.dstar(h2, k2, l2)))
        if degrees:
            return np.degrees(angle)
        else:
            return angle

    @property
    def volume(self):
        """Returns the unit cell volume"""
        return np.sqrt(np.linalg.det(self.__G))

    @property
    def reciprocalVolume(self):
        """Returns the unit cell reciprocal volume"""
        return np.sqrt(np.linalg.det(self.Gstar))

    @property
    def reciprocalCell(self):
        """Return the reciprocal unit cell parameters (*a**, *b**, *c**,
        *alpha**, *beta**, *gamma**) in degrees.

        """
        return (self.ra, self.rb, self.rc,
                np.degrees(self.ralpha),
                np.degrees(self.rbeta),
                np.degrees(self.rgamma))

    def __calculateReciprocalLattice(self):
        """Calculates the reciropcal lattice from G*"""
        Gstar = self.Gstar
        self.ra = np.sqrt(Gstar[0, 0])
        self.rb = np.sqrt(Gstar[1, 1])
        self.rc = np.sqrt(Gstar[2, 2])
        self.ralpha = np.arccos(Gstar[1, 2] / (self.rb*self.rc))
        self.rbeta = np.arccos(Gstar[0, 2] / (self.ra*self.rc))
        self.rgamma = np.arccos(Gstar[0, 1] / (self.ra*self.rb))

    def __calculateG(self):
        """Calculates the metric tensor from unti cell parameters"""
        if ((self.alpha > self.beta + self.gamma) or
                (self.beta > self.alpha + self.gamma) or
                (self.gamma > self.alpha + self.beta)):
            raise ValueError("Invalid angles")
        ca = np.cos(self.alpha)
        cb = np.cos(self.beta)
        cg = np.cos(self.gamma)
        self.__G = np.array([[self.a**2,            self.a * self.b * cg, self.a * self.c * cb],
                             [self.a * self.b * cg, self.b**2,            self.b * self.c * ca],
                             [self.a * self.c * cb, self.b * self.c * ca, self.c**2]])

    def __calculateB(self):
        """Calculated B matrix from lattice vectors"""
        self.__B = np.array([[self.ra, self.rb * np.cos(self.rgamma),
                              self.rc * np.cos(self.rbeta)],
                             [0, self.rb * np.sin(self.rgamma),
                              - self.rc * np.sin(self.rbeta) * np.cos(self.alpha)],
                             [0,        0, 1/self.c]])
