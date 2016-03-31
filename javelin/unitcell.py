import numpy as np


class UnitCell(object):
    """The UnitCell ojbect can be set with either 1, 3 or 6 parameters
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
        self.__G = np.matrix(np.eye(3))
        self.__Gstar = np.matrix(np.eye(3))
        if args:
            self.cell = args

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
            self.a = np.float(args[0])
            self.b = np.float(args[1])
            self.c = np.float(args[2])
            self.alpha = self.beta = self.gamma = np.radians(90)
        elif args.size == 6:
            self.a = np.float(args[0])
            self.b = np.float(args[1])
            self.c = np.float(args[2])
            self.alpha = np.radians(args[3])
            self.beta = np.radians(args[4])
            self.gamma = np.radians(args[5])
        else:
            raise ValueError("Invalid number of variables, unit cell unchanged")
        self.__calculateG()
        self.__calculateReciprocalLattice()

    @property
    def G(self):
        """Returns the metric tensor **G**"""
        return self.__G

    @property
    def Gstar(self):
        """Returns the reciprocal metric tensor **G***"""
        return self.G.getI()

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
        self.__G[0, 0] = self.a**2
        self.__G[1, 1] = self.b**2
        self.__G[2, 2] = self.c**2
        self.__G[0, 1] = self.a * self.b * np.cos(self.gamma)
        self.__G[0, 2] = self.a * self.c * np.cos(self.beta)
        self.__G[1, 2] = self.b * self.c * np.cos(self.alpha)
        self.__G[1, 0] = self.__G[0, 1]
        self.__G[2, 0] = self.__G[0, 2]
        self.__G[2, 1] = self.__G[1, 2]
