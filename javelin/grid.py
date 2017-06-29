"""
====
grid
====

Grid class to allow the Q-space grid to be definied in different
ways Should allow for corners and bins, or a matrix and the axis to be
defined.
"""

from __future__ import division
import numpy as np


class Grid(object):
    """Grid object
    """
    def __init__(self):
        # vectors of grid
        self._v1 = np.array([1, 0, 0])
        self._v2 = np.array([0, 1, 0])
        self._v3 = np.array([0, 0, 1])
        # min max of each vector
        self._r1 = np.array([0, 1])
        self._r2 = np.array([0, 1])
        self._r3 = np.array([0, 1])
        # number of bins in each direction
        self.bins = (101, 101, 1)

        self.units = 'r.l.u'

    def __str__(self):
        return """lower left  corner :     {}
lower right corner :     {}
upper left  corner :     {}
top   left  corner :     {}

hor. increment     :     {}
vert. increment    :     {}
top   increment    :     {}

# of points        :     {} x {} x {}""".format(self.ll, self.lr, self.ul, self.tl,
                                                self.v1_delta, self.v2_delta, self.v3_delta,
                                                *self.bins)

    def set_corners(self,
                    ll=(0, 0, 0),
                    lr=None,
                    ul=None,
                    tl=None):
        """Define the axis vectors by the corners of the reciprocal
        volume. The corners values with be converted into axis
        vectors, see :func:`javelin.grid.corners_to_vectors` for
        details.

        :param ll: lower-left corner
        :type ll: array-like of length 3
        :param lr: lower-right corner
        :type lr: array-like of length 3
        :param ul: upper-left corner
        :type ul: array-like of length 3
        :param tl: top-left corner
        :type tl: array-like of length 3

        """
        self.v1, self.v2, self.v3, self.r1, self.r2, self.r3 = corners_to_vectors(ll, lr, ul, tl)

    @property
    def bins(self):
        """The number of bins in each direction

        >>> grid = Grid()
        >>> grid.bins
        (101, 101, 1)

        >>> grid.bins = 5
        >>> grid.bins
        (5, 1, 1)

        >>> grid.bins = 5, 6
        >>> grid.bins
        (5, 6, 1)

        >>> grid.bins = 5, 6, 7
        >>> grid.bins
        (5, 6, 7)

        :getter: Returns the number of bins in each direction
        :setter: Sets the number of bins, provide 1, 2 or 3 integers
        :type: :class:`numpy.ndarray` of int
        """
        return self._n1, self._n2, self._n3

    @bins.setter
    def bins(self, dims):
        dims = np.asarray(dims, dtype=int)
        if dims.size == 1:
            self._n1 = int(dims)  # abscissa  (lr - ll)
            self._n2 = 1
            self._n3 = 1
        elif dims.size == 2:
            self._n1 = dims[0]  # abscissa  (lr - ll)
            self._n2 = dims[1]  # ordinate  (ul - ll)
            self._n3 = 1
        elif dims.size == 3:
            self._n1 = dims[0]  # abscissa  (lr - ll)
            self._n2 = dims[1]  # ordinate  (ul - ll)
            self._n3 = dims[2]  # applicate (tl - ll)
        else:
            raise ValueError("Must provide up to 3 dimensions")

    @property
    def ll(self):
        """
        :return: Lower-left corner of reciprocal volume
        :rtype: :class:`numpy.ndarray`"""
        return self.v1*self._r1[0] + self.v2*self._r2[0] + self.v3*self._r3[0]

    @property
    def lr(self):
        """
        :return: Lower-right corner of reciprocal volume
        :rtype: :class:`numpy.ndarray`"""
        return self.v1*self._r1[1] + self.v2*self._r2[0] + self.v3*self._r3[0]

    @property
    def ul(self):
        """
        :return: Upper-left corner of reciprocal volume
        :rtype: :class:`numpy.ndarray`"""
        return self.v1*self._r1[0] + self.v2*self._r2[1] + self.v3*self._r3[0]

    @property
    def tl(self):
        """
        :return: Top-left corner of reciprocal volume
        :rtype: :class:`numpy.ndarray`"""
        return self.v1*self._r1[0] + self.v2*self._r2[0] + self.v3*self._r3[1]

    @property
    def v1(self):
        """
        :return: Vector of first axis
        :rtype: :class:`numpy.ndarray`"""
        return self._v1

    @v1.setter
    def v1(self, v):
        if len(v) != 3:
            raise ValueError("Must provide vector of length 3")
        self._v1 = np.asarray(v)

    @property
    def v2(self):
        """
        :return: Vector of second axis
        :rtype: :class:`numpy.ndarray`"""
        return self._v2

    @v2.setter
    def v2(self, v):
        if len(v) != 3:
            raise ValueError("Must provide vector of length 3")
        self._v2 = np.asarray(v)

    @property
    def v3(self):
        """
        :return: Vector of third axis
        :rtype: :class:`numpy.ndarray`"""
        return self._v3

    @v3.setter
    def v3(self, v):
        if len(v) != 3:
            raise ValueError("Must provide vector of length 3")
        self._v3 = np.asarray(v)

    @property
    def r1(self):
        """
        :return: Range of first axis
        :rtype: :class:`numpy.ndarray`"""
        return np.linspace(self._r1[0], self._r1[1], self.bins[0])

    @r1.setter
    def r1(self, r):
        if len(r) != 2:
            raise ValueError("Must provide 2 values, min and max")
        self._r1 = np.asarray(r)

    @property
    def r2(self):
        """
        :return: Range of second axis
        :rtype: :class:`numpy.ndarray`"""
        return np.linspace(self._r2[0], self._r2[1], self.bins[1])

    @r2.setter
    def r2(self, r):
        if len(r) != 2:
            raise ValueError("Must provide 2 values, min and max")
        self._r2 = np.asarray(r)

    @property
    def r3(self):
        """
        :return: Range of third axis
        :rtype: :class:`numpy.ndarray`"""
        return np.linspace(self._r3[0], self._r3[1], self.bins[2])

    @r3.setter
    def r3(self, r):
        if len(r) != 2:
            raise ValueError("Must provide 2 values, min and max")
        self._r3 = np.asarray(r)

    @property
    def v1_delta(self):
        """
        :return: Increment vector of first axis
        :rtype: :class:`numpy.ndarray`"""
        return self.v1 if self.r1.size == 1 else (self.r1[1]-self.r1[0]) * self.v1

    @property
    def v2_delta(self):
        """
        :return: Increment vector of second axis
        :rtype: :class:`numpy.ndarray`"""
        return self.v2 if self.r2.size == 1 else (self.r2[1]-self.r2[0]) * self.v2

    @property
    def v3_delta(self):
        """
        :return: Increment vector of third axis
        :rtype: :class:`numpy.ndarray`"""
        return self.v3 if self.r3.size == 1 else (self.r3[1]-self.r3[0]) * self.v3

    def get_axes_names(self):
        """
        >>> grid = Grid()
        >>> grid.get_axes_names()
        ('[1 0 0]', '[0 1 0]', '[0 0 1]')

        :return: Axis names, vector of each direction
        :rtype: tuple of str"""
        return str(self.v1), str(self.v2), str(self.v3)

    def get_q_meshgrid(self):
        """Equivalent to :obj:`numpy.mgrid` for this volume

        :return: mesh-grid :class:`numpy.ndarray` all of the same dimensions
        :rtype: tuple of :class:`numpy.ndarray`
        """
        x = self.r1.reshape((self._n1, 1, 1))
        y = self.r2.reshape((1, self._n2, 1))
        z = self.r3.reshape((1, 1, self._n3))
        qx = x*self.v1[0] + y*self.v2[0] + z*self.v3[0]
        qy = x*self.v1[1] + y*self.v2[1] + z*self.v3[1]
        qz = x*self.v1[2] + y*self.v2[2] + z*self.v3[2]
        return qx, qy, qz

    def get_squashed_q_meshgrid(self):
        """Almost equivalent to :obj:`numpy.ogrid` for this volume. It may
        have more than one dimension not equal to 1. This can be used
        with numpy broadcasting.

        :return: mesh-grid :class:`numpy.ndarray` with some dimension equal to 1
        :rtype: tuple of :class:`numpy.ndarray`

        """
        xbins = self._get_bin_number(0)
        ybins = self._get_bin_number(1)
        zbins = self._get_bin_number(2)
        qx = xbins[0]*self.v1[0] + xbins[1]*self.v2[0] + xbins[2]*self.v3[0]
        qy = ybins[0]*self.v1[1] + ybins[1]*self.v2[1] + ybins[2]*self.v3[1]
        qz = zbins[0]*self.v1[2] + zbins[1]*self.v2[2] + zbins[2]*self.v3[2]
        return qx, qy, qz

    def _get_bin_number(self, index):
        binx = np.zeros((1, 1, 1)) if self.v1[index] == 0 else self.r1.reshape((self.bins[0], 1, 1))
        biny = np.zeros((1, 1, 1)) if self.v2[index] == 0 else self.r2.reshape((1, self.bins[1], 1))
        binz = np.zeros((1, 1, 1)) if self.v3[index] == 0 else self.r3.reshape((1, 1, self.bins[2]))
        return binx, biny, binz


def length(v):
    return np.linalg.norm(v)


def check_parallel(v1, v2):
    return (np.cross(v1, v2) == 0).all()


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (length(v1) * length(v2)))


def norm(v):
    return v/length(v)


def norm1(v):
    return v/np.min(np.abs(v[np.nonzero(v)]))


def corners_to_vectors(ll=None, lr=None, ul=None, tl=None):
    if ll is None or lr is None:
        raise ValueError("Need to provide at least ll (lower-left) and lr (lower-right) corners")
    elif ul is None:  # 1D
        v1 = get_vector_from_points(ll, lr)
        v2, v3 = find_other_vectors(v1)
        _r0 = np.linalg.solve(np.transpose([v1, v2, v3]), ll)
        _r1 = np.linalg.solve(np.transpose([v1, v2, v3]), lr)
        r1 = (_r0[0], _r1[0])
        r2 = (_r0[1], _r0[1])
        r3 = (_r0[2], _r0[2])
    elif tl is None:  # 2D
        v1 = get_vector_from_points(ll, lr)
        v2 = get_vector_from_points(ll, ul)
        try:
            v3 = norm1(np.cross(v1, v2))
        except ValueError:
            raise ValueError("Vector from ll to lr is parallel with vector from ll to ul")
        _r0 = np.linalg.solve(np.transpose([v1, v2, v3]), ll)
        _r1 = np.linalg.solve(np.transpose([v1, v2, v3]), lr)
        _r2 = np.linalg.solve(np.transpose([v1, v2, v3]), ul)
        r1 = (_r0[0], _r1[0])
        r2 = (_r0[1], _r2[1])
        r3 = (_r0[2], _r0[2])
    else:  # 3D
        v1 = get_vector_from_points(ll, lr)
        v2 = get_vector_from_points(ll, ul)
        v3 = get_vector_from_points(ll, tl)
        try:
            _r0 = np.linalg.solve(np.transpose([v1, v2, v3]), ll)
            _r1 = np.linalg.solve(np.transpose([v1, v2, v3]), lr)
            _r2 = np.linalg.solve(np.transpose([v1, v2, v3]), ul)
            _r3 = np.linalg.solve(np.transpose([v1, v2, v3]), tl)
        except np.linalg.linalg.LinAlgError:
            raise ValueError("Unable to determine vectors, check ll, lr, ul and tl values")
        r1 = (_r0[0], _r1[0])
        r2 = (_r0[1], _r2[1])
        r3 = (_r0[2], _r3[2])

    return v1, v2, v3, r1, r2, r3


def get_vector_from_points(p1, p2):
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    try:
        return norm1(p2 - p1)
    except:
        raise ValueError("Points provided must be different")


def find_other_vectors(v):
    from itertools import combinations
    c = combinations(np.eye(3), 2)
    while True:
        v0, v1 = next(c)
        if np.linalg.cond(np.transpose([v, v0, v1])) == np.inf:
            continue
        else:
            break
    return v0, v1
