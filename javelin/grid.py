"""
====
grid
====

Grid class to allow the Q-space grid to be definied in different
ways Should allow for corners and bins, or a matrix and the axis to be
defined.

The grid can de defined in with reciprocal lattice units (r.l.u) or q.
"""

from __future__ import division
import numpy as np
from itertools import combinations


class Grid(object):
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

    def set_corners(self,
                    ll=(0, 0, 0),
                    lr=None,
                    ul=None,
                    tl=None):
        self.v1, self.v2, self.v3, self.r1, self.r2, self.r3 = corners_to_vectors(ll, lr, ul, tl)

    @property
    def bins(self):
        return self._n1, self._n2, self._n3

    @bins.setter
    def bins(self, dims):
        if isinstance(dims, int):
            self._n1 = dims  # abscissa  (lr - ll)
            self._n2 = 1
            self._n3 = 1
        else:
            dims = np.asarray(dims, dtype=int)
            if dims.size == 2:
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
        return self.v1*self._r1[0] + self.v2*self._r2[0] + self.v3*self._r3[0]

    @property
    def lr(self):
        return self.v1*self._r1[1] + self.v2*self._r2[0] + self.v3*self._r3[0]

    @property
    def ul(self):
        return self.v1*self._r1[0] + self.v2*self._r2[1] + self.v3*self._r3[0]

    @property
    def tl(self):
        return self.v1*self._r1[0] + self.v2*self._r2[0] + self.v3*self._r3[1]

    @property
    def v1(self):
        return self._v1

    @v1.setter
    def v1(self, v):
        if len(v) != 3:
            raise ValueError("Must provide vector of length 3")
        self._v1 = np.asarray(v)

    @property
    def v2(self):
        return self._v2

    @v2.setter
    def v2(self, v):
        if len(v) != 3:
            raise ValueError("Must provide vector of length 3")
        self._v2 = np.asarray(v)

    @property
    def v3(self):
        return self._v3

    @v3.setter
    def v3(self, v):
        if len(v) != 3:
            raise ValueError("Must provide vector of length 3")
        self._v3 = np.asarray(v)

    @property
    def r1(self):
        return np.linspace(self._r1[0], self._r1[1], self.bins[0])

    @r1.setter
    def r1(self, r):
        if len(r) != 2:
            raise ValueError("Must provide 2 values, min and max")
        self._r1 = np.asarray(r)

    @property
    def r2(self):
        return np.linspace(self._r2[0], self._r2[1], self.bins[1])

    @r2.setter
    def r2(self, r):
        if len(r) != 2:
            raise ValueError("Must provide 2 values, min and max")
        self._r2 = np.asarray(r)

    @property
    def r3(self):
        return np.linspace(self._r3[0], self._r3[1], self.bins[2])

    @r3.setter
    def r3(self, r):
        if len(r) != 2:
            raise ValueError("Must provide 2 values, min and max")
        self._r3 = np.asarray(r)

    def get_axis_names(self):
        return str(self.v1), str(self.v2), str(self.v3)

    def get_q_meshgrid(self):
        x = self.r1.reshape((self._n1, 1, 1))
        y = self.r2.reshape((1, self._n2, 1))
        z = self.r3.reshape((1, 1, self._n3))
        qx = x*self.v1[0] + y*self.v2[0] + z*self.v3[0]
        qy = x*self.v1[1] + y*self.v2[1] + z*self.v3[1]
        qz = x*self.v1[2] + y*self.v2[2] + z*self.v3[2]
        return qx, qy, qz

    def get_squashed_q_meshgrid(self):
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


def corners_to_vectors(ll=None, lr=None, ul=None, tl=None):  # noqa
    if lr is None:
        raise ValueError("Need to provide at least ll (lower-left) and lr (lower-right) corners")
    elif ul is None:
        dims = 1
    elif tl is None:
        dims = 2
    else:
        dims = 3

    ll = np.asarray(ll)
    lr = np.asarray(lr)
    ul = np.asarray(ul)
    tl = np.asarray(tl)

    v1 = norm1(lr - ll)

    if dims == 1:
        c = combinations(np.eye(3), 2)
        while True:
            try:
                v2, v3 = next(c)
                _r0 = np.linalg.solve(np.transpose([v1, v2, v3]), ll)
            except np.linalg.linalg.LinAlgError:
                continue
            else:
                break
        _r1 = np.linalg.solve(np.transpose([v1, v2, v3]), lr)
        r1 = (_r0[0], _r1[0])
        r2 = (_r0[1], _r0[1])
        r3 = (_r0[2], _r0[2])
    elif dims == 2:
        v2 = norm1(ul - ll)
        v3 = norm1(np.cross(v1, v2))
        _r0 = np.linalg.solve(np.transpose([v1, v2, v3]), ll)
        _r1 = np.linalg.solve(np.transpose([v1, v2, v3]), lr)
        _r2 = np.linalg.solve(np.transpose([v1, v2, v3]), ul)
        r1 = (_r0[0], _r1[0])
        r2 = (_r0[1], _r2[1])
        r3 = (_r0[2], _r0[2])
    elif dims == 3:
        v2 = norm1(ul - ll)
        v3 = norm1(tl - ll)
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
