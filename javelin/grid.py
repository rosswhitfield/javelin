"""
====
grid
====

Grid class to allow the Q-space grid to be definied in different
ways Should allow for corners and bins, or a matrix and the axis to be
defined.

The grid can de defined in with reciprocal lattice units (r.l.u) or q.
"""

import numpy as np


class Grid(object):
    def __init__(self,
                 ll=(0.0, 0.0, 0.0),
                 lr=(1.0, 0.0, 0.0),
                 ul=(0.0, 1.0, 0.0),
                 tl=(0.0, 0.0, 1.0),
                 bins=(101, 101)):
        # User providied information
        self._vertices = {'ll': np.array(ll),  # lower left
                          'lr': np.array(lr),  # lower right
                          'ul': np.array(ul),  # upper left
                          'tl': np.array(tl)}  # top left

        # Calculated grid info
        self.bins = bins
        self.__vertices_to_vectors()

        self._unitcell = None
        self.units = 'r.l.u'

    @property
    def bins(self):
        if self._2D:
            return self._n1, self._n2
        else:
            return self._n1, self._n2, self._n3

    @bins.setter
    def bins(self, dims):
        dims = np.asarray(dims)
        if (dims < 2).any():
            raise ValueError("Must have more than 1 bin in each direction")
        if len(dims) == 2:
            self._dims = 2
            self._2D = True
            self._n1 = dims[0]  # abscissa  (lr - ll)
            self._n2 = dims[1]  # ordinate  (ul - ll)
            self._n3 = 1
        elif len(dims) == 3:
            self._dims = 3
            self._2D = False
            self._n1 = dims[0]  # abscissa  (lr - ll)
            self._n2 = dims[1]  # ordinate  (ul - ll)
            self._n3 = dims[2]  # applicate (tl - ll)
        else:
            raise ValueError("Must provide 2 or 3 dimensions")
        self.__vertices_to_vectors()

    @property
    def twoD(self):
        return self._2D

    @property
    def ll(self):
        return self._vertices['ll']

    @ll.setter
    def ll(self, ll):
        if len(ll) != 3:
            raise ValueError("Must have length 3")
        self._vertices['ll'] = np.asarray(ll)
        self.__vertices_to_vectors()

    @property
    def lr(self):
        return self._vertices['lr']

    @lr.setter
    def lr(self, lr):
        if len(lr) != 3:
            raise ValueError("Must have length 3")
        self._vertices['lr'] = np.asarray(lr)
        self.__vertices_to_vectors()

    @property
    def ul(self):
        return self._vertices['ul']

    @ul.setter
    def ul(self, ul):
        if len(ul) != 3:
            raise ValueError("Must have length 3")
        self._vertices['ul'] = np.asarray(ul)
        self.__vertices_to_vectors()

    @property
    def tl(self):
        return self._vertices['tl']

    @tl.setter
    def tl(self, tl):
        if len(tl) != 3:
            raise ValueError("Must have length 3")
        self._vertices['tl'] = np.asarray(tl)
        self.__vertices_to_vectors()

    @property
    def v1(self):
        return self._v1

    @property
    def v2(self):
        return self._v2

    @property
    def v3(self):
        return self._v3

    @property
    def r1(self):
        return self._r1

    @property
    def r2(self):
        return self._r2

    @property
    def r3(self):
        return self._r3

    def __validate_vectors(self):
        vertices = ['lr', 'ul'] if self._2D else ['lr', 'ul', 'tl']
        vector_dict = {}
        for vertex in vertices:
            # Create vector from ll to 'vertex'
            vector = self._vertices[vertex] - self._vertices['ll']
            # Check length of vector
            if length(vector) == 0:
                raise ValueError("Distance between ll and " + vertex + " is 0")
            # Compare vector with previous to check it parallel
            for item in vector_dict:
                if check_parallel(vector, vector_dict[item]):
                    raise ValueError("Vector from ll to " + vertex +
                                     " is parallel with the vector from ll to "+item)
            vector_dict[vertex] = vector  # Store to allow comparison with other vectors

    def __vertices_to_vectors(self):
        self.__validate_vectors()
        self._origin = self._vertices['ll']
        self._v1 = norm(self._vertices['lr']-self._vertices['ll'])
        self._v2 = norm(self._vertices['ul']-self._vertices['ll'])
        self._v3 = np.array([0, 0, 1]) if self._2D else norm(self._vertices['tl'] -
                                                             self._vertices['ll'])
        self._r1 = np.linspace(0,
                               length(self._vertices['lr']-self._vertices['ll'])/length(self.v1),
                               self._n1)
        self._r2 = np.linspace(0,
                               length(self._vertices['ul']-self._vertices['ll'])/length(self.v2),
                               self._n2)
        self._r3 = np.array([0]) if self._2D else np.linspace(0,
                                                              length(self._vertices['tl'] -
                                                                     self._vertices['ll']) /
                                                              length(self.v3),
                                                              self._n3)

    def get_axis_names(self):
        return (str(self._origin) + str(' + x') + str(self.v1),
                str(self._origin) + str(' + y') + str(self.v2),
                str(self._origin) + str(' + z') + str(self.v3))

    def get_q_meshgrid(self):
        self.__validate_vectors()
        dx = (self.lr - self.ll)/(self._n1-1)
        dy = (self.ul - self.ll)/(self._n2-1)
        x = np.arange(self._n1).reshape((self._n1, 1))
        y = np.arange(self._n2).reshape((1, self._n2))
        if self._2D:
            qx = self.ll[0] + x*dx[0] + y*dy[0]
            qy = self.ll[1] + x*dx[1] + y*dy[1]
            qz = self.ll[2] + x*dx[2] + y*dy[2]
        else:  # assume _dims == 3
            x.shape = (self._n1, 1, 1)
            y.shape = (1, self._n2, 1)
            z = np.arange(self._n3).reshape((1, 1, self._n3))
            dz = (self.tl - self.ll)/(self._n3-1)
            qx = self.ll[0] + x*dx[0] + y*dy[0] + z*dz[0]
            qy = self.ll[1] + x*dx[1] + y*dy[1] + z*dz[1]
            qz = self.ll[2] + x*dx[2] + y*dy[2] + z*dz[2]
        return qx, qy, qz

    def get_squashed_q_meshgrid(self):
        self.__validate_vectors()
        dx = (self.lr - self.ll)/(self._n1-1)
        dy = (self.ul - self.ll)/(self._n2-1)
        qx_bins = get_bin_number(self.v1, self.v2, self.v3, self.bins, 0)
        qy_bins = get_bin_number(self.v1, self.v2, self.v3, self.bins, 1)
        qz_bins = get_bin_number(self.v1, self.v2, self.v3, self.bins, 2)
        qx = np.zeros(qx_bins)
        qy = np.zeros(qy_bins)
        qz = np.zeros(qz_bins)
        if self._2D:
            x = np.arange(qx_bins[0]).reshape((qx_bins[0], 1))
            y = np.arange(qx_bins[1]).reshape((1, qx_bins[1]))
            qx = self.ll[0] + x*dx[0] + y*dy[0]
            x = np.arange(qy_bins[0]).reshape((qy_bins[0], 1))
            y = np.arange(qy_bins[1]).reshape((1, qy_bins[1]))
            qy = self.ll[1] + x*dx[1] + y*dy[1]
            x = np.arange(qz_bins[0]).reshape((qz_bins[0], 1))
            y = np.arange(qz_bins[1]).reshape((1, qz_bins[1]))
            qz = self.ll[2] + x*dx[2] + y*dy[2]
        else:
            dz = (self.tl - self.ll)/(self._n3-1)
            x = np.arange(qx_bins[0]).reshape((qx_bins[0], 1, 1))
            y = np.arange(qx_bins[1]).reshape((1, qx_bins[1], 1))
            z = np.arange(qx_bins[2]).reshape((1, 1, qx_bins[2]))
            qx = self.ll[0] + x*dx[0] + y*dy[0] + z*dz[0]
            x = np.arange(qy_bins[0]).reshape((qy_bins[0], 1, 1))
            y = np.arange(qy_bins[1]).reshape((1, qy_bins[1], 1))
            z = np.arange(qy_bins[2]).reshape((1, 1, qy_bins[2]))
            qy = self.ll[1] + x*dx[1] + y*dy[1] + z*dz[1]
            x = np.arange(qz_bins[0]).reshape((qz_bins[0], 1, 1))
            y = np.arange(qz_bins[1]).reshape((1, qz_bins[1], 1))
            z = np.arange(qz_bins[2]).reshape((1, 1, qz_bins[2]))
            qz = self.ll[2] + x*dx[2] + y*dy[2] + z*dz[2]
        return qx, qy, qz


def get_bin_number(vabs, vord, vapp, bins, index):
    binx = 1 if vabs[index] == 0 else bins[0]
    biny = 1 if vord[index] == 0 else bins[1]
    if len(bins) == 2:
        return binx, biny
    else:
        binz = 1 if vapp[index] == 0 else bins[2]
    return binx, biny, binz


def length(v):
    return np.linalg.norm(v)


def check_parallel(v1, v2):
    return (np.cross(v1, v2) == 0).all()


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (length(v1) * length(v2)))


def norm(v):
    return v/length(v)
