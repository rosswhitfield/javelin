"""
====
grid
====
"""

import numpy as np


class Grid:
    """Grid class to allow the Q-space grid to be defined in different
    ways. The grid can be defined be either specifying the corners of
    the volume or by the axis vectors.

    The grid is defined by three vectors **v1**, **v2** and **v3**;
    the range of these vectors **r1**, **r2** and **r3**; and the
    number of bin in each of these directions.

    :examples:

    Setting grid by defining corners

    >>> grid = Grid()
    >>> grid.set_corners(ll=[-2,-2,-3], lr=[2,2,-3], ul=[-2,-2,3])
    >>> grid.bins = 5, 3
    >>> grid.v1
    array([ 1.,  1.,  0.])
    >>> grid.v2
    array([ 0.,  0.,  1.])
    >>> grid.v3
    array([ 1., -1.,  0.])
    >>> grid.r1
    array([-2., -1.,  0.,  1.,  2.])
    >>> grid.r2
    array([-3.,  0.,  3.])
    >>> grid.r3
    array([ 0.])
    >>> print(grid)
    lower left  corner :     [-2. -2. -3.]
    lower right corner :     [ 2.  2. -3.]
    upper left  corner :     [-2. -2.  3.]
    top   left  corner :     [-2. -2. -3.]
    <BLANKLINE>
    hor. increment     :     [ 1.  1.  0.]
    vert. increment    :     [ 0.  0.  3.]
    top   increment    :     [ 1. -1.  0.]
    <BLANKLINE>
    # of points        :     5 x 3 x 1

    Setting grid by vectors and ranges

    >>> grid = Grid()
    >>> grid.v1 = [1, 1, 1]
    >>> grid.v2 = [2, -1, -1]
    >>> grid.v3 = [0, 1, -1]
    >>> grid.r1 = [-1, 1]
    >>> grid.r2 = [0, 2]
    >>> grid.r3 = [0, 2]
    >>> grid.bins = 3, 3, 3
    >>> print(grid)
    lower left  corner :     [-1 -1 -1]
    lower right corner :     [1 1 1]
    upper left  corner :     [ 3 -3 -3]
    top   left  corner :     [-1  1 -3]
    <BLANKLINE>
    hor. increment     :     [ 1.  1.  1.]
    vert. increment    :     [ 2. -1. -1.]
    top   increment    :     [ 0.  1. -1.]
    <BLANKLINE>
    # of points        :     3 x 3 x 3

    """
    def __init__(self):
        # vectors of grid
        self._v1 = np.array([1., 0., 0.])
        self._v2 = np.array([0., 1., 0.])
        self._v3 = np.array([0., 0., 1.])

        # min max of each vector
        self._r1 = np.array([0., 1.])
        self._r2 = np.array([0., 1.])
        self._r3 = np.array([0., 1.])

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
        :getter: Set the first axis
        :setter: Get the first axis
        :type: :class:`numpy.ndarray`"""
        return self._v1

    @v1.setter
    def v1(self, v):
        if len(v) != 3:
            raise ValueError("Must provide vector of length 3")
        self._v1 = np.asarray(v)

    @property
    def v2(self):
        """
        :getter: Set the second axis
        :setter: Get the second axis
        :type: :class:`numpy.ndarray`"""
        return self._v2

    @v2.setter
    def v2(self, v):
        if len(v) != 3:
            raise ValueError("Must provide vector of length 3")
        self._v2 = np.asarray(v)

    @property
    def v3(self):
        """
        :getter: Set the third axis
        :setter: Get the third axis
        :type: :class:`numpy.ndarray`"""
        return self._v3

    @v3.setter
    def v3(self, v):
        if len(v) != 3:
            raise ValueError("Must provide vector of length 3")
        self._v3 = np.asarray(v)

    @property
    def r1(self):
        """Set the range of the first axis, two values min and max

        :getter: Array of values for each bin in the axis
        :setter: Range of first axis, two values
        :type: :class:`numpy.ndarray`"""
        return np.linspace(self._r1[0], self._r1[1], self.bins[0])

    @r1.setter
    def r1(self, r):
        if len(r) != 2:
            raise ValueError("Must provide 2 values, min and max")
        self._r1 = np.asarray(r)

    @property
    def r2(self):
        """Set the range of the second axis, two values min and max

        :getter: Array of values for each bin in the axis
        :setter: Range of second axis, two values
        :type: :class:`numpy.ndarray`"""
        return np.linspace(self._r2[0], self._r2[1], self.bins[1])

    @r2.setter
    def r2(self, r):
        if len(r) != 2:
            raise ValueError("Must provide 2 values, min and max")
        self._r2 = np.asarray(r)

    @property
    def r3(self):
        """Set the range of the third axis, two values min and max

        :getter: Array of values for each bin in the axis
        :setter: Range of third axis, two values
        :type: :class:`numpy.ndarray`"""
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
        ('[ 1.  0.  0.]', '[ 0.  1.  0.]', '[ 0.  0.  1.]')

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
    """Calculates the length of a vector

    :param v: vector
    :type v: array-like object of numbers
    :return: length of vector
    :rtype: float

    :examples:

    >>> length([1,0,0])
    1.0

    >>> length([1,-1,0])
    1.4142135623730951

    >>> length([2,2,2])
    3.4641016151377544
    """
    return np.linalg.norm(v)


def check_parallel(v1, v2):
    """Checks if two vectors are parallel

    :param v1: vector1
    :type v1: array-like object of numbers
    :param v2: vector2
    :type v2: array-like object of numbers
    :return: if parallel
    :rtype: bool

    :examples:

    >>> check_parallel([1,0,0], [-1,0,0])
    True

    >>> check_parallel([1,0,0], [0,1,0])
    False
    """
    return (np.cross(v1, v2) == 0).all()


def angle(v1, v2):
    """Calculates the angle between two vectors

    :param v1: vector 1
    :type v1: array-like object of numbers
    :param v2: vector 2
    :type v2: array-like object of numbers
    :return: angle (radians)
    :rtype: float

    :examples:

    >>> angle([1,0,0], [-1,0,0])
    3.1415926535897931

    >>> angle([1,0,0], [0,1,0])
    1.5707963267948966
    """
    return np.arccos(np.dot(v1, v2) / (length(v1) * length(v2)))


def norm(v):
    """Calculates the normalised vector

    :param v: vector
    :type v: array-like object of numbers
    :return: normalised vector
    :rtype: :class:`numpy.ndarray`

    :examples:

    >>> norm([5, 0, 0])
    array([ 1.,  0.,  0.])

    >>> norm([1, 1, 0])
    array([ 0.70710678,  0.70710678,  0.        ])
    """
    return v/length(v)


def norm1(v):
    """Calculate the equivalent vector with the smallest non-zero
    component equal to one.

    :param v: vector
    :type v: array-like object of numbers
    :return: normalised1 vector
    :rtype: :class:`numpy.ndarray`

    :examples:

    >>> norm1([5, 10, 0])
    array([ 1.,  2.,  0.])

    >>> norm1([1, 1, 0])
    array([ 1.,  1.,  0.])

    """
    v = np.asarray(v)
    return v/np.min(np.abs(v[np.nonzero(v)]))


def corners_to_vectors(ll=None, lr=None, ul=None, tl=None):
    """This function converts the provided corners into axes vectors and
    axes ranges. It will also calculate sensible vectors for any
    unprovided corners.

    You must provide at minimum the lower-left (**ll**) and
    lower-right (**lr**) corners.

    :param ll: lower-left corner (required)
    :type ll: array-like object of numbers
    :param lr: lower-right corner (required)
    :type lr: array-like object of numbers
    :param ul: upper-left corner
    :type ul: array-like object of numbers
    :param tl: top-left corner
    :type tl: array-like object of numbers
    :return: three axes vectors and three axes ranges
    :rtype: tuple of three :class:`numpy.ndarray` and three tuple ranges

    :examples:

    Using only **ll** and **lr**, the other two vector are calculated
    using :func:`javelin.grid.find_other_vectors`

    >>> v1, v2, v3, r1, r2, r3 = corners_to_vectors(ll=[-3,-3,0], lr=[3, 3, 0])
    >>> print(v1, v2, v3)
    [ 1.  1.  0.] [ 1.  0.  0.] [ 0.  0.  1.]
    >>> print(r1, r2, r3) # doctest: +SKIP
    (-3.0, 3.0) (0.0, 0.0) (0.0, 0.0)

    Using **ll**, **lr** and **ul**, the other vector is the
    :func:`javelin.grid.norm1` of the cross product of the first two
    vectors defined by the corners.

    >>> v1, v2, v3, r1, r2, r3 = corners_to_vectors(ll=[-3,-3,-2], lr=[3, 3, -2], ul=[-3, -3, 2])
    >>> print(v1, v2, v3)
    [ 1.  1.  0.] [ 0.  0.  1.] [ 1. -1.  0.]
    >>> print(r1, r2, r3) # doctest: +SKIP
    (-3.0, 3.0) (-2.0, 2.0) (0.0, 0.0)

    Finally defining all corners

    >>> v1,v2,v3,r1,r2,r3 = corners_to_vectors(ll=[-5,-6,-7],lr=[-5,-6,7],ul=[-5,6,-7],tl=[5,-6,-7])
    >>> print(v1, v2, v3)
    [ 0.  0.  1.] [ 0.  1.  0.] [ 1.  0.  0.]
    >>> print(r1, r2, r3)
    (-7.0, 7.0) (-6.0, 6.0) (-5.0, 5.0)

    If you provided corners which will create parallel vectors you will get a ValueError

    >>> corners_to_vectors(ll=[0, 0, 0], lr=[1, 0, 0], ul=[1, 0, 0])
    Traceback (most recent call last):
        ...
    ValueError: Vector from ll to lr is parallel with vector from ll to ul
    """
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
    """Calculates the vector form two points

    :param p1: point 1
    :type p1: array-like object of numbers
    :param p2: point 2
    :type p2: array-like object of numbers
    :return: vector between points
    :rtype: :class:`numpy.ndarray`

    :examples:

    >>> get_vector_from_points([-1, -1, 0], [1, 1, 0])
    array([ 1.,  1.,  0.])

    >>> get_vector_from_points([0, 0, 0], [2, 2, 4])
    array([ 1.,  1.,  2.])
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    try:
        return norm1(p2 - p1)
    except ValueError:
        raise ValueError("Points provided must be different")


def find_other_vectors(v):
    """This will find two new vectors which in combination with the
    provided vector (**v**) will form a basis for a complete space filling
    set.

    :param v: vector
    :type v: array-like object of numbers
    :return: two new space filling vectors
    :rtype: tuple of two :class:`numpy.ndarray`

    :examples:

    >>> find_other_vectors([1, 0, 0])
    (array([ 0.,  1.,  0.]), array([ 0.,  0.,  1.]))

    >>> find_other_vectors([0, 0, 1])
    (array([ 1.,  0.,  0.]), array([ 0.,  1.,  0.]))

    >>> find_other_vectors([1, 1, 0])
    (array([ 1.,  0.,  0.]), array([ 0.,  0.,  1.]))

    """
    import warnings
    from itertools import combinations
    c = combinations(np.eye(3), 2)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='divide by zero encountered in true_divide',
                                category=RuntimeWarning)
        while True:
            v0, v1 = next(c)
            if np.linalg.cond(np.transpose([v, v0, v1])) == np.inf:
                continue
            else:
                break
    return v0, v1
