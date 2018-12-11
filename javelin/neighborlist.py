"""
============
neighborlist
============
"""

import numpy as np


class NeighborList:
    """The NeighborList class

    Contains an `n x 5` array of neighbor vectors each being
    `[origin_site, target_site, i, j, k]`.

    >>> nl = NeighborList()
    >>> print(nl)
          |     site      |    vector
    index | origin target |   i   j   k
    <BLANKLINE>
    >>> nl = NeighborList([[0,1,1,0,0],[0,1,0,1,0],[0,1,0,0,1]])
    >>> print(nl)
          |     site      |    vector
    index | origin target |   i   j   k
        0 |      0      1 |   1   0   0
        1 |      0      1 |   0   1   0
        2 |      0      1 |   0   0   1

    You can slice a NeighborList:

    >>> print(nl[0])
          |     site      |    vector
    index | origin target |   i   j   k
        0 |      0      1 |   1   0   0
    >>> print(nl[-1])
          |     site      |    vector
    index | origin target |   i   j   k
        0 |      0      1 |   0   0   1
    >>> print(nl[1:3])
          |     site      |    vector
    index | origin target |   i   j   k
        0 |      0      1 |   0   1   0
        1 |      0      1 |   0   0   1
    >>> print(nl[0,2])
          |     site      |    vector
    index | origin target |   i   j   k
        0 |      0      1 |   1   0   0
        1 |      0      1 |   0   0   1

    You can set a vector:

    >>> nl[0] = [5, 5, 5, 5, 5]
    >>> nl[-1] = [2, 2, 2, 2, 2]
    >>> print(nl)
          |     site      |    vector
    index | origin target |   i   j   k
        0 |      5      5 |   5   5   5
        1 |      0      1 |   0   1   0
        2 |      2      2 |   2   2   2

    You can add NeighborLists together:

    >>> nl2 = NeighborList([[1,1,1,0,0],[1,1,0,1,0]])
    >>> nl3 = nl + nl2
    >>> print(nl3)
          |     site      |    vector
    index | origin target |   i   j   k
        0 |      5      5 |   5   5   5
        1 |      0      1 |   0   1   0
        2 |      2      2 |   2   2   2
        3 |      1      1 |   1   0   0
        4 |      1      1 |   0   1   0

    And finallly you can delete vectors from the NeighborList

    >>> del nl3[-1]
    >>> print(nl3)
          |     site      |    vector
    index | origin target |   i   j   k
        0 |      5      5 |   5   5   5
        1 |      0      1 |   0   1   0
        2 |      2      2 |   2   2   2
        3 |      1      1 |   1   0   0
    >>> del nl3[1:3]
    >>> print(nl3)
          |     site      |    vector
    index | origin target |   i   j   k
        0 |      5      5 |   5   5   5
        1 |      1      1 |   1   0   0
    """
    def __init__(self, vectors=None):
        if vectors is None:
            self._vectors = np.empty((0, 5), dtype=int)
        else:
            try:
                vectors = np.asarray(vectors, dtype=int)
                if vectors.ndim != 2 or vectors.shape[1] != 5:
                    raise ValueError
                else:
                    self._vectors = vectors
            except ValueError:
                raise ValueError("vectors must be array-like of shape (n, 5) of type int, "
                                 "[[origin_site1, target_site1, i1, j1, k1], "
                                 "[origin_site2, target_site2, i2, j2, k2]]")

    def __len__(self):
        return len(self._vectors)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return NeighborList(self._vectors[key])
        elif isinstance(key, int):
            return NeighborList(self._vectors[key, None])
        else:
            try:
                key = np.asarray(key, dtype=int)
                if key.ndim == 1:
                    return NeighborList(self._vectors[key])
                else:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError("Must index with int, slice or 1D array-like object")

    def __setitem__(self, key, vector):
        try:
            vector = np.asarray(vector, dtype=int)
            self._vectors[key] = vector
        except ValueError:
            raise ValueError("Must be array-like of length 5, [origin_site, target_site, i, j, k]")

    def __delitem__(self, key):
        self._vectors = np.delete(self._vectors, key, axis=0)

    def __iter__(self):
        return iter(self._vectors)

    def __add__(self, other):
        return NeighborList(np.append(self.values, other.values, axis=0))

    def __array__(self):
        return self._vectors

    def __str__(self):
        return ("      |     site      |    vector\nindex | origin target |   i   j   k\n" +
                '\n'.join('{: 5} | {: 6} {: 6} | {: 3} {: 3} {: 3}'.
                          format(i, *v) for i, v in enumerate(self._vectors)))

    def __repr__(self):
        return "NeighborList(" + str(self._vectors).replace('\n', '\n             ') + ")"

    def append(self, vectors):
        """Append one or more vectors to the NeighborList

        >>> nl = NeighborList()
        >>> print(nl)
              |     site      |    vector
        index | origin target |   i   j   k
        <BLANKLINE>
        >>> nl.append([0,1,1,0,0])
        >>> print(nl)
              |     site      |    vector
        index | origin target |   i   j   k
            0 |      0      1 |   1   0   0
        >>> nl.append([[0,1,0,1,0],[0,1,0,0,1]])
        >>> print(nl)
              |     site      |    vector
        index | origin target |   i   j   k
            0 |      0      1 |   1   0   0
            1 |      0      1 |   0   1   0
            2 |      0      1 |   0   0   1
        """
        try:
            vectors = np.asarray(vectors, dtype=int)
            if vectors.ndim == 1:
                self._vectors = np.append(self._vectors, [vectors], axis=0)
            else:
                self._vectors = np.append(self._vectors, vectors, axis=0)
        except ValueError:
            raise ValueError("Must be a array-like of length 5, [origin_site, target_site, i, j, k]"
                             ", or shape (n, 5), [[origin_site1, target_site1, i1, j1, k1], "
                             "[origin_site2, target_site2, i2, j2, k2]]")

    @property
    def values(self):
        """Returns the neighbor vectors as a :class:`numpy.ndarray`

        This allows you to directly modify or set the vectors but be
        careful to maintain an (n, 5) array.

        >>> nl = NeighborList([[0,1,1,0,0],[0,1,0,1,0],[0,1,0,0,1]])
        >>> print(nl)
              |     site      |    vector
        index | origin target |   i   j   k
            0 |      0      1 |   1   0   0
            1 |      0      1 |   0   1   0
            2 |      0      1 |   0   0   1
        >>> nl.values
        array([[0, 1, 1, 0, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 0, 0, 1]])
        >>> nl.values[:,1] = 2
        >>> print(nl)
              |     site      |    vector
        index | origin target |   i   j   k
            0 |      0      2 |   1   0   0
            1 |      0      2 |   0   1   0
            2 |      0      2 |   0   0   1

        :return: array of neighbor vectors
        :rtype: :class:`numpy.ndarray`

        """
        return self._vectors
