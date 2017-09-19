import pytest
from javelin.neighborlist import NeighborList
from numpy.testing import assert_array_equal


def test_init():
    nl = NeighborList()
    assert nl.values.shape == (0, 5)
    assert len(nl) == 0
    assert repr(nl) == 'NeighborList([])'
    assert str(nl) == '      |     site      |    vector\nindex | origin target |   i   j   k\n'

    nl = NeighborList([[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [1, 2, 3, 4, 5]])
    assert nl.values.shape == (3, 5)
    assert len(nl) == 3
    assert (repr(nl) ==
            'NeighborList([[ 0  0  0  0  0]\n'
            '              [ 0  0 -1  0  0]\n'
            '              [ 1  2  3  4  5]])')
    assert (str(nl) ==
            '      |     site      |    vector\n'
            'index | origin target |   i   j   k\n'
            '    0 |      0      0 |   0   0   0\n'
            '    1 |      0      0 |  -1   0   0\n'
            '    2 |      1      2 |   3   4   5')

    with pytest.raises(ValueError):
        NeighborList([0, 0, 0, 0, 0])

    with pytest.raises(ValueError):
        NeighborList([[0, 0, 0, 0]])

    with pytest.raises(ValueError):
        NeighborList([[0, 0, 0, 0, 'a']])


def test__getitem__():
    nl = NeighborList([[0, 0, 0, 0, 0],
                       [0, 0, -1, 0, 0],
                       [1, 2, 3, 4, 5],
                       [5, 4, 3, 2, 1]])
    nl0 = nl[0]
    assert isinstance(nl0, NeighborList)
    assert_array_equal(nl0.values, [[0, 0, 0, 0, 0]])

    nl1 = nl[-1]
    assert isinstance(nl1, NeighborList)
    assert_array_equal(nl1.values, [[5, 4, 3, 2, 1]])

    nl2 = nl[2:4]
    assert isinstance(nl2, NeighborList)
    assert_array_equal(nl2.values, [[1, 2, 3, 4, 5],
                                    [5, 4, 3, 2, 1]])

    nl3 = nl[:]
    assert isinstance(nl3, NeighborList)
    assert_array_equal(nl3.values, nl.values)

    nl4 = nl[1, 3]
    assert isinstance(nl4, NeighborList)
    assert_array_equal(nl4.values, [[0, 0, -1, 0, 0],
                                    [5, 4, 3, 2, 1]])

    nl5 = nl[(1, 3)]
    assert isinstance(nl5, NeighborList)
    assert_array_equal(nl5.values, [[0, 0, -1, 0, 0],
                                    [5, 4, 3, 2, 1]])

    with pytest.raises(ValueError):
        nl[[[0], [0]]]

    with pytest.raises(IndexError):
        nl[4]


def test__setitem__():
    nl = NeighborList([[0, 0, 0, 0, 0],
                       [1, 2, 3, 4, 5]])

    nl[0] = [0, 1, 2, 3, 4]
    assert_array_equal(nl.values, [[0, 1, 2, 3, 4],
                                   [1, 2, 3, 4, 5]])

    nl[:] = [5, 4, 3, 2, 1]
    assert_array_equal(nl.values, [[5, 4, 3, 2, 1],
                                   [5, 4, 3, 2, 1]])

    with pytest.raises(IndexError):
        nl[2] = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError):
        nl[0] = [1, 2, 3, 4]

    with pytest.raises(ValueError):
        nl[0] = [1, 2, 3, 4, 'a']


def test__delitem__():
    nl = NeighborList([[0, 0, 0, 0, 0],
                       [0, 0, -1, 0, 0],
                       [1, 2, 3, 4, 5],
                       [5, 4, 3, 2, 1]])
    assert_array_equal(nl.values, [[0, 0, 0, 0, 0],
                                   [0, 0, -1, 0, 0],
                                   [1, 2, 3, 4, 5],
                                   [5, 4, 3, 2, 1]])

    del nl[1, 3]
    assert_array_equal(nl.values, [[0, 0, 0, 0, 0],
                                   [1, 2, 3, 4, 5]])

    del nl[-1]
    assert_array_equal(nl.values, [[0, 0, 0, 0, 0]])

    del nl[0]
    assert nl.values.shape == (0, 5)

    with pytest.raises(IndexError):
        del nl[0]


def test__add__():
    nl1 = NeighborList([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    nl2 = NeighborList([[1, 2, 3, 4, 5]])

    nl3 = nl1 + nl2
    assert_array_equal(nl3.values, [[0, 1, 2, 3, 4],
                                    [5, 6, 7, 8, 9],
                                    [1, 2, 3, 4, 5]])

    nl1 += nl2
    assert_array_equal(nl1.values, nl3.values)


def test__array__():
    import numpy as np
    nl = NeighborList([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    arr = np.asarray(nl)
    assert_array_equal(arr, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])


def test__append():
    nl = NeighborList()

    nl.append([1, 2, 3, 4, 5])
    assert_array_equal(nl.values, [[1, 2, 3, 4, 5]])

    nl.append([[0, 1, 2, 3, 4], [0, 1, 0, 1, 0]])
    assert_array_equal(nl.values, [[1, 2, 3, 4, 5],
                                   [0, 1, 2, 3, 4],
                                   [0, 1, 0, 1, 0]])

    with pytest.raises(ValueError):
        nl.append([[0, 1, 2, 3], [0, 1, 0, 1]])
