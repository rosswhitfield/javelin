from javelin.mccore import Target, mcrun
from javelin.neighborlist import NeighborList
from javelin.energies import Energy
from javelin.modifier import BaseModifier, Swap
from javelin.random import set_seed
from numpy.testing import assert_equal, assert_almost_equal
import numpy as np
import pytest
import sys


def create_test_arrays(n=2):
    np.random.seed(100)
    x = np.random.normal(0, 0.1, size=n**2).reshape((n, n, 1, 1))
    y = np.random.normal(0, 0.1, size=n**2).reshape((n, n, 1, 1))
    z = np.random.normal(0, 0.1, size=n**2).reshape((n, n, 1, 1))
    a = np.random.choice([13, 42], n**2).reshape((n, n, 1, 1))
    return a, x, y, z


def test_Target():
    nl = NeighborList([[0, 1, 1, 0, 0],
                       [0, 1, 0, 1, 0],
                       [0, 1, 0, 0, 1]])
    e = Energy()
    target = Target(np.asarray(nl).astype(np.intp), e)
    assert target.number_of_neighbours == 3
    assert str(target) == "Target(number_of_neighbours=3)"

    with pytest.raises(ValueError):
        Target(np.array([[0.0, 0.0, 1.0, 0.0, 0.0]]), e)

    with pytest.raises(ValueError):
        Target(np.array([0, 0, 1, 0, 0]).astype(np.intp), e)

    with pytest.raises(TypeError):
        Target(1, e)

    with pytest.raises(TypeError):
        Target([[0, 0, 0, 0, 0]], e)

    with pytest.raises(AssertionError):
        Target(np.array([[0, 0, 1, 0]]).astype(np.intp), e)

    with pytest.raises(TypeError):
        Target(np.array([[0, 0, 1, 0, 0]]).astype(np.intp), 1)


def test_mcrun_should_do_nothing():
    nl = NeighborList([[0, 1, 1, 0, 0],
                       [0, 1, -1, 0, 0]])
    target = Target(np.asarray(nl).astype(np.intp), Energy())
    mod = BaseModifier(1)

    a, x, y, z = create_test_arrays(n=10)

    a_copy = a.copy()
    x_copy = x.copy()
    y_copy = y.copy()
    z_copy = z.copy()

    accepted = mcrun(mod,
                     np.array([target]),
                     10, 0,
                     a, x, y, z)

    assert accepted == (0, 0, 0)

    assert_equal(a, a_copy)
    assert_equal(x, x_copy)
    assert_equal(y, y_copy)
    assert_equal(z, z_copy)

    with pytest.raises(ValueError):
        mcrun(mod,
              np.array([1]),
              10, 0,
              a, x, y, z)

    with pytest.raises(TypeError):
        mcrun(1,
              np.array([target]),
              10, 0,
              a, x, y, z)

    with pytest.raises(ValueError):
        mcrun(mod,
              np.array([target]),
              10, 0,
              x, x, y, z)

    with pytest.raises(ValueError):
        mcrun(mod,
              np.array([target]),
              10, 0,
              a, x, y, a)


@pytest.mark.skipif(sys.platform == 'win32',
                    reason="skipping test for windows, different c random numbers")
def test_mcrun_accept_all():
    nl = NeighborList([[0, 1, 1, 0, 0],
                       [0, 1, -1, 0, 0]])
    target = Target(np.asarray(nl).astype(np.intp), Energy())
    mod = Swap(0)

    a, x, y, z = create_test_arrays(n=10)

    set_seed()
    accepted = mcrun(mod,
                     np.array([target]),
                     100, 1,
                     a, x, y, z)

    assert accepted == (0, 100, 0)

    assert_equal(a[:, 0, 0, 0], [13, 42, 42, 42, 42, 13, 13, 42, 13, 42])
    assert_almost_equal(x[:, 0, 0, 0], [-0.1749765, -0.0079611,  0.1029733,  0.02224, -0.0842436,
                                        -0.0376903,  0.0018639, -0.1733096, -0.2487152,  0.0778822])
    assert_almost_equal(y[:, 0, 0, 0], [-0.1704651,  0.1093687,  0.0693391, -0.149772, -0.1097172,
                                        0.2077712, -0.0543198, -0.0343298,  0.1576167,  0.1236908])
    assert_almost_equal(z[:, 0, 0, 0], [0.0604424,  0.0111823, -0.0159517, -0.1094557,  0.0327245,
                                        -0.119944, -0.1884834,  0.1159969,  0.0182343, -0.0589709])
