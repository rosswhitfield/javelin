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
    a = np.random.choice([13, 42], n**2).reshape((n, n, 1, 1)).astype(np.int64, copy=False)
    return a, x, y, z


def test_Target():
    nl = NeighborList([[0, 1, 1, 0, 0],
                       [0, 1, 0, 1, 0],
                       [0, 1, 0, 0, 1]])
    e = Energy()
    target = Target(np.asarray(nl).astype(np.intp), e)
    assert target.number_of_neighbors == 3
    assert str(target) == """Target(Energy=Energy()
Neighbors=[[0 1 1 0 0]
 [0 1 0 1 0]
 [0 1 0 0 1]])"""

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
    mod = BaseModifier(1, 0)

    a, x, y, z = create_test_arrays(n=10)

    a_copy = a.copy()
    x_copy = x.copy()
    y_copy = y.copy()
    z_copy = z.copy()

    accepted = mcrun(np.array([mod]),
                     np.array([target]),
                     10, 0,
                     a, x, y, z)

    assert accepted == (0, 0, 0)

    assert_equal(a, a_copy)
    assert_equal(x, x_copy)
    assert_equal(y, y_copy)
    assert_equal(z, z_copy)

    with pytest.raises(ValueError):
        mcrun(np.array([mod]),
              np.array([1]),
              10, 0,
              a, x, y, z)

    with pytest.raises(ValueError):
        mcrun(np.array([1]),
              np.array([target]),
              10, 0,
              a, x, y, z)

    with pytest.raises(TypeError):
        mcrun(mod,
              np.array([target]),
              10, 0,
              a, x, y, z)

    with pytest.raises(ValueError):
        mcrun(np.array([mod]),
              np.array([target]),
              10, 0,
              x, x, y, z)

    with pytest.raises(ValueError):
        mcrun(np.array([mod]),
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
    accepted = mcrun(np.array([mod]),
                     np.array([target]),
                     100, 1,
                     a, x, y, z)

    assert accepted == (0, 100, 0)

    assert_equal(a[:, 0, 0, 0], [13, 13, 13, 13, 42, 13, 13, 42, 13, 42])
    assert_almost_equal(x[:, 0, 0, 0], [-0.05497462, -0.0458027,  0.00135485,  0.07362052,
                                        -0.04381356, 0.0108872,  0.10269214, -0.06129387,
                                        0.00186389, -0.09833101])
    assert_almost_equal(y[:, 0, 0, 0], [-0.01652096, -0.12963918, -0.16094389, 0.09493609,
                                        -0.19580812, -0.07044182, -0.20151887, -0.04381209,
                                        -0.0543198, -0.06166294])
    assert_almost_equal(z[:, 0, 0, 0], [0.04603492, 0.05273691, -0.04455883, -0.04022899,
                                        0.12410821, -0.01768969, -0.05366404, -0.02749398,
                                        -0.18848344,  0.07021845])
