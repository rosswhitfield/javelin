import numpy as np
import pytest
import sys
from numpy.testing import assert_equal, assert_almost_equal
from javelin.modifier import (BaseModifier, SwapOccupancy,
                              SwapDisplacement, Swap,
                              ShiftDisplacementRange,
                              ShiftDisplacementNormal,
                              SetDisplacementRange,
                              SetDisplacementNormal)
from javelin.random import set_seed

if sys.platform.startswith("win"):
        pytest.skip("skipping tests for windows, different c random numbers",
                    allow_module_level=True)


def create_test_arrays(n=2):
    np.random.seed(100)
    x = np.random.normal(0, 0.1, size=n**2).reshape((n, n, 1, 1))
    y = np.random.normal(0, 0.1, size=n**2).reshape((n, n, 1, 1))
    z = np.random.normal(0, 0.1, size=n**2).reshape((n, n, 1, 1))
    a = np.random.choice([13, 42], n**2).reshape((n, n, 1, 1))
    return a, x, y, z


def test_BaseModifier():
    a, x, y, z = create_test_arrays()

    assert_equal(a, [[[[42]], [[42]]],
                     [[[13]], [[13]]]])

    mod = BaseModifier(3)
    assert str(mod) == 'BaseModifier(number_of_cells=3)'
    assert mod.number_of_cells == 3
    assert_equal(np.asarray(mod.cells), [[0, 0, 0],
                                         [0, 0, 0],
                                         [0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(100, 100, 100)
    assert_equal(np.asarray(cells), [[3, 32, 69],
                                     [42, 20, 25],
                                     [63, 86, 30]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))
    mod.run(a, x, y, z)

    assert_equal(a, [[[[42]], [[42]]],
                     [[[13]], [[13]]]])


def test_SwapOccupancy():
    a, x, y, z = create_test_arrays()

    assert_equal(a, [[[[42]], [[42]]],
                     [[[13]], [[13]]]])

    mod = SwapOccupancy(0)
    assert str(mod) == 'SwapOccupancy(swap_site=0)'
    assert mod.number_of_cells == 2
    assert_equal(np.asarray(mod.cells), [[0, 0, 0],
                                         [0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(3, 3, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0],
                                     [1, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_equal(a, [[[[13]], [[42]]],
                     [[[42]], [[13]]]])

    mod.undo_last_run(a, x, y, z)

    assert_equal(a, [[[[42]], [[42]]],
                     [[[13]], [[13]]]])


def test_SwapDisplacement():
    a, x, y, z = create_test_arrays()

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])

    mod = SwapDisplacement(0)
    assert str(mod) == 'SwapDisplacement(swap_site=0)'
    assert mod.number_of_cells == 2
    assert_equal(np.asarray(mod.cells), [[0, 0, 0],
                                         [0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(3, 3, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0],
                                     [1, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[0.11530358]],
                             [[0.03426804]]],
                            [[[-0.17497655]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.02211797]],
                             [[0.05142188]]],
                            [[[0.09813208]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.0458027]],
                             [[0.02550014]]],
                            [[[-0.01894958]],
                             [[0.04351635]]]])

    mod.undo_last_run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])


def test_Swap():
    a, x, y, z = create_test_arrays()

    assert_equal(a, [[[[42]], [[42]]],
                     [[[13]], [[13]]]])

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])

    mod = Swap(0)
    assert str(mod) == 'Swap(swap_site=0)'
    assert mod.number_of_cells == 2
    assert_equal(np.asarray(mod.cells), [[0, 0, 0],
                                         [0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(3, 3, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0],
                                     [1, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_equal(a, [[[[13]], [[42]]],
                     [[[42]], [[13]]]])

    assert_almost_equal(x, [[[[0.11530358]],
                             [[0.03426804]]],
                            [[[-0.17497655]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.02211797]],
                             [[0.05142188]]],
                            [[[0.09813208]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.0458027]],
                             [[0.02550014]]],
                            [[[-0.01894958]],
                             [[0.04351635]]]])

    mod.undo_last_run(a, x, y, z)

    assert_equal(a, [[[[42]], [[42]]],
                     [[[13]], [[13]]]])

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])


def test_ShiftDisplacementRange():
    a, x, y, z = create_test_arrays()

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])

    mod = ShiftDisplacementRange(0, -0.1, 0.1)
    assert str(mod) == 'ShiftDisplacementRange(site=0,minimum=-0.1,maximum=0.1)'
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.19047921]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.03938511]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.06892389]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])

    mod.undo_last_run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])


def test_ShiftDisplacementNormal():
    a, x, y, z = create_test_arrays()

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])

    mod = ShiftDisplacementNormal(0, 0, 0.1)
    assert str(mod) == 'ShiftDisplacementNormal(site=0,mu=0.0,sigma=0.1)'
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.1393564]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[-0.010735]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.0362182]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])

    mod.undo_last_run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])


def test_SetDisplacementRange():
    a, x, y, z = create_test_arrays()

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])

    mod = SetDisplacementRange(0, -0.1, 0.1)
    assert str(mod) == 'SetDisplacementRange(site=0,minimum=-0.1,maximum=0.1)'
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.0155027]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[-0.058747]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.0499743]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])

    mod.undo_last_run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])


def test_SetDisplacementNormal():
    a, x, y, z = create_test_arrays()

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])

    mod = SetDisplacementNormal(0, 0.5, 0.1)
    assert str(mod) == 'SetDisplacementNormal(site=0,mu=0.5,sigma=0.1)'
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[0.5356201]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.3911329]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[0.4827314]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])

    mod.undo_last_run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.17497655]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.09813208]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[-0.01894958]],
                             [[0.02550014]]],
                            [[[-0.0458027]],
                             [[0.04351635]]]])
