import numpy as np
import pytest
import sys
from numpy.testing import assert_equal, assert_almost_equal
from javelin.modifier import (BaseModifier, SwapOccupancy,
                              SwapDisplacement, Swap,
                              ShiftDisplacementRange,
                              ShiftDisplacementNormal,
                              SetDisplacementRange,
                              SetDisplacementNormal,
                              ShiftDisplacementRangeXYZ,
                              ShiftDisplacementNormalXYZ,
                              SetDisplacementRangeXYZ,
                              SetDisplacementNormalXYZ)
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

    mod = BaseModifier(3, [1, 3, 5, 7])
    assert str(mod) == 'BaseModifier(number_of_cells=3,sites=[1 3 5 7])'
    assert mod.number_of_cells == 3
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(100, 100, 100)
    assert_equal(np.asarray(cells), [[3, 32, 69,  3],
                                     [20, 25, 63,  7],
                                     [30,  2, 36,  7]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))
    mod.run(a, x, y, z)

    assert_equal(a, [[[[42]], [[42]]],
                     [[[13]], [[13]]]])


def test_SwapOccupancy():
    a, x, y, z = create_test_arrays()

    assert_equal(a, [[[[42]], [[42]]],
                     [[[13]], [[13]]]])

    mod = SwapOccupancy(0)
    assert str(mod) == 'SwapOccupancy(swap_sites=[0])'
    assert mod.number_of_cells == 2
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0],
                                         [0, 0, 0, 0]])
    set_seed(16)
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0, 0],
                                     [1, 0, 0, 0]])
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
    assert str(mod) == 'SwapDisplacement(swap_sites=[0])'
    assert mod.number_of_cells == 2
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0],
                                         [0, 0, 0, 0]])
    set_seed(16)
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0, 0],
                                     [1, 0, 0, 0]])
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
    assert str(mod) == 'Swap(swap_sites=[0])'
    assert mod.number_of_cells == 2
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0],
                                         [0, 0, 0, 0]])
    set_seed(16)
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0, 0],
                                     [1, 0, 0, 0]])
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
    assert str(mod) == 'ShiftDisplacementRange(sites=[0],minimum=-0.1,maximum=0.1)'
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.23372352]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.04815777]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[0.00836212]],
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
    assert str(mod) == 'ShiftDisplacementNormal(sites=[0],mu=0.0,sigma=0.1)'
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.17511995]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.1603671]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[0.13397637]],
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
    assert str(mod) == 'SetDisplacementRange(sites=[0],minimum=-0.1,maximum=0.1)'
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.05874697]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[-0.04997431]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[0.0273117]],
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
    assert str(mod) == 'SetDisplacementNormal(sites=[0],mu=0.5,sigma=0.1)'
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[0.49985659]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.56223502]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[0.65292595]],
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


def test_ShiftDisplacementRangeXYZ():
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

    mod = ShiftDisplacementRangeXYZ(0, -0.6, -0.4, -0.1, 0.1, 0.4, 0.6)
    assert (str(mod) ==
            "ShiftDisplacementRangeXYZ(sites=[0],"
            "min_x=-0.6,max_x=-0.4,min_y=-0.1,max_y=0.1,min_z=0.4,max_z=0.6)")
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.73372352]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.04815777]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[0.50836212]],
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


def test_ShiftDisplacementNormalXYZ():
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

    mod = ShiftDisplacementNormalXYZ(0, -0.5, 0.05, 0, 0.1, 0.1, 0.2)
    assert (str(mod) == "ShiftDisplacementNormalXYZ(sites=[0],"
            "mu_x=-0.5,sigma_x=0.05,mu_y=0.0,sigma_y=0.1,mu_z=0.1,sigma_z=0.2)")
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.67504825]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.1603671]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[0.38690232]],
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


def test_SetDisplacementRangeXYZ():
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

    mod = SetDisplacementRangeXYZ(0, -0.6, -0.4, -0.1, 0.1, 0.4, 0.6)
    assert (str(mod) == "SetDisplacementRangeXYZ(sites=[0],"
            "min_x=-0.6,max_x=-0.4,min_y=-0.1,max_y=0.1,min_z=0.4,max_z=0.6)")
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.55874697]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[-0.04997431]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[0.5273117]],
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


def test_SetDisplacementNormalXYZ():
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

    mod = SetDisplacementNormalXYZ(0, -0.5, 0.05, 0, 0.1, 0.1, 0.2)
    assert (str(mod) == "SetDisplacementNormalXYZ(sites=[0],"
            "mu_x=-0.5,sigma_x=0.05,mu_y=0.0,sigma_y=0.1,mu_z=0.1,sigma_z=0.2)")
    assert mod.number_of_cells == 1
    assert_equal(np.asarray(mod.cells), [[0, 0, 0, 0]])
    set_seed()
    cells = mod.get_random_cells(2, 2, 1)
    assert_equal(np.asarray(cells), [[0, 0, 0, 0]])
    assert_equal(np.asarray(mod.cells), np.asarray(cells))

    mod.run(a, x, y, z)

    assert_almost_equal(x, [[[[-0.5000717]],
                             [[0.03426804]]],
                            [[[0.11530358]],
                             [[-0.0252436]]]])

    assert_almost_equal(y, [[[[0.06223502]],
                             [[0.05142188]]],
                            [[[0.02211797]],
                             [[-0.10700433]]]])

    assert_almost_equal(z, [[[[0.4058519]],
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
