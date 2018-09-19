import pytest
from javelin.unitcell import UnitCell
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal


def test_UnitCell_init():
    from numpy import pi
    unitcell = UnitCell()
    assert unitcell.cell == (1, 1, 1, 90, 90, 90)
    assert unitcell.reciprocalCell == (1, 1, 1, 90, 90, 90)
    assert unitcell.d(1, 0, 0) == 1
    assert unitcell.dstar(1, 0, 0) == 1
    assert unitcell.recAngle(1, 0, 0, 1, 0, 0) == 0
    assert unitcell.recAngle(1, 0, 0, 0, 1, 0, degrees=True) == 90
    assert_almost_equal(unitcell.recAngle(0, 1, 0, 0, 1, 1), pi/4)
    assert_almost_equal(unitcell.recAngle(0, 1, 0, 0, 1, 1, degrees=True), 45)
    assert unitcell.volume == 1
    assert unitcell.reciprocalVolume == 1
    assert_array_equal(unitcell.G, [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
    assert_array_equal(unitcell.B, [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
    assert_array_equal(unitcell.Binv, [[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])

    unitcell = UnitCell(5)
    assert unitcell.cell == (5, 5, 5, 90, 90, 90)
    assert_array_almost_equal(unitcell.reciprocalCell, (0.2, 0.2, 0.2, 90, 90, 90))
    assert_almost_equal(unitcell.volume, 125)
    assert_almost_equal(unitcell.reciprocalVolume, 0.008)
    assert_array_almost_equal(unitcell.G, [[25, 0, 0],
                                           [0, 25, 0],
                                           [0, 0, 25]])
    assert_array_almost_equal(unitcell.B, [[0.2, 0, 0],
                                           [0, 0.2, 0],
                                           [0, 0, 0.2]])
    assert_array_almost_equal(unitcell.Binv, [[5, 0, 0],
                                              [0, 5, 0],
                                              [0, 0, 5]])

    unitcell = UnitCell(1, 2, 3)
    assert unitcell.cell == (1, 2, 3, 90, 90, 90)
    assert_array_almost_equal(unitcell.reciprocalCell, (1, 0.5, 0.333333, 90, 90, 90))
    assert unitcell.volume == 6
    assert_almost_equal(unitcell.reciprocalVolume, 0.1666667)
    assert_array_almost_equal(unitcell.G, [[1, 0, 0],
                                           [0, 4, 0],
                                           [0, 0, 9]])

    unitcell = UnitCell(4, 5, 6, 90, 90, 120)
    assert unitcell.d(1, 0, 0) == 3.4641016151377548
    assert unitcell.d(0, 1, 0) == 4.3301270189221936
    assert unitcell.d(0, 0, 1) == 6
    assert_almost_equal(unitcell.recAngle(1, 0, 0, 1, 0, 0), 0)
    assert_almost_equal(unitcell.recAngle(1, 0, 0, 1, 0, 0, degrees=True), 0, decimal=5)
    assert_almost_equal(unitcell.recAngle(1, 0, 0, 0, 1, 0), pi/3)
    assert_almost_equal(unitcell.recAngle(1, 0, 0, 0, 1, 0, degrees=True), 60)
    assert_almost_equal(unitcell.recAngle(1, 0, 0, 0, 0, 1), pi/2)
    assert_almost_equal(unitcell.recAngle(1, 0, 0, 0, 0, 1, degrees=True), 90)
    assert_array_almost_equal(unitcell.cell,
                              (4, 5, 6, 90, 90, 120))
    assert_array_almost_equal(unitcell.reciprocalCell,
                              (0.288675, 0.2309401, 0.1666667, 90, 90, 60))
    assert_almost_equal(unitcell.volume, 103.9230485)
    assert_almost_equal(unitcell.reciprocalVolume, 0.0096225)
    assert_array_almost_equal(unitcell.G, [[16, -10, 0],
                                           [-10, 25, 0],
                                           [0,    0, 36]])
    assert_array_almost_equal(unitcell.B, [[0.288675, 0.11547, 0],
                                           [0,        0.2,     0],
                                           [0,        0,       0.166667]])

    assert_array_almost_equal(unitcell.Binv, [[3.464101, -2, 0],
                                              [0,         5, 0],
                                              [0,         0, 6]])

    assert_array_almost_equal(unitcell.cartesian([1, 0, 0]), [3.464102, -2, 0])
    assert_array_almost_equal(unitcell.cartesian([[1,   0,   0],
                                                  [0.1, 0.3, 0.5]]), [[3.464102, -2, 0],
                                                                      [0.34641, 1.3, 3]])
    assert_array_almost_equal(unitcell.fractional([3.464102, -2, 0]), [1, 0, 0])
    assert_array_almost_equal(unitcell.fractional([[0, 5, 0],
                                                   [0, 0, 3]]), [[0, 1, 0],
                                                                 [0, 0, 0.5]])

    # test __eq__
    assert unitcell != UnitCell()
    assert unitcell != UnitCell(5)
    assert unitcell == UnitCell(4, 5, 6, 90, 90, 120)
    assert unitcell != UnitCell(6, 5, 4, 120, 90, 90)

    unitcell = UnitCell([5, 6, 7, 89, 92, 121])
    assert unitcell.cell == (5, 6, 7, 89, 92, 121)
    assert_array_almost_equal(unitcell.reciprocalCell,
                              (0.233433, 0.194439, 0.142944, 89.965076, 88.267509, 59.014511))
    assert_almost_equal(unitcell.volume, 179.8954455)
    assert_almost_equal(unitcell.reciprocalVolume, 0.0055587844)
    assert_array_almost_equal(unitcell.G, [[25, -15.45114225, -1.22148238],
                                           [-15.45114225, 36, 0.73300107],
                                           [-1.22148238, 0.73300107, 49]])
    # Test __str__
    assert str(unitcell) == 'a=5.0, b=6.0, c=7.0, alpha=89.0, beta=92.0, gamma=121.0'


def test_UnitCell_cell_setter():
    unitcell = UnitCell()

    unitcell.cell = 7
    assert unitcell.cell == (7, 7, 7, 90, 90, 90)
    assert_almost_equal(unitcell.volume, 343)
    assert_array_almost_equal(unitcell.G, [[49, 0, 0],
                                           [0, 49, 0],
                                           [0, 0, 49]])

    unitcell.cell = 4, 5, 6
    assert unitcell.cell == (4, 5, 6, 90, 90, 90)
    unitcell.cell = [6, 5, 4]
    assert unitcell.cell == (6, 5, 4, 90, 90, 90)
    unitcell.cell = (6, 4, 5)
    assert unitcell.cell == (6, 4, 5, 90, 90, 90)
    unitcell.cell = 7, 6, 5, 120, 90, 45
    assert_array_almost_equal(unitcell.cell,
                              (7, 6, 5, 120, 90, 45))


def test_UnitCell_exceptions():
    unitcell = UnitCell()

    with pytest.raises(ValueError):
        unitcell.cell = (1, 2)

    with pytest.raises(ValueError):
        unitcell.cell = (1, 2, 3, 4, 5)

    with pytest.raises(Exception):
        unitcell.cell = "foobor"

    with pytest.raises(ValueError):
        UnitCell(1, 1, 1, 90, 90, 200)

    with pytest.raises(ValueError):
        UnitCell(1, 1, 1, 360, 90, 90)
