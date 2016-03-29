from javelin.unitcell import UnitCell
from numpy.testing import assert_array_almost_equal


def test_UnitCell_init():
    unitcell = UnitCell()
    assert unitcell.cell == (1, 1, 1, 90, 90, 90)
    unitcell = UnitCell(5)
    assert unitcell.cell == (5, 5, 5, 90, 90, 90)
    unitcell = UnitCell(1, 2, 3)
    assert unitcell.cell == (1, 2, 3, 90, 90, 90)
    unitcell = UnitCell(4, 5, 6, 90, 91, 120)
    assert_array_almost_equal(unitcell.cell,
                              (4, 5, 6, 90, 91, 120))
    unitcell = UnitCell([5, 6, 7, 89, 92, 121])
    assert unitcell.cell == (5, 6, 7, 89, 92, 121)


def test_UnitCell_cell_setter():
    unitcell = UnitCell()
    unitcell.cell = 7
    assert unitcell.cell == (7, 7, 7, 90, 90, 90)
    unitcell.cell = 4, 5, 6
    assert unitcell.cell == (4, 5, 6, 90, 90, 90)
    unitcell.cell = [6, 5, 4]
    assert unitcell.cell == (6, 5, 4, 90, 90, 90)
    unitcell.cell = 7, 6, 5, 120, 90, 45
    assert_array_almost_equal(unitcell.cell,
                              (7, 6, 5, 120, 90, 45))
