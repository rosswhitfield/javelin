import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal


def test_read_stru_znse():
    ase = pytest.importorskip("javelin.ase")
    znse = ase.read_stru('tests/znse.cell')
    assert len(znse) == 2
    assert_array_almost_equal(znse.get_cell(), [[3.997, 0, 0],
                                                [-1.9985, 3.461504, 0],
                                                [0, 0, 6.501]])
    assert_array_equal(znse.get_scaled_positions(),
                       [[0.3333333, 0.6666667, 0.3671],
                        [0.3333333, 0.6666667, 0.]])
    assert znse.get_chemical_formula() == 'SeZn'


def test_read_stru_pzn():
    ase = pytest.importorskip("javelin.ase")
    pzn = ase.read_stru('tests/pzn.stru')
    assert len(pzn) == 5
    assert_array_almost_equal(pzn.get_cell(), [[4.06, 0, 0],
                                               [0, 4.06, 0],
                                               [0, 0, 4.06]])
    assert_array_almost_equal(pzn.get_scaled_positions(),
                              [[0.,  0.,  0.],
                               [0.5, 0.5, 0.5],
                               [0.5, 0.5, 1.],
                               [0.5, 0.,  1.5],
                               [0.,  0.5, 1.5]])
    assert pzn.get_chemical_formula() == 'NbO3Pb'


def test_read_stru_missing_cell():
    ase = pytest.importorskip("javelin.ase")
    pzn = ase.read_stru('tests/missing_cell.cell')
    assert len(pzn) == 1
    assert_array_equal(pzn.get_cell(), [[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]])
    assert_array_equal(pzn.get_scaled_positions(), [[0.5, 0., 0.25]])
    assert pzn.get_chemical_formula() == 'C'
