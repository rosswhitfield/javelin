from javelin.utils import unit_cell_to_vectors, unit_vectors_to_cell
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal


class Test_unit_cell_to_vectors:
    def test_cubic(self):
        cell = unit_cell_to_vectors(5, 5, 5, 90, 90, 90)
        assert_array_equal(cell, [[5, 0, 0],
                                  [0, 5, 0],
                                  [0, 0, 5]])

    def test_orthorhombic(self):
        cell = unit_cell_to_vectors(5, 6, 7, 90, 90, 90)
        assert_array_equal(cell, [[5, 0, 0],
                                  [0, 6, 0],
                                  [0, 0, 7]])

    def test_hexagonal(self):
        cell = unit_cell_to_vectors(5, 5, 7, 90, 90, 120)
        assert_array_almost_equal(cell, [[5,    0,          0],
                                         [-2.5, 4.33012702, 0],
                                         [0,    0,          7]])


class Test_unit_vectors_to_cell:
    def test_cubic(self):
        cell = [[5, 0, 0],
                [0, 5, 0],
                [0, 0, 5]]
        a, b, c, alpha, beta, gamma = unit_vectors_to_cell(cell)
        assert a == 5
        assert b == 5
        assert c == 5
        assert alpha == 90
        assert beta == 90
        assert gamma == 90

    def test_orthorhombic(self):
        cell = [[5, 0, 0],
                [0, 6, 0],
                [0, 0, 7]]
        a, b, c, alpha, beta, gamma = unit_vectors_to_cell(cell)
        assert a == 5
        assert b == 6
        assert c == 7
        assert alpha == 90
        assert beta == 90
        assert gamma == 90

    def test_hexagonal(self):
        cell = [[5,    0,          0],
                [-2.5, 4.33012702, 0],
                [0,    0,          7]]
        a, b, c, alpha, beta, gamma = unit_vectors_to_cell(cell)
        assert a == 5
        assert_almost_equal(b, 5)
        assert c == 7
        assert alpha == 90
        assert beta == 90
        assert_almost_equal(gamma, 120)
