import pytest
from numpy.testing import assert_array_equal
from javelin.utils import get_atomic_number_symbol, get_unitcell, get_positions, get_atomic_numbers


def test_one():
    Z, symbol = get_atomic_number_symbol(Z=8)
    assert Z[0] == 8
    assert symbol[0] == 'O'

    Z, symbol = get_atomic_number_symbol(symbol='C')
    assert Z[0] == 6
    assert symbol[0] == 'C'

    Z, symbol = get_atomic_number_symbol(Z=10, symbol='Ne')
    assert Z[0] == 10
    assert symbol[0] == 'Ne'


def test_array():
    Z, symbol = get_atomic_number_symbol(Z=[6, 7, 8, 0])
    assert_array_equal(Z, [6, 7, 8, 0])
    assert_array_equal(symbol, ['C', 'N', 'O', 'VD'])

    Z, symbol = get_atomic_number_symbol(symbol=['Au', 'Ag', 'VD'])
    assert_array_equal(Z, [79, 47, 0])
    assert_array_equal(symbol, ['Au', 'Ag', 'VD'])

    Z, symbol = get_atomic_number_symbol(Z=[1, 2, 3], symbol=['B', 'C', 'F'])
    assert_array_equal(Z, [5, 6, 9])
    assert_array_equal(symbol, ['B', 'C', 'F'])


def test_raises():
    with pytest.raises(ValueError):
        get_atomic_number_symbol()

    with pytest.raises(ValueError):
        get_atomic_number_symbol(symbol='A')

    with pytest.raises(KeyError):
        get_atomic_number_symbol(Z=1000)

    with pytest.raises(ValueError):
        get_unitcell(0)

    with pytest.raises(ValueError):
        get_positions(0)

    with pytest.raises(ValueError):
        get_atomic_numbers(0)
