"""Test that the api of javelin.structure.Structure is compatible that
of diffpy.Structure.Structure

To maintain api compatibility between diffpy and javelin structures
object the following properties must return the same thing:
    xyz
    xyz_cartn
    element

And be able to get the same unitcell from both structures.
"""
import pytest
import os
import numpy as np
from javelin.structure import Structure
from numpy.testing import assert_array_equal, assert_array_almost_equal
dps = pytest.importorskip("diffpy.structure")


def test_hex():
    positions = [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, 0, 0]]
    symbols = ['C']*6
    unitcell = (1.4, 1.4, 1, 90, 90, 120)

    hex_javelin = Structure(unitcell=unitcell,
                            symbols=symbols,
                            positions=positions)

    hex_diffpy = dps.Structure(atoms=[dps.Atom(atype='C', xyz=xyz) for xyz in positions],
                               lattice=dps.Lattice(1.4, 1.4, 1, 90, 90, 120))
    assert len(hex_diffpy) == 6

    # unitcell
    assert_array_almost_equal(hex_javelin.unitcell.cell,
                              hex_diffpy.lattice.abcABG())
    # element
    assert_array_equal(hex_javelin.element,
                       np.array(hex_diffpy.element))
    # xyz
    assert_array_almost_equal(hex_javelin.xyz,
                              hex_diffpy.xyz)
    # xyz_cartn
    assert_array_almost_equal(hex_javelin.xyz_cartn,
                              hex_diffpy.xyz_cartn)


def test_read_stru_znse():
    from javelin.io import read_stru
    filename = os.path.join(os.path.dirname(__file__), 'data', 'znse.cell')
    znse_javelin = read_stru(filename)
    znse_diffpy = dps.Structure()
    znse_diffpy.read(filename, 'discus')
    assert len(znse_diffpy) == 2

    # unitcell
    assert_array_almost_equal(znse_javelin.unitcell.cell,
                              znse_diffpy.lattice.abcABG())
    # element
    assert_array_equal(znse_javelin.element,
                       np.array(znse_diffpy.element))
    # xyz
    assert_array_almost_equal(znse_javelin.xyz,
                              znse_diffpy.xyz)
    # xyz_cartn
    assert_array_almost_equal(znse_javelin.xyz_cartn,
                              znse_diffpy.xyz_cartn)


def test_read_stru_pzn():
    from javelin.io import read_stru
    filename = os.path.join(os.path.dirname(__file__), 'data', 'pzn.stru')
    pzn_javelin = read_stru(filename, starting_cell=(0, 0, 0))
    pzn_diffpy = dps.Structure()
    pzn_diffpy.read(filename, 'discus')
    assert len(pzn_diffpy) == 15

    # element
    assert_array_equal(pzn_javelin.element,
                       np.array(pzn_diffpy.element))
    # xyz_cartn
    assert_array_almost_equal(pzn_javelin.xyz_cartn,
                              pzn_diffpy.xyz_cartn)


def test_diffpy_to_javelin():
    positions = [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, 0, 0]]

    hex_diffpy = dps.Structure(atoms=[dps.Atom(atype='C', xyz=xyz) for xyz in positions],
                               lattice=dps.Lattice(1.4, 1.4, 1, 90, 90, 120))

    hex_javelin = Structure(hex_diffpy)

    assert len(hex_diffpy) == 6

    # unitcell
    assert_array_almost_equal(hex_javelin.unitcell.cell,
                              hex_diffpy.lattice.abcABG())
    # element
    assert_array_equal(hex_javelin.element,
                       np.array(hex_diffpy.element))
    # xyz
    assert_array_almost_equal(hex_javelin.xyz,
                              hex_diffpy.xyz)
    # xyz_cartn
    assert_array_almost_equal(hex_javelin.xyz_cartn,
                              hex_diffpy.xyz_cartn)
