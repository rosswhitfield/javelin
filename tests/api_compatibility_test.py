"""Test that the api of javelin.structure.Structure is compatible that
of diffpy.Structure.Structure and ase.Atoms
"""
import pytest
from javelin.structure import Structure
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_ase():
    """To maintain api compatibility between ASE and javelin structures
    object the following methods must return the same thing:
    get_positions
    get_scaled_positions
    get_atomic_numbers
    get_magnetic_moments

    And be able to get the same unitcell from both structures.
    """
    ase = pytest.importorskip("ase")

    positions = [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, 0, 0]]
    symbols = ['C']*6
    unitcell = (1.4, 1.4, 1, 90, 90, 120)

    hex_javelin = Structure(unitcell=unitcell,
                            symbols=symbols,
                            positions=positions)

    hex_ase = ase.Atoms(symbols=symbols, scaled_positions=positions, cell=hex_javelin.unitcell.Binv)

    assert len(hex_ase) == 6

    # unitcell
    assert_array_equal(hex_javelin.unitcell.cell,
                       ase.geometry.cell_to_cellpar(hex_ase.cell))
    # get_atomic_numbers
    assert_array_equal(hex_javelin.get_atomic_numbers(),
                       hex_ase.get_atomic_numbers())
    # get_positions
    assert_array_almost_equal(hex_javelin.get_positions(),
                              hex_ase.get_positions())
    # get_scaled_positions
    assert_array_almost_equal(hex_javelin.get_scaled_positions(),
                              hex_ase.get_scaled_positions())


def test_diffpy():
    """To maintain api compatibility between diffpy and javelin structures
    object the following properties must return the same thing:
    xyz
    xyz_cartn
    element

    And be able to get the same unitcell from both structures.
    """
    dps = pytest.importorskip("diffpy.Structure")
    import numpy as np

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
