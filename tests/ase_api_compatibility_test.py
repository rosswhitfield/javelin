"""Test that the api of javelin.structure.Structure is compatible that
of diffpy.Structure.Structure and ase.Atoms

To maintain api compatibility between ASE and javelin structures
object the following methods must return the same thing:
    get_scaled_positions
    get_atomic_numbers
    get_magnetic_moments

And be able to get the same unitcell from both structures.
"""
import pytest
import os
from numpy.testing import assert_array_equal, assert_array_almost_equal
from javelin.structure import Structure
ase = pytest.importorskip("ase")


def test_hex():
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


def test_read_stru_znse():
    from javelin.io import read_stru, read_stru_to_ase
    filename = os.path.join(os.path.dirname(__file__), 'data', 'znse.cell')
    znse_javelin = read_stru(filename)
    znse_ase = read_stru_to_ase(filename)
    assert len(znse_ase) == 2
    assert_array_almost_equal(znse_ase.get_cell(), [[3.997, 0, 0],
                                                    [-1.9985, 3.461504, 0],
                                                    [0, 0, 6.501]])
    assert_array_equal(znse_ase.get_scaled_positions(),
                       [[0.3333333, 0.6666667, 0.3671],
                        [0.3333333, 0.6666667, 0.]])
    assert znse_ase.get_chemical_formula() == 'SeZn'

    # unitcell
    assert_array_almost_equal(znse_javelin.unitcell.cell,
                              ase.geometry.cell_to_cellpar(znse_ase.cell))
    # get_atomic_numbers
    assert_array_equal(znse_javelin.get_atomic_numbers(),
                       znse_ase.get_atomic_numbers())
    # get_positions
    # assert_array_almost_equal(znse_javelin.get_positions(),
    #                          znse_ase.get_positions())
    # get_scaled_positions
    assert_array_almost_equal(znse_javelin.get_scaled_positions(),
                              znse_ase.get_scaled_positions())


def test_read_stru_pzn():
    from javelin.io import read_stru, read_stru_to_ase
    filename = os.path.join(os.path.dirname(__file__), 'data', 'pzn.stru')
    pzn_javelin = read_stru(filename, starting_cell=(0, 0, 0))
    pzn_ase = read_stru_to_ase(filename)
    assert len(pzn_ase) == 15
    assert_array_almost_equal(pzn_ase.get_cell(), [[4.06, 0, 0],
                                                   [0, 4.06, 0],
                                                   [0, 0, 4.06]])
    assert pzn_ase.get_chemical_formula() == 'Nb2O9Pb3Zn'

    # unitcell
    assert_array_equal(pzn_javelin.unitcell.cell,
                       ase.geometry.cell_to_cellpar(pzn_ase.cell))
    # get_atomic_numbers
    assert_array_equal(pzn_javelin.get_atomic_numbers(),
                       pzn_ase.get_atomic_numbers())
    # get_positions
    assert_array_almost_equal(pzn_javelin.get_positions(),
                              pzn_ase.get_positions())
    # get_scaled_positions
    assert_array_almost_equal(pzn_javelin.get_scaled_positions(),
                              pzn_ase.get_scaled_positions())


def test_read_stru_missing_cell():
    from javelin.io import read_stru, read_stru_to_ase
    filename = os.path.join(os.path.dirname(__file__), 'data', 'missing_cell.cell')
    c_javelin = read_stru(filename)
    c_ase = read_stru_to_ase(filename)
    assert len(c_ase) == 1
    assert_array_equal(c_ase.get_cell(), [[1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 1]])
    assert_array_equal(c_ase.get_scaled_positions(), [[0.5, 0., 0.25]])
    assert c_ase.get_chemical_formula() == 'C'

    # unitcell
    assert_array_equal(c_javelin.unitcell.cell,
                       ase.geometry.cell_to_cellpar(c_ase.cell))
    # get_atomic_numbers
    assert_array_equal(c_javelin.get_atomic_numbers(),
                       c_ase.get_atomic_numbers())
    # get_positions
    assert_array_almost_equal(c_javelin.get_positions(),
                              c_ase.get_positions())
    # get_scaled_positions
    assert_array_almost_equal(c_javelin.get_scaled_positions(),
                              c_ase.get_scaled_positions())


def test_ase_to_javelin():
    positions = [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, 0, 0]]
    symbols = ['C']*6
    unitcell = (1.4, 1.4, 1, 90, 90, 120)

    hex_ase = ase.Atoms(symbols=symbols, scaled_positions=positions,
                        cell=ase.geometry.cellpar_to_cell(unitcell))

    hex_javelin = Structure(hex_ase)

    # unitcell
    assert_array_equal(hex_javelin.unitcell.cell,
                       ase.geometry.cell_to_cellpar(hex_ase.cell))
    # get_atomic_numbers
    assert_array_equal(hex_javelin.get_atomic_numbers(),
                       hex_ase.get_atomic_numbers())


def test_javelin_to_ase():
    positions = [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, 0, 0]]
    symbols = ['C']*6
    unitcell = (1.4, 1.4, 1, 90, 90, 120)

    hex_javelin = Structure(symbols=symbols,
                            unitcell=unitcell,
                            positions=positions)

    hex_ase = hex_javelin.to_ase()

    # unitcell
    assert_array_equal(hex_javelin.unitcell.cell,
                       ase.geometry.cell_to_cellpar(hex_ase.cell))
    # get_atomic_numbers
    assert_array_equal(hex_javelin.get_atomic_numbers(),
                       hex_ase.get_atomic_numbers())


def test_ase_plot_atoms():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use('Agg')

    from ase.visualize.plot import plot_atoms

    positions = [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, 0, 0]]
    symbols = ['C']*6
    unitcell = (1.4, 1.4, 1, 90, 90, 120)

    structure = Structure(symbols=symbols,
                          unitcell=unitcell,
                          positions=positions)

    ax = plot_atoms(structure)

    assert isinstance(ax, matplotlib.axes.Subplot)
