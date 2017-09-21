import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from javelin.structure import Structure


def test_empty():
    structure = Structure()
    assert structure.number_of_atoms == 0
    assert_array_equal(structure.element, np.empty(0))
    assert_array_equal(structure.get_atom_symbols(), np.empty(0))
    assert_array_equal(structure.get_atom_count(), 0)
    assert_array_equal(structure.get_chemical_symbols(), np.empty(0))
    assert_array_equal(structure.get_atom_Zs(), np.empty(0))
    assert_array_equal(structure.get_atomic_numbers(), np.empty(0))
    assert_array_equal(structure.x, np.empty(0))
    assert_array_equal(structure.y, np.empty(0))
    assert_array_equal(structure.z, np.empty(0))
    assert_array_equal(structure.xyz, np.empty((0, 3)))
    assert_array_equal(structure.get_scaled_positions(), np.empty((0, 3)))
    assert_array_equal(structure.xyz_cartn, np.empty((0, 3)))
    assert_array_equal(structure.get_positions(), np.empty((0, 3)))
    assert_array_equal(structure.unitcell.cell, (1.0, 1.0, 1.0, 90.0, 90.0, 90.0))


def test_one_atom_init():
    structure = Structure(symbols=['Au'], positions=[[0.5, 0, 0.25]], unitcell=(3, 4, 5))
    assert structure.number_of_atoms == 1
    assert_array_equal(structure.element, ['Au'])
    assert_array_equal(structure.get_atom_symbols(), ['Au'])
    assert_array_equal(structure.get_atom_count(), 1)
    assert_array_equal(structure.get_chemical_symbols(), ['Au'])
    assert_array_equal(structure.get_atom_Zs(), [79])
    assert_array_equal(structure.get_atomic_numbers(), [79])
    assert_array_equal(structure.x, [0.5])
    assert_array_equal(structure.y, [0])
    assert_array_equal(structure.z, [0.25])
    assert_array_equal(structure.xyz, [[0.5, 0, 0.25]])
    assert_array_equal(structure.get_scaled_positions(), [[0.5, 0, 0.25]])
    assert_array_almost_equal(structure.xyz_cartn, [[1.5, 0, 1.25]])
    assert_array_almost_equal(structure.get_positions(), [[1.5, 0, 1.25]])
    assert_array_equal(structure.unitcell.cell, (3.0, 4.0, 5.0, 90.0, 90.0, 90.0))
    av_site = structure.get_average_site(site=0)
    assert av_site['Au']['x'] == 0.5
    assert av_site['Au']['y'] == 0.0
    assert av_site['Au']['z'] == 0.25
    assert av_site['Au']['occ'] == 1.0
    av_stru = structure.get_average_structure()
    assert av_stru[0]['Au']['x'] == 0.5
    assert av_stru[0]['Au']['y'] == 0.0
    assert av_stru[0]['Au']['z'] == 0.25
    assert av_stru[0]['Au']['occ'] == 1.0


def test_one_atom_add():
    structure = Structure()
    structure.unitcell.cell = (3.0, 4.0, 5.0, 90.0, 90.0, 90.0)
    structure.add_atom(symbol='Au', position=[0.5, 0, 0.25])
    assert structure.number_of_atoms == 1
    assert_array_equal(structure.element, ['Au'])
    assert_array_equal(structure.get_atom_symbols(), ['Au'])
    assert_array_equal(structure.get_atom_count(), 1)
    assert_array_equal(structure.get_chemical_symbols(), ['Au'])
    assert_array_equal(structure.get_atom_Zs(), [79])
    assert_array_equal(structure.get_atomic_numbers(), [79])
    assert_array_equal(structure.x, [0.5])
    assert_array_equal(structure.y, [0])
    assert_array_equal(structure.z, [0.25])
    assert_array_equal(structure.xyz, [[0.5, 0, 0.25]])
    assert_array_equal(structure.get_scaled_positions(), [[0.5, 0, 0.25]])
    assert_array_almost_equal(structure.xyz_cartn, [[1.5, 0, 1.25]])
    assert_array_almost_equal(structure.get_positions(), [[1.5, 0, 1.25]])
    assert_array_equal(structure.unitcell.cell, (3.0, 4.0, 5.0, 90.0, 90.0, 90.0))


def test_hex():
    from javelin.unitcell import UnitCell
    positions = [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, 0, 0]]
    symbols = ['C']*6
    unitcell = (1.4, 1.4, 1, 90, 90, 120)

    hex_cell = Structure(unitcell=UnitCell(unitcell),
                         symbols=symbols,
                         positions=positions)

    assert hex_cell.number_of_atoms == 6
    assert_array_equal(hex_cell.element, ['C', 'C', 'C', 'C', 'C', 'C'])
    assert_array_equal(hex_cell.get_atom_symbols(), ['C'])
    assert_array_equal(hex_cell.get_atom_count(), 6)
    assert_array_equal(hex_cell.get_chemical_symbols(), ['C', 'C', 'C', 'C', 'C', 'C'])
    assert_array_equal(hex_cell.get_atom_Zs(), [6])
    assert_array_equal(hex_cell.get_atomic_numbers(), [6, 6, 6, 6, 6, 6])
    assert_array_equal(hex_cell.x, [0,  1,  1,  0, -1, -1])
    assert_array_equal(hex_cell.y, [1,  1,  0, -1, -1,  0])
    assert_array_equal(hex_cell.z, [0, 0, 0, 0, 0, 0])
    assert_array_equal(hex_cell.xyz, positions)
    assert_array_equal(hex_cell.get_scaled_positions(), positions)
    real_positions = [[0, 1.4, 0],
                      [1.21243557, 0.7, 0],
                      [1.21243557, -0.7, 0],
                      [0, -1.4, 0],
                      [-1.21243557, -0.7, 0],
                      [-1.21243557, 0.7, 0]]
    assert_array_almost_equal(hex_cell.xyz_cartn, real_positions)
    assert_array_almost_equal(hex_cell.get_positions(), real_positions)
    assert_array_almost_equal(hex_cell.unitcell.cell, unitcell)


def test_repeat():
    structure = Structure(unitcell=5, symbols=['Au', 'Ag'], positions=[[0, 0, 0], [0.5, 0.5, 0.5]])

    assert structure.number_of_atoms == 2
    assert_array_equal(structure.get_atom_count(), [1, 1])
    assert_array_equal(structure.get_atom_Zs(), [79, 47])
    assert_array_equal(structure.get_atomic_numbers(), [79, 47])
    assert_array_equal(structure.element, ['Au', 'Ag'])
    assert_array_equal(structure.get_atom_symbols(), ['Au', 'Ag'])
    assert_array_equal(structure.get_chemical_symbols(), ['Au', 'Ag'])
    assert_array_equal(structure.xyz, [[0., 0., 0.],
                                       [0.5, 0.5, 0.5]])
    assert_array_almost_equal(structure.xyz_cartn, [[0., 0., 0.],
                                                    [2.5, 2.5, 2.5]])

    structure.repeat((2, 3, 1))

    assert structure.number_of_atoms == 12
    assert_array_equal(structure.get_atom_count(), [6, 6])
    assert_array_equal(structure.get_atom_Zs(), [79, 47])
    assert_array_equal(structure.get_atomic_numbers(), [79, 47, 79, 47, 79, 47,
                                                        79, 47, 79, 47, 79, 47])
    assert_array_equal(structure.element, ['Au', 'Ag', 'Au', 'Ag', 'Au', 'Ag',
                                           'Au', 'Ag', 'Au', 'Ag', 'Au', 'Ag'])
    assert_array_equal(structure.get_atom_symbols(), ['Au', 'Ag'])
    assert_array_equal(structure.get_chemical_symbols(), ['Au', 'Ag', 'Au', 'Ag', 'Au', 'Ag',
                                                          'Au', 'Ag', 'Au', 'Ag', 'Au', 'Ag'])
    assert_array_equal(structure.xyz, np.tile([[0., 0., 0.],
                                               [0.5, 0.5, 0.5]], (6, 1)))
    assert_array_almost_equal(structure.xyz_cartn, [[0.0,  0.0, 0.0],
                                                    [2.5,  2.5, 2.5],
                                                    [0.0,  5.0, 0.0],
                                                    [2.5,  7.5, 2.5],
                                                    [0.0, 10.0, 0.0],
                                                    [2.5, 12.5, 2.5],
                                                    [5.0,  0.0, 0.0],
                                                    [7.5,  2.5, 2.5],
                                                    [5.0,  5.0, 0.0],
                                                    [7.5,  7.5, 2.5],
                                                    [5.0, 10.0, 0.0],
                                                    [7.5, 12.5, 2.5]])

    structure = Structure(unitcell=5, symbols=['Au', 'Ag'], positions=[[0, 0, 0], [0.5, 0.5, 0.5]])
    structure.repeat(2)

    assert structure.number_of_atoms == 16
    assert_array_equal(structure.get_atom_count(), [8, 8])
    assert_array_equal(structure.get_atom_Zs(), [79, 47])
    assert_array_equal(structure.get_atomic_numbers(), [79, 47, 79, 47, 79, 47, 79, 47,
                                                        79, 47, 79, 47, 79, 47, 79, 47])
    assert_array_equal(structure.element, ['Au', 'Ag', 'Au', 'Ag',
                                           'Au', 'Ag', 'Au', 'Ag',
                                           'Au', 'Ag', 'Au', 'Ag',
                                           'Au', 'Ag', 'Au', 'Ag'])
    assert_array_equal(structure.get_atom_symbols(), ['Au', 'Ag'])
    assert_array_equal(structure.get_chemical_symbols(), ['Au', 'Ag', 'Au', 'Ag',
                                                          'Au', 'Ag', 'Au', 'Ag',
                                                          'Au', 'Ag', 'Au', 'Ag',
                                                          'Au', 'Ag', 'Au', 'Ag'])
    assert_array_equal(structure.xyz, np.tile([[0., 0., 0.],
                                               [0.5, 0.5, 0.5]], (8, 1)))
    assert_array_almost_equal(structure.xyz_cartn, [[0.0, 0.0, 0.0],
                                                    [2.5, 2.5, 2.5],
                                                    [0.0, 0.0, 5.0],
                                                    [2.5, 2.5, 7.5],
                                                    [0.0, 5.0, 0.0],
                                                    [2.5, 7.5, 2.5],
                                                    [0.0, 5.0, 5.0],
                                                    [2.5, 7.5, 7.5],
                                                    [5.0, 0.0, 0.0],
                                                    [7.5, 2.5, 2.5],
                                                    [5.0, 0.0, 5.0],
                                                    [7.5, 2.5, 7.5],
                                                    [5.0, 5.0, 0.0],
                                                    [7.5, 7.5, 2.5],
                                                    [5.0, 5.0, 5.0],
                                                    [7.5, 7.5, 7.5]])

    av_site0 = structure.get_average_site(site=0)
    assert av_site0['Au']['x'] == 0.0
    assert av_site0['Au']['y'] == 0.0
    assert av_site0['Au']['z'] == 0.0
    assert av_site0['Au']['occ'] == 1.0
    av_site1 = structure.get_average_site(site=1)
    assert av_site1['Ag']['x'] == 0.5
    assert av_site1['Ag']['y'] == 0.5
    assert av_site1['Ag']['z'] == 0.5
    assert av_site1['Ag']['occ'] == 1.0
    av_stru = structure.get_average_structure()
    assert av_stru[0]['Au']['x'] == 0.0
    assert av_stru[0]['Au']['y'] == 0.0
    assert av_stru[0]['Au']['z'] == 0.0
    assert av_stru[0]['Au']['occ'] == 1.0
    assert av_stru[1]['Ag']['x'] == 0.5
    assert av_stru[1]['Ag']['y'] == 0.5
    assert av_stru[1]['Ag']['z'] == 0.5
    assert av_stru[1]['Ag']['occ'] == 1.0
    av_stru = structure.get_average_structure(separate_sites=False)
    assert av_stru[0]['x'] == 0.0
    assert av_stru[0]['y'] == 0.0
    assert av_stru[0]['z'] == 0.0
    assert av_stru[1]['x'] == 0.5
    assert av_stru[1]['y'] == 0.5
    assert av_stru[1]['z'] == 0.5


def test__getitem__():
    structure = Structure(unitcell=5, symbols=['Au', 'Ag'], positions=[[0, 0, 0], [0.5, 0.5, 0.5]])
    structure.repeat((2, 3, 1))

    structure_slice = structure[:, :, :, :]
    assert structure_slice.unitcell == structure.unitcell
    assert_array_equal(structure_slice.atoms, structure.atoms)

    structure_slice = structure[:, 2, :, :]
    assert structure_slice.unitcell == structure.unitcell
    assert_array_equal(structure_slice.atoms, structure.atoms[::3])

    structure_slice = structure[1, 1, 0, 1]
    assert structure_slice.unitcell == structure.unitcell
    assert_array_equal(structure_slice.atoms, structure.atoms.loc[1, 1, 0, 1])

    structure_slice = structure[:, 1:3, :, 0]
    assert structure_slice.unitcell == structure.unitcell
    assert_array_equal(structure_slice.atoms,
                       structure.atoms.loc[(slice(None), slice(1, 3), slice(None), 0), :])


def test_reindex():
    structure = Structure(unitcell=5, symbols=['Au', 'Ag', 'Pt', 'Pb'], positions=[[0, 0, 0],
                                                                                   [0, 0, 0],
                                                                                   [0, 0, 0],
                                                                                   [0, 0, 0]])

    assert_array_equal(structure.atoms.index.tolist(), [(0, 0, 0, 0),
                                                        (0, 0, 0, 1),
                                                        (0, 0, 0, 2),
                                                        (0, 0, 0, 3)])
    assert_array_almost_equal(structure.xyz_cartn, [[0., 0., 0.],
                                                    [0., 0., 0.],
                                                    [0., 0., 0.],
                                                    [0., 0., 0.]])
    structure.reindex([2, 2, 1, 1])
    assert_array_equal(structure.atoms.index.tolist(), [(0, 0, 0, 0),
                                                        (0, 1, 0, 0),
                                                        (1, 0, 0, 0),
                                                        (1, 1, 0, 0)])
    assert_array_almost_equal(structure.xyz_cartn, [[0., 0., 0.],
                                                    [0., 5., 0.],
                                                    [5., 0., 0.],
                                                    [5., 5., 0.]])
    structure.reindex([1, 1, 2, 2])
    assert_array_equal(structure.atoms.index.tolist(), [(0, 0, 0, 0),
                                                        (0, 0, 0, 1),
                                                        (0, 0, 1, 0),
                                                        (0, 0, 1, 1)])
    assert_array_almost_equal(structure.xyz_cartn, [[0., 0., 0.],
                                                    [0., 0., 0.],
                                                    [0., 0., 5.],
                                                    [0., 0., 5.]])

    av_stru = structure.get_average_structure()
    assert av_stru[0]['Au']['x'] == 0.0
    assert av_stru[0]['Au']['y'] == 0.0
    assert av_stru[0]['Au']['z'] == 0.0
    assert av_stru[0]['Au']['occ'] == 0.5
    assert av_stru[0]['Pt']['x'] == 0.0
    assert av_stru[0]['Pt']['y'] == 0.0
    assert av_stru[0]['Pt']['z'] == 0.0
    assert av_stru[0]['Pt']['occ'] == 0.5
    av_stru = structure.get_average_structure(separate_sites=False)
    assert av_stru[0]['x'] == 0.0
    assert av_stru[0]['y'] == 0.0
    assert av_stru[0]['z'] == 0.0


def test_get_occupational_correlation():
    structure = Structure(numbers=[11, 17]*8, positions=[[0, 0, 0]]*16, ncells=[4, 4, 1, 1])
    assert structure.get_occupational_correlation([[0, 0, 1, 0, 0]], 11) == 1
    assert structure.get_occupational_correlation([[0, 0, 0, 1, 0]], 11) == -1
    assert structure.get_occupational_correlation([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]], 11) == 0

    structure = Structure(numbers=[11, 17, 11, 11, 11, 17, 11, 11, 17, 11, 11, 11],
                          positions=[[0, 0, 0]]*12, ncells=[12, 1, 1, 1])
    assert structure.get_occupational_correlation([[0, 0, 1, 0, 0]], 11) == -0.3333333333333333


def test_get_displacement_correlation():
    structure = Structure(numbers=[11, 17]*8,
                          positions=[[0.01, 0.02, 0], [-0.01, 0.02, 0]]*8,
                          ncells=[4, 4, 1, 1])
    assert structure.get_displacement_correlation([[0, 0, 1, 0, 0]]) == 1.0
    assert structure.get_displacement_correlation([[0, 0, 0, 1, 0]]) == -1.0
    assert structure.get_displacement_correlation([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]) == 0.0


def test_except():
    with pytest.raises(ValueError):
        Structure(symbols=['U'], positions=[[0, 0, 0]], ncells=[1, 1, 1, 2])

    structure = Structure()
    with pytest.raises(ValueError):
        structure.add_atom(symbol='U')


def test_axisAngle2Versor():
    from javelin.structure import axisAngle2Versor
    assert_array_equal(axisAngle2Versor(1, 0, 0, 0),
                       [1.0, 0.0, 0.0, 0.0])
    assert_array_equal(axisAngle2Versor(1, 1, 1, 0), [1.0, 0.0, 0.0, 0.0])
    assert_array_equal(axisAngle2Versor(10, -3, -3.5, 0), [1.0, 0.0, 0.0, 0.0])
    assert_array_almost_equal(axisAngle2Versor(1, 0, 0, 90),
                              [1/np.sqrt(2), 1/np.sqrt(2), 0.0, 0.0])
    temp = np.sin(np.deg2rad(45))/np.sqrt(3)
    assert_array_almost_equal(axisAngle2Versor(1, 1, 1, 90),
                              [1/np.sqrt(2), temp, temp, temp])

    with pytest.raises(ValueError):
        axisAngle2Versor(0, 0, 0, 0)


def test_get_rotation_matrix():
    from javelin.structure import get_rotation_matrix
    assert_array_equal(get_rotation_matrix(1, 0, 0, 0), np.eye(3))
    assert_array_almost_equal(get_rotation_matrix(1, 0, 0, 90),
                              [[1, 0, 0],
                               [0, 0, 1],
                               [0, -1, 0]])
    assert_array_almost_equal(get_rotation_matrix(1, 1, 1, 90),
                              [[0.33333333, 0.9106836, -0.24401694],
                               [-0.24401694, 0.33333333, 0.9106836],
                               [0.9106836, -0.24401694, 0.33333333]])


def test_get_rotation_matrix_from_versor():
    from javelin.structure import get_rotation_matrix_from_versor
    assert_array_equal(get_rotation_matrix_from_versor(1, 0, 0, 0), np.eye(3))
    assert_array_almost_equal(get_rotation_matrix_from_versor(1/np.sqrt(2), 1/np.sqrt(2), 0.0, 0.0),
                              [[1, 0, 0],
                               [0, 0, 1],
                               [0, -1, 0]])
    temp = np.sin(np.deg2rad(45))/np.sqrt(3)
    assert_array_almost_equal(get_rotation_matrix_from_versor(1/np.sqrt(2), temp, temp, temp),
                              [[0.33333333, 0.9106836, -0.24401694],
                               [-0.24401694, 0.33333333, 0.9106836],
                               [0.9106836, -0.24401694, 0.33333333]])
