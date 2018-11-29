import pytest
from javelin.fourier import Fourier
from javelin.structure import Structure
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose
import numpy as np


def test_Fourier_init():
    four = Fourier()
    assert four.radiation == 'neutron'
    assert four.grid.bins == (101, 101, 1)
    assert_array_equal(four.grid.ll, [0.0, 0.0, 0.0])
    assert_array_equal(four.grid.lr, [1.0, 0.0, 0.0])
    assert_array_equal(four.grid.ul, [0.0, 1.0, 0.0])
    assert_array_equal(four.grid.tl, [0.0, 0.0, 1.0])


def test_except():
    from javelin.fourier import get_ff

    four = Fourier()

    with pytest.raises(ValueError):
        four.radiation = 'electron'

    with pytest.raises(ValueError):
        four.lots = 4, 1

    with pytest.raises(TypeError):
        four.average = 0

    with pytest.raises(TypeError):
        four.magnetic = 0

    with pytest.raises(TypeError):
        four.approximate = 0

    with pytest.raises(TypeError):
        four.calc()

    with pytest.raises(ValueError):
        get_ff(10, 'electron')


def test_Fourier_two_atoms():
    atom = Structure(symbols=['C', 'O'], positions=[(0., 0., 0.), (1., 0., 0.)])
    four = Fourier()
    four.grid.bins = [21, 2]
    four.grid.set_corners(lr=[2.0, 0.0, 0.0])

    expected_result = [155.08717156, 140.34558984, 101.75162784,
                       54.04686728, 15.45290528, 0.71132356,
                       15.45290528, 54.04686728, 101.75162784,
                       140.34558984, 155.08717156, 140.34558984,
                       101.75162784, 54.04686728, 15.45290528,
                       0.71132356, 15.45290528, 54.04686728,
                       101.75162784, 140.34558984, 155.08717156]

    four.approximate = False

    results = four.calc(atom)
    assert_array_almost_equal(results[:, 0, 0], expected_result)

    four._fast = True
    four._cython = False
    results = four.calc(atom)
    assert_array_almost_equal(results[:, 0, 0], expected_result)

    four._fast = True
    four._cython = True
    results = four.calc(atom)
    assert_array_almost_equal(results[:, 0, 0], expected_result)

    four.radiation = 'xray'
    expected_result = [1.95913322e+02, 1.66402201e+02, 1.01484495e+02,
                       4.32239232e+01, 1.19068361e+01, 3.45460097e+00,
                       6.43534873e+00, 1.22093365e+01, 1.64467400e+01,
                       1.77365894e+01, 1.61657550e+01, 1.25362565e+01,
                       8.00992867e+00, 3.83255349e+00, 1.01694147e+00,
                       6.55602457e-02, 8.51717961e-01, 2.71350261e+00,
                       4.72586766e+00, 6.04617986e+00, 6.20124769e+00]

    results = four.calc(atom)
    assert_array_almost_equal(results[:, 0, 0], expected_result)

    four._fast = True
    four._cython = False
    results = four.calc(atom)
    assert_array_almost_equal(results[:, 0, 0], expected_result)

    four._fast = True
    four._cython = True
    results = four.calc(atom)
    assert_array_almost_equal(results[:, 0, 0], expected_result)

    # Should skip invalid atom Z
    four = Fourier()
    atom = Structure(symbols=['C'], positions=[(0., 0., 0.)])
    atom.atoms.Z = 1000
    results = four.calc(atom)
    assert_array_equal(results, np.zeros((101, 101, 1)))


def test_Foutier_C_Ring():
    cnt = Structure(symbols=['C']*12, unitcell=(8.02140913, 8.02140913, 2.42487113),
                    positions=[[0.75,       0.5,        0.25],
                               [0.69151111, 0.6606969,  0.25],
                               [0.625,      0.71650635, 0.75],
                               [0.45658796, 0.74620194, 0.75],
                               [0.375,      0.71650635, 0.25],
                               [0.26507684, 0.58550504, 0.25],
                               [0.25,       0.5,        0.75],
                               [0.30848889, 0.3393031,  0.75],
                               [0.375,      0.28349365, 0.25],
                               [0.54341204, 0.25379806, 0.25],
                               [0.625,      0.28349365, 0.75],
                               [0.73492316, 0.41449496, 0.75]])
    four = Fourier()
    four.grid.bins = [6, 6, 2]
    four.grid.set_corners(ll=[0.0, 0.0, 0.0],
                          lr=[2.0, 0.0, 0.0],
                          ul=[0.0, 2.0, 0.0],
                          tl=[0.0, 0.0, 0.5])
    four.approximate = False
    expected_result = [[[6.36497605e+03, 3.18248802e+03],
                        [5.19826149e+03, 2.62430530e+03],
                        [2.62792897e+03, 1.44984089e+03],
                        [5.38964665e+02, 4.77287745e+02],
                        [1.77345695e+01, 5.96777314e+01],
                        [5.61330978e+02, 2.13239655e-01]],
                       [[5.19824633e+03, 2.58464387e+03],
                        [4.19804025e+03, 2.08045238e+03],
                        [2.02566353e+03, 1.13472923e+03],
                        [3.43108617e+02, 3.82910389e+02],
                        [4.89415326e+01, 6.29938444e+01],
                        [5.92919274e+02, 4.58411202e+00]],
                       [[2.62726845e+03, 1.23823941e+03],
                        [2.02670888e+03, 8.48260647e+02],
                        [7.91958685e+02, 3.42380363e+02],
                        [3.33463043e+01, 5.27598283e+01],
                        [2.30729879e+02, 5.28798082e-01],
                        [7.98830713e+02, 8.58597753e+00]],
                       [[5.35796488e+02, 1.75090464e+02],
                        [3.44314604e+02, 3.58530907e+01],
                        [3.54586437e+01, 1.01212697e+01],
                        [1.11489247e+02, 1.22750858e+02],
                        [6.43708532e+02, 2.21345294e+02],
                        [1.09996108e+03, 1.70969791e+02]],
                       [[2.07782598e+01, 8.83691594e+01],
                        [4.95216290e+01, 3.06779385e+02],
                        [2.12904313e+02, 6.20110108e+02],
                        [6.00173980e+02, 8.55469020e+02],
                        [1.07400513e+03, 8.34634177e+02],
                        [1.26315718e+03, 5.49870647e+02]],
                       [[6.17667281e+02, 7.27129707e+02],
                        [6.16759232e+02, 1.22358914e+03],
                        [7.37019696e+02, 1.60499641e+03],
                        [9.37950983e+02, 1.66971853e+03],
                        [1.09292598e+03, 1.35970021e+03],
                        [1.03004758e+03, 8.27690735e+02]]]
    results = four.calc(cnt)
    assert_array_almost_equal(results, expected_result, 5)
    four._cython = False
    results = four.calc(cnt)
    assert_array_almost_equal(results, expected_result, 5)
    four._fast = False
    results = four.calc(cnt)
    assert_array_almost_equal(results, expected_result, 5)

    four.radiation = 'xray'

    expected_result = [[[5.17915927e+03, 1.84520226e+03],
                        [4.14079683e+03, 1.49403563e+03],
                        [1.96612817e+03, 7.82168405e+02],
                        [3.64548911e+02, 2.36155155e+02],
                        [1.04927182e+01, 2.63280310e+01],
                        [2.82969419e+02, 8.20150027e-02]],
                       [[4.14078476e+03, 1.47145610e+03],
                        [3.27428489e+03, 1.16317490e+03],
                        [1.48473429e+03, 6.01482538e+02],
                        [2.27561202e+02, 1.86296052e+02],
                        [2.84272873e+01, 2.73553635e+01],
                        [2.93854508e+02, 1.73761591e+00]],
                       [[1.96563399e+03, 6.68012436e+02],
                        [1.48550048e+03, 4.49634990e+02],
                        [5.46390863e+02, 1.72307091e+02],
                        [2.08728021e+01, 2.44270522e+01],
                        [1.26929387e+02, 2.19188322e-01],
                        [3.76562850e+02, 3.11777720e+00]],
                       [[3.62405996e+02, 8.66322593e+01],
                        [2.28361053e+02, 1.74434788e+01],
                        [2.21950009e+01, 4.68600433e+00],
                        [6.35847874e+01, 5.24769977e+01],
                        [3.24496306e+02, 8.51325467e+01],
                        [4.78399877e+02, 5.79418881e+01]],
                       [[1.22935279e+01, 3.89858313e+01],
                        [2.87642316e+01, 1.33220343e+02],
                        [1.17123166e+02, 2.57037419e+02],
                        [3.02550346e+02, 3.29025546e+02],
                        [4.82189861e+02, 2.90671251e+02],
                        [4.93736251e+02, 1.70033011e+02]],
                       [[3.11368797e+02, 2.79664422e+02],
                        [3.05669741e+02, 4.63803665e+02],
                        [3.47425597e+02, 5.82813221e+02],
                        [4.07937737e+02, 5.65869815e+02],
                        [4.27197174e+02, 4.20451470e+02],
                        [3.54323772e+02, 2.29315232e+02]]]

    four._fast = True
    four._cython = True
    results = four.calc(cnt)
    assert_array_almost_equal(results, expected_result, 5)
    four._cython = False
    results = four.calc(cnt)
    assert_array_almost_equal(results, expected_result, 5)
    four._fast = False
    results = four.calc(cnt)
    assert_array_almost_equal(results, expected_result, 5)


def test_lots():
    np.random.seed(42)  # Make random lot selection not random

    structure = Structure(symbols=['C', 'O'], positions=[(0, 0, 0), (0.5, 0, 0)], unitcell=5)
    structure.repeat(5)

    four = Fourier()
    four.grid.bins = [2, 21]
    four.grid.set_corners(lr=[4.0, 0.0, 0.0],
                          ul=[0.0, 2.0, 0.0])
    four.lots = None
    four.approximate = False
    results = four.calc(structure)

    expected_result = [2.42323706e+06, 1.01505872e+06, 1.46887112e-26,
                       1.48095071e+05, 1.16378024e-26, 9.69294822e+04,
                       5.01755975e-26, 1.48095071e+05, 1.35809407e-25,
                       1.01505872e+06, 2.42323706e+06, 1.01505872e+06,
                       6.70205900e-25, 1.48095071e+05, 1.03080955e-25,
                       9.69294822e+04, 5.35400081e-26, 1.48095071e+05,
                       1.77832044e-25, 1.01505872e+06, 2.42323706e+06]
    assert_allclose(results[0, :, 0], expected_result, atol=1e-25)

    four._cython = False
    results = four.calc(structure)
    assert_allclose(results[0, :, 0], expected_result, atol=1e-25)

    four.lots = 3, 3, 3
    four.number_of_lots = 3
    four._cython = True
    results = four.calc(structure)

    expected_result = [523419.204015, 398618.732824, 152258.807386,
                       8485.092521,   22214.260619,  58157.689335,
                       22214.260619,  8485.092521,   152258.807386,
                       398618.732824, 523419.204015, 398618.732824,
                       152258.807386, 8485.092521,   22214.260619,
                       58157.689335,  22214.260619,  8485.092521,
                       152258.807386, 398618.732824, 523419.204015]
    assert_allclose(results[0, :, 0], expected_result)

    # Random move + average subtraction
    structure.rattle(seed=0)
    four.average = True
    results = four.calc(structure)

    expected_result = [0.,           0.0047178662, 0.015698443,
                       0.0362217496, 0.0796665753, 0.1387246122,
                       0.1791635212, 0.19709221,   0.2511439595,
                       0.3821917992, 0.5227045787, 0.5707425005,
                       0.5651595792, 0.6805063491, 0.9763282278,
                       1.2484581803, 1.2733639152, 1.1616699955,
                       1.2711365957, 1.7033907726, 2.0905425265]
    assert_allclose(results[0, :, 0], expected_result)


def test_average():
    structure = Structure(symbols=['C', 'O'],
                          positions=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], unitcell=5)
    structure.repeat(5)

    four = Fourier()
    four.approximate = False
    four.grid.bins = [2, 21]
    four.grid.set_corners(lr=[4.0, 0.0, 0.0],
                          ul=[0.0, 2.0, 0.0])
    four.average = True
    results = four.calc(structure)

    expected_result = [1.32348898e-23, 5.71594728e-24, 8.12875148e-56,
                       8.37681929e-25, 6.37686377e-56, 5.46068451e-25,
                       2.54939667e-55, 8.37681929e-25, 7.16891360e-55,
                       5.21188409e-24, 1.32348898e-23, 5.71594728e-24,
                       3.46656800e-54, 8.37681929e-25, 5.74150325e-55,
                       5.46068451e-25, 2.74171118e-55, 8.37681929e-25,
                       1.00428514e-54, 5.21188409e-24, 1.32348898e-23]

    assert_array_almost_equal(results[0, :, 0], expected_result)

    four._cython = False
    results = four.calc(structure)
    assert_array_almost_equal(results[0, :, 0], expected_result)

    four._fast = False
    results = four.calc(structure)
    assert_array_almost_equal(results[0, :, 0], expected_result)

    # Random move
    structure.rattle(seed=0)
    results = four.calc(structure)

    expected_result = [1.32348898e-23, 8.88939203e-04, 7.27741076e-03,
                       1.91258313e-02, 5.94954973e-02, 1.26500450e-01,
                       1.34462469e-01, 1.04512735e-01, 1.17228978e-01,
                       7.24444566e-02, 5.87435134e-26, 1.06907925e-01,
                       2.60217008e-01, 3.57817285e-01, 7.25525103e-01,
                       1.13842501e+00, 9.60372378e-01, 6.18668588e-01,
                       5.97492048e-01, 3.24844518e-01, 2.34454638e-25]
    assert_array_almost_equal(results[0, :, 0], expected_result)

    four._fast = True
    results = four.calc(structure)
    assert_array_almost_equal(results[0, :, 0], expected_result)

    four._cython = True
    results = four.calc(structure)
    assert_array_almost_equal(results[0, :, 0], expected_result)


def test_magnetic():
    structure = Structure(symbols=['V']*25, positions=np.tile([0.0, 0.0, 0.0], (25, 1)),
                          unitcell=5, ncells=(5, 5, 1, 1), magnetic_moments=True)

    structure.magmons.spinx = 0
    structure.magmons.spiny = 0
    structure.magmons.spinz = 1

    four = Fourier()
    four.grid.bins = [11, 2]
    four.magnetic = True
    four.grid.set_corners(lr=[1.0, 0.0, 0.0],
                          ul=[0.0, 0.5, 0.0])

    expected_result = [[np.nan, 2.35532329e+01],
                       [2.61228206e+02, 9.84260348e+00],
                       [3.17478196e-30, 9.20193707e-32],
                       [3.73888434e+01, 1.40890591e+00],
                       [2.37298181e-30, 1.11786157e-31],
                       [2.35532329e+01, 8.87748695e-01],
                       [1.31222666e-29, 3.94457437e-31],
                       [3.39894870e+01, 1.28153986e+00],
                       [2.85072172e-29, 4.81542884e-31],
                       [2.16003995e+02, 8.14786919e+00],
                       [4.93162100e+02, 1.86074202e+01]]

    results = four.calc(structure)
    assert_array_almost_equal(results[:, :, 0], expected_result)
    four._fast = False
    results = four.calc(structure)
    assert_array_almost_equal(results[:, :, 0], expected_result)

    structure.magmons.spinz[1::2] = -1

    expected_result = [[np.nan, 2.35532329e+01],
                       [0.00000000e+00, 8.57288301e-31],
                       [1.51357800e+00, 3.56445825e+01],
                       [9.44108954e-31, 1.93226489e-29],
                       [1.00804339e+01, 2.37433966e+02],
                       [2.35532329e+01, 5.54842934e+02],
                       [9.61093588e+00, 2.26440052e+02],
                       [6.69067886e-31, 1.98917474e-29],
                       [1.31214770e+00, 3.09272932e+01],
                       [2.94920500e-31, 8.84219640e-30],
                       [7.89059360e-01, 1.86074202e+01]]

    results = four.calc(structure)
    assert_array_almost_equal(results[:, :, 0], expected_result)
    four._fast = True
    results = four.calc(structure)
    assert_array_almost_equal(results[:, :, 0], expected_result)

    # Should skip missing magnetic form factor
    structure = Structure(numbers=[8], positions=[[0.0, 0.0, 0.0]], magnetic_moments=True)
    structure.magmons.spinx = 0
    structure.magmons.spiny = 0
    structure.magmons.spinz = 1

    results = four.calc(structure)

    assert_array_equal(results[:, :, 0], [[np.nan, 0],
                                          [0, 0],
                                          [0, 0],
                                          [0, 0],
                                          [0, 0],
                                          [0, 0],
                                          [0, 0],
                                          [0, 0],
                                          [0, 0],
                                          [0, 0],
                                          [0, 0]])
